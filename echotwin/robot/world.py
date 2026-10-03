"""World: MuJoCo model with an arm, the objects on the table, stepping, and grasping by contact.

The arm is described by a small JSON (`arm.py`): the policy still says "tool to (x, y, z), jaws closed or open",
the arm's inverse kinematics turns that into joint targets, the servos follow with limited speed and force, and
an object is held only if the pads really squeeze it. It can slip, drop, tip over, or be too wide for the jaws.
"""
import copy
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import arm as A
from .scene import Layout, build_xml

CTRL_HZ = 20
CTRL_DT = 1.0 / CTRL_HZ
VMAX = 0.25                 # tool speed, m/s
JOINT_SPEED = 2.5           # rad/s, how fast the servos may move their setpoints
Z_MIN = 0.004               # lowest tool height above the table
BASE_INSET = 0.07           # the arm's base stands this far in from the front edge of the table
REST_RISE = 0.14            # resting tool height
GRIP_MARGIN = 0.004         # an object must be this much narrower than the opening
MIN_THICKNESS = 0.015       # thinner than this cannot be pinched from a table (measured, docs/RESULTS.md)


@dataclass
class HandState:
    """Everything about the arm that is not in MjData; copied with it for imagine-first."""
    grip: bool = False
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))   # where the tool is asked to be
    yaw: float = 0.0                                                   # direction the jaws close along
    q: np.ndarray = field(default_factory=lambda: np.zeros(5))        # joint setpoints
    held: str | None = None                                            # the object the pads squeeze right now

    @property
    def attached(self) -> str | None:
        return self.held

    def clone(self) -> "HandState":
        return HandState(self.grip, self.target.copy(), self.yaw, self.q.copy(), self.held)


class World:
    name = "sim"                      # the back-end this is (see backend.py)

    def __init__(self, layout: Layout | None = None, arm: "A.ArmSpec | str | None" = None):
        self.arm = arm if isinstance(arm, A.ArmSpec) else A.load(arm)
        self.layout = layout or Layout()
        self.build(self.layout)

    # ---------- model ----------
    def build(self, layout: Layout):
        new_layout = layout.copy()
        base = self.base_xy(new_layout)
        spec = mujoco.MjSpec.from_string(build_xml(new_layout))
        A.compose(spec, self.arm, base)
        model = spec.compile()                      # raises before anything changes
        self.layout = new_layout
        self.model = model
        self.data = mujoco.MjData(model)
        m = model
        pre = self.arm.prefix
        self.base = np.array(base)
        self.ik = A.ArmIK(m, self.arm, base)
        self.act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, pre + n) for n in self.arm.actuators]
        self.gact = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, pre + n) for n in self.arm.gripper["actuators"]]
        self.gdof = [int(m.jnt_dofadr[m.actuator_trnid[a][0]]) for a in self.gact]
        self.site = self.ik.site
        self.pads = {g for g in range(m.ngeom) if m.geom(g).name.startswith(pre)
                     and any(p in m.geom(g).name for p in self.arm.pads)}
        self.obj_qadr, self.obj_dadr, self.obj_body = {}, {}, {}
        for i in range(len(self.layout.props)):     # everyday objects are free bodies
            n = f"prop_{i}"
            self.obj_qadr[n] = m.jnt_qposadr[m.joint(n).id]
            self.obj_dadr[n] = m.jnt_dofadr[m.joint(n).id]
            self.obj_body[n] = m.body(n).id
        self._body_to_prop = {b: n for n, b in self.obj_body.items()}
        for a, v in zip(self.ik.qadr, self.arm.home):
            self.data.qpos[a] = v
        mujoco.mj_forward(m, self.data)
        self.ik.seed_down((0.0, 0.22))
        self.workspace = A.Workspace.for_arm(self.ik, self.arm)
        self.hand = HandState()
        self.go_rest(teleport=True)

    @staticmethod
    def base_xy(layout: Layout) -> tuple[float, float]:
        return (0.0, -layout.table_half[1] + BASE_INSET)

    def rest_target(self) -> np.ndarray:
        r = 0.5 * (self.workspace.r_min + self.workspace.r_max)
        return np.array([self.base[0], self.base[1] + r, REST_RISE])

    def half(self, name: str) -> float:
        """Half height (grasp height above the table)."""
        return self.layout.props[int(name[5:])]["size"][2]

    def radius(self, name: str) -> float:
        hx, hy, _ = self.layout.props[int(name[5:])]["size"]
        return max(hx, hy)

    def tallest(self) -> float:
        """Height of the tallest thing on the table: objects and fixed obstacles."""
        hs = [2 * self.half(n) for n in self.things()] + [2 * o["size"][2] for o in self.layout.obstacles]
        return max(hs, default=0.0)

    def things(self) -> list[str]:
        """Everything the gripper might pick up."""
        return [f"prop_{i}" for i in range(len(self.layout.props))]

    def reset(self, layout: Layout | None = None):
        self.build(layout or self.layout)

    # ---------- what the arm can do ----------
    def reachable(self, xy, z: float = 0.02) -> bool:
        return self.workspace.reachable(np.asarray(xy, float) - self.base, z)

    def workspace_sample(self, rng) -> tuple[float, float]:
        """A random reachable spot on the table, in table coordinates."""
        rx, ry = self.workspace.sample(rng)
        return (float(self.base[0] + rx), float(self.base[1] + ry))

    def grasp_width(self, name: str) -> float:
        """How far the jaws must open: the object's narrower horizontal side."""
        hx, hy, _ = self.layout.props[int(name[5:])]["size"]
        shape = self.layout.props[int(name[5:])].get("shape")
        return 2 * ((hx + hy) / 2 if shape == "cylinder" else min(hx, hy))

    def grasp_yaw(self, name: str) -> float:
        """Direction to close the jaws along: across the narrower side (object axes, plus its own rotation)."""
        pr = self.layout.props[int(name[5:])]
        hx, hy, _ = pr["size"]
        yaw = np.radians(float(pr.get("yaw", 0.0)))
        if pr.get("shape") in ("cylinder", "round") or hx <= hy:
            return float(yaw)
        return float(yaw + np.pi / 2)

    def auto_yaw(self, xy) -> float:
        """Jaw direction for a hand steered by a person: aligned with the nearest object within 10 cm, else unchanged."""
        best, yaw = 0.10, self.hand.yaw
        for n in self.things():
            dist = float(np.hypot(*(self.obj_pos(n)[:2] - np.asarray(xy)[:2])))
            if dist < best:
                best, yaw = dist, self.grasp_yaw(n)
        return yaw

    def can_grasp(self, name: str) -> tuple[bool, str]:
        """(ok, the reason in plain words when not)."""
        w = self.grasp_width(name)
        opening = float(self.arm.max_opening or 0.08)
        h = 2 * self.half(name)
        if w > opening - GRIP_MARGIN:
            return False, f"it is {w * 100:.0f} cm wide and my gripper opens {opening * 100:.0f} cm"
        if h < MIN_THICKNESS:
            return False, f"it is only {h * 1000:.0f} mm thick, too thin to pick up from the table"
        return True, ""

    def refusal(self, name: str, goal_xy=None) -> str:
        """Why the arm cannot move this object (to this spot), in plain words; empty when it can."""
        ok, why = self.can_grasp(name)
        if not ok:
            return why
        if not self.reachable(self.obj_pos(name)[:2]):
            return "it is out of my reach"
        if goal_xy is not None and not self.reachable(goal_xy):
            return "the spot is out of my reach"
        return ""

    def grasp_offset(self, name: str) -> np.ndarray:
        """Where the tool must stand relative to the object's centre (horizontal). Zero for a parallel gripper;
        a single moving jaw closes against its fixed pad, so the fixed pad has to touch the object's side."""
        if self.arm.gripper["mode"] != "single":
            return np.zeros(3)
        y = self.grasp_yaw(name)
        side = float(self.arm.gripper.get("fixed_side", 1))
        return np.array([np.cos(y), np.sin(y), 0.0]) * side * (self.grasp_width(name) / 2 + 0.003)

    # ---------- queries ----------
    def hand_pos(self, d=None) -> np.ndarray:
        d = d if d is not None else self.data
        return d.site_xpos[self.site].copy()

    def obj_pos(self, name: str, d=None) -> np.ndarray:
        d = d if d is not None else self.data
        a = self.obj_qadr[name]
        return d.qpos[a:a + 3].copy()

    def grip_settled(self, d=None) -> bool:
        d = d if d is not None else self.data
        return bool(np.abs(d.qvel[self.gdof]).max() < 0.05)

    def _held(self, d) -> str | None:
        """The object the pads squeeze: touched by pads of two different parts of the gripper."""
        touching: dict[str, set] = {}
        gb = self.model.geom_bodyid
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if (a in self.pads) == (b in self.pads):
                continue
            pad, other = (a, b) if a in self.pads else (b, a)
            n = self._body_to_prop.get(int(gb[other]))
            if n:
                touching.setdefault(n, set()).add(int(gb[pad]))
        for n, parts in touching.items():
            if len(parts) >= 2:
                return n
        return None

    # ---------- stepping ----------
    def clone(self):
        return copy.copy(self.data), self.hand.clone()

    def _clamp(self, target: np.ndarray, hs: HandState) -> np.ndarray:
        """Keep the wanted tool position where the arm can go: on the table, inside its reach."""
        th = self.layout.table_half
        t = np.array([np.clip(target[0], -th[0] + 0.02, th[0] - 0.02), np.clip(target[1], -th[1] + 0.02, th[1] - 0.02), target[2]])
        rel = self.workspace.clamp(t - np.array([self.base[0], self.base[1], 0.0]), Z_MIN)
        return rel + np.array([self.base[0], self.base[1], 0.0])

    def command(self, action, hs: HandState | None = None) -> np.ndarray:
        """The controller half of a tick: the wanted tool motion -> joint setpoints (hs.q is updated and returned).
        action = (vx, vy, vz, grip in {0,1}[, yaw of the jaws]); the same for every back-end."""
        hs = hs if hs is not None else self.hand
        a = np.asarray(action, dtype=float)
        v = np.clip(a[:3], -VMAX, VMAX)
        if len(a) > 4:
            hs.yaw = float(a[4])
        hs.target = self._clamp(hs.target + v * CTRL_DT, hs)
        hs.grip = bool(a[3] > 0.5)
        q, _, _ = self.ik.solve(hs.target, hs.yaw, hs.q)
        hs.q = hs.q + np.clip(q - hs.q, -JOINT_SPEED * CTRL_DT, JOINT_SPEED * CTRL_DT)
        return hs.q

    def advance(self, d, hs: HandState, q=None, grip=None):
        """The physics half of a tick: servo targets (`q`, default hs.q; `grip` closed or not, default hs.grip) into
        the actuators, step the world for one control period, and find out what the pads hold."""
        q = hs.q if q is None else q
        grip = hs.grip if grip is None else grip
        for act, val in zip(self.act, q):
            d.ctrl[act] = val
        key = "closed" if grip else "open"
        for act, val in zip(self.gact, self.arm.gripper[key]):
            d.ctrl[act] = val
        for _ in range(max(1, int(round(CTRL_DT / self.model.opt.timestep)))):
            mujoco.mj_step(self.model, d)
        hs.held = self._held(d) if grip else None

    def step(self, action, d=None, hs: HandState | None = None):
        """Advance one control tick. action = (vx, vy, vz, grip in {0,1}[, yaw of the jaws])."""
        d = d if d is not None else self.data
        hs = hs if hs is not None else self.hand
        self.command(action, hs)
        self.advance(d, hs)

    # ---------- the robot contract (backend.py): the rest of it ----------
    @property
    def twin(self) -> "World":
        """The simulation that plans and imagines. For the sim back-end that is the world itself."""
        return self

    @property
    def view(self) -> "World":
        """The world to draw on the dashboard: what the robot is really doing."""
        return self

    def enable(self):
        """Let the arm move (a real arm waits for this before its first move)."""

    def stop(self):
        """Emergency stop. The simulation has nothing to switch off (Sim halts the ticks)."""

    def resume(self):
        pass

    def close(self):
        pass

    def arm_ready(self) -> bool:
        return True

    def observe_pose(self, away_from=None) -> np.ndarray:
        """Where to hold the tool to look at the table: folded in beside the base, out of the way. With `away_from` (x, y),
        on the side of the base opposite to that spot, so the arm is not next to what is being looked at."""
        r = self.workspace.r_min + 0.03
        if away_from is None:
            return np.array([self.base[0], self.base[1] + r, 0.06])
        side = -1.0 if away_from[0] >= self.base[0] else 1.0
        return np.array([self.base[0] + side * r, self.base[1] + 0.03, 0.06])

    def observe(self, name: str):
        """Look at an object: its table position (x, y), or None when it cannot be seen. The simulation is its own truth."""
        return self.obj_pos(name)[:2].copy()

    def set_obj_pose(self, name: str, xy, d=None):
        """Move an object to (x, y) at its current height: used to correct what the robot believes. Velocity is cleared."""
        d = d if d is not None else self.data
        a, v = self.obj_qadr[name], self.obj_dadr[name]
        d.qpos[a:a + 2] = np.asarray(xy, float)[:2]
        d.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(self.model, d)

    def nudge(self, name: str, dxy):
        """Push an object by (dx, dy) metres, as if somebody moved it (for tests and the disturbance check)."""
        self.set_obj_pose(name, self.obj_pos(name)[:2] + np.asarray(dxy, float)[:2])

    def tilt(self, name: str, d=None) -> float:
        """Angle (degrees) between an object's up axis and the world's."""
        d = d if d is not None else self.data
        w, x, y, z = d.qpos[self.obj_qadr[name] + 3:self.obj_qadr[name] + 7]
        return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))

    def carry_height(self) -> float:
        """The highest the tool can carry something over the table, a little below the arm's ceiling."""
        ws = self.workspace
        return float(max(ws.HEIGHTS[ws.ok.any(axis=0)].max(), 0.06)) - 0.01

    def settle(self, ticks: int = 20):
        for _ in range(ticks):
            self.step((0, 0, 0, 1 if self.hand.grip else 0))

    # ---------- resting pose ----------
    def go_rest(self, d=None, hs: HandState | None = None, teleport: bool = False):
        """Tool above the middle of the reachable ring, jaws open. With `teleport`, the arm is simply put there."""
        d = d if d is not None else self.data
        hs = hs if hs is not None else self.hand
        hs.target = self.rest_target()
        hs.grip = False
        hs.held = None
        if teleport:
            q, _, _ = self.ik.solve(hs.target, 0.0, self.ik.q_down, iters=300)
            hs.q = q.copy()
            hs.yaw = 0.0
            for adr, val in zip(self.ik.qadr, q):
                d.qpos[adr] = val
            for act, val in zip(self.act, q):
                d.ctrl[act] = val
            for act, val in zip(self.gact, self.arm.gripper["open"]):
                d.ctrl[act] = val
            mujoco.mj_forward(self.model, d)
            for _ in range(40):
                mujoco.mj_step(self.model, d)


__all__ = ["World", "HandState", "CTRL_DT", "CTRL_HZ", "VMAX"]
