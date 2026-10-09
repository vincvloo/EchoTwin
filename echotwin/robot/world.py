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
from . import robots as RB
from .scene import Layout, build_xml

CTRL_HZ = 20
CTRL_DT = 1.0 / CTRL_HZ
VMAX = 0.25                 # tool speed, m/s
JOINT_SPEED = 2.5           # rad/s, how fast the servos may move their setpoints
Z_MIN = 0.004               # lowest tool height above the table
BASE_INSET = 0.07           # the arm's base stands this far in from the front edge of the table
REST_RISE = 0.14            # resting tool height
TILT_RATE = 1.0             # rad/s, how fast the tool may change how far it leans out (arms with tilts_deg)
FIXED_JAW_GAP = 0.008       # a single jaw: the fixed jaw comes down this far beside the thing, then the moving jaw pushes it over
GRIP_MARGIN = 0.004         # an object must be this much narrower than the opening
STACK_CLEAR = 0.015         # m: a carried object's bottom passes this far above the top it is to be set on
MIN_THICKNESS = 0.015       # thinner than this cannot be pinched from a table (measured, docs/RESULTS.md)


@dataclass
class HandState:
    """Everything about the arm that is not in MjData; copied with it for imagine-first."""
    grip: bool = False
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))   # where the tool is asked to be
    yaw: float = 0.0                                                   # direction the jaws close along
    q: np.ndarray = field(default_factory=lambda: np.zeros(5))        # joint setpoints
    held: str | None = None                                            # the object the pads squeeze right now
    base: np.ndarray = field(default_factory=lambda: np.zeros(3))      # where the base stands: x, y, turn (a fixed one never moves)
    drive: np.ndarray | None = None                                    # where a mobile base is driving to, or None
    tilt: float = 0.0                                                  # how far the tool leans out from straight down (rad)

    @property
    def attached(self) -> str | None:
        return self.held

    def clone(self) -> "HandState":
        return HandState(self.grip, self.target.copy(), self.yaw, self.q.copy(), self.held, self.base.copy(),
                         None if self.drive is None else self.drive.copy(), self.tilt)


class World:
    name = "sim"                      # the back-end this is (see backend.py)

    def __init__(self, layout: Layout | None = None, arm: "A.ArmSpec | str | None" = None, robot: "RB.RobotSpec | str | None" = None):
        self.arm = arm if isinstance(arm, A.ArmSpec) else A.load(arm)
        self.robot = robot if isinstance(robot, RB.RobotSpec) else RB.load(robot)     # the base: fixed (default) or mobile
        self.layout = layout or Layout()
        self.build(self.layout)

    # ---------- model ----------
    def build(self, layout: Layout):
        new_layout = layout.copy()
        base = self.base_xy(new_layout)
        spec = mujoco.MjSpec.from_string(build_xml(new_layout))
        A.compose(spec, self.arm, base, self.robot)
        model = spec.compile()                      # raises before anything changes
        self.layout = new_layout
        self.model = model
        self.data = mujoco.MjData(model)
        m = model
        pre = self.arm.prefix
        self.ik = A.ArmIK(m, self.arm, base, self.robot.mount_height if self.robot.mobile else 0.0)
        self.base_act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in A.BASE_JOINTS] if self.ik.base_qadr else []
        self.hand = HandState(base=np.array([base[0], base[1], 0.0]))
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
        for a, act, v in zip(self.ik.base_qadr, self.base_act, self.hand.base):
            self.data.qpos[a], self.data.ctrl[act] = v, v
        mujoco.mj_forward(m, self.data)
        self.ik.seed_down((0.0, 0.22))
        self.workspace = A.Workspace.for_arm(self.ik, self.arm)
        self._palms = {}
        self.go_rest(teleport=True)

    # ---------- where the base stands ----------
    @property
    def mobile(self) -> bool:
        return self.robot.mobile

    @property
    def base(self) -> np.ndarray:
        """Where the arm's base stands (x, y) right now."""
        return self.hand.base[:2]

    @staticmethod
    def _rel(xy, pose) -> np.ndarray:
        """A table point seen from a base standing at `pose` (x, y, turn): ahead is +y."""
        d = np.asarray(xy, float)[:2] - pose[:2]
        if pose[2] == 0.0:
            return d
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    @staticmethod
    def _abs(rel, pose) -> np.ndarray:
        """The other way round: a point seen from the base -> table coordinates."""
        rel = np.asarray(rel, float)[:2]
        if pose[2] == 0.0:
            return rel + pose[:2]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]]) + pose[:2]

    def standoff(self, xy, ignore: str | None = None, near=None) -> np.ndarray | None:
        """Where a mobile base should stand to work on `xy`: facing it, at the middle of the arm's reach, its footprint
        clear of the things on the table and the furniture (`ignore`: the object it works on). The spot nearest to
        `near` (default: where the base is) comes first. None when there is no free spot."""
        xy = np.asarray(xy, float)[:2]
        near = self.hand.base[:2] if near is None else np.asarray(near, float)[:2]
        body = 0.5 * float(np.hypot(*self.robot.footprint)) + 0.02
        front = 0.5 * self.robot.footprint[1] + 0.05           # the arm cannot work under its own base
        r = float(np.clip(0.5 * (self.workspace.r_min + self.workspace.r_max), front, self.workspace.r_max - 0.02))
        home = float(np.arctan2(*(near - xy)[::-1]))
        for k in sorted(range(24), key=lambda k: abs(np.angle(np.exp(1j * (2 * np.pi * k / 24 - home))))):
            a = 2 * np.pi * k / 24
            spot = xy + r * np.array([np.cos(a), np.sin(a)])
            clear = all(np.hypot(*(self.obj_pos(n)[:2] - spot)) > body + self.radius(n)
                        for n in self.things() if n != ignore)
            clear = clear and all(np.hypot(*(np.asarray(o["pos"], float) - spot)) > body + max(o["size"][:2])
                                  for o in self.layout.obstacles)
            if clear:
                d = xy - spot
                return np.array([spot[0], spot[1], float(np.arctan2(-d[0], d[1]))])
        return None

    def _overlap(self, xy, ignore: str | None = None) -> float:
        """How far a base standing at `xy` cuts into the things and the furniture (0: it touches nothing)."""
        body = 0.5 * float(np.hypot(*self.robot.footprint)) + 0.02
        cuts = [body + self.radius(n) - float(np.hypot(*(self.obj_pos(n)[:2] - xy))) for n in self.things() if n != ignore]
        cuts += [body + max(o["size"][:2]) - float(np.hypot(*(np.asarray(o["pos"], float) - xy))) for o in self.layout.obstacles]
        return max([0.0, *cuts])

    def drive_by(self, forward: float, sideways: float, turn: float, hs: HandState | None = None) -> bool:
        """Drive a mobile base by hand for one control period: forward, sideways and turn are -1..1 of its speed and
        turn rate, in the base's own frame. The arm holds its joints and rides along. It does not drive further into
        a thing or off the mapped area (plus half a metre); returns False when it stopped for that."""
        hs = hs if hs is not None else self.hand
        if not self.mobile:
            return False
        f, sd, t = (float(np.clip(v, -1, 1)) for v in (forward, sideways, turn))
        step = self.robot.speed * CTRL_DT
        pose = hs.base.copy()
        xy = self._abs((sd * step, f * step), pose)
        yaw = pose[2] + t * self.robot.turn * CTRL_DT
        th = self.layout.table_half
        inside = abs(xy[0]) <= th[0] + 0.5 and abs(xy[1]) <= th[1] + 0.5
        worse = self._overlap(xy, hs.held) > self._overlap(pose[:2], hs.held) + 1e-9
        if not inside or worse:
            if t == 0.0:
                return False
            xy = pose[:2]                                   # turning on the spot is always fine
        hs.base = np.array([xy[0], xy[1], yaw])
        self.ik.set_base(hs.base)
        hs.target = self.ik.pose(hs.q)[0]                   # the tool moved with the base
        return bool(inside and not worse)

    def _drive_tick(self, hs: HandState):
        """Move a mobile base one control period towards hs.drive; the arm holds its joints. On arrival the tool target
        is where the tool now is, so the next move starts from there."""
        to = hs.drive
        d = to[:2] - hs.base[:2]
        n = float(np.linalg.norm(d))
        step = self.robot.speed * CTRL_DT
        dyaw = float(np.angle(np.exp(1j * (to[2] - hs.base[2]))))
        turn = self.robot.turn * CTRL_DT
        hs.base = np.array([*(to[:2] if n <= step else hs.base[:2] + d / n * step),
                            to[2] if abs(dyaw) <= turn else hs.base[2] + np.sign(dyaw) * turn])
        if n <= step and abs(dyaw) <= turn:
            hs.drive = None
            self.ik.set_base(hs.base)
            hs.target = self.ik.pose(hs.q)[0]

    def base_xy(self, layout: Layout) -> tuple[float, float]:
        """Where the arm's base stands: at the front edge, a bigger arm a little further in. A mobile base starts just in
        front of the mapped area, so it never starts on top of the things."""
        if self.robot.mobile:
            return (0.0, -layout.table_half[1] - 0.5 * self.robot.footprint[1] - 0.05)
        return (0.0, -layout.table_half[1] + BASE_INSET * self.arm.scale)

    def rest_target(self) -> np.ndarray:
        r = 0.5 * (self.workspace.r_min + self.workspace.r_max)
        return np.array([*self._abs((0.0, r), self.hand.base), REST_RISE])

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
        return self.workspace.reachable(self._rel(xy, self.hand.base), z)

    def reachable_from(self, pose, xy, z: float = 0.02) -> bool:
        """Reachable if the base stood at `pose` (x, y, turn)."""
        return self.workspace.reachable(self._rel(xy, np.asarray(pose, float)), z)

    def ceiling(self, xy, pose=None) -> float:
        """The highest the tool can go above this table point (from a base at `pose`, default where it stands),
        a little below the arm's limit, like carry_height()."""
        pose = self.hand.base if pose is None else np.asarray(pose, float)
        return self.workspace.z_max(float(np.hypot(*self._rel(xy, pose)))) - 0.01

    def stands(self, pick, place, ignore: str | None = None) -> tuple:
        """Where the base stands to pick up at `pick` and to set down at `place`: ((pose, drove there), (pose, drove there)),
        the drive spot None when it does not move. A fixed base always stands where it is."""
        pose = self.hand.base.copy()
        at_pick = at_place = None
        if self.mobile and not self.reachable_from(pose, pick):
            at_pick = self.standoff(pick, ignore=ignore)
            pose = pose if at_pick is None else at_pick
        first = pose
        if self.mobile and not self.reachable_from(pose, place):
            at_place = self.standoff(place, ignore=ignore, near=pose[:2])
            pose = pose if at_place is None else at_place
        return (first, at_pick), (pose, at_place)

    def path_ceiling(self, pick, place, ignore: str | None = None) -> float:
        """The lowest ceiling the tool meets while it carries something from `pick` to `place` (see ceiling())."""
        (p0, _), (p1, at_place) = self.stands(pick, place, ignore)
        pick, place = np.asarray(pick, float)[:2], np.asarray(place, float)[:2]
        if at_place is not None:                       # it drives on the way: lifts over the pick, lowers over the place
            return min(self.ceiling(pick, p0), self.ceiling(place, p1))
        return min(self.ceiling(pick + (place - pick) * f, p0) for f in np.linspace(0.0, 1.0, 5))

    def palm(self, name: str) -> float:
        """How far the tool point can go down over this object's top before a part of the arm above the jaws
        (the palm, the wrist) sits on it: rays up from the object's top, the tool pointing straight down."""
        key = (round(self.grasp_width(name), 4), round(self.radius(name), 4))
        if key in self._palms:
            return self._palms[key]
        m, s = self.model, self.ik.scratch
        q, _, _ = self.ik.solve(self.ik.base + np.array([0.0, 0.5 * (self.workspace.r_min + self.workspace.r_max), 0.15]),
                                0.0, self.ik.q_down)
        jaws = [int(m.jnt_qposadr[m.actuator_trnid[a][0]]) for a in self.gact]
        was = s.qpos[jaws].copy()
        s.qpos[jaws] = self.arm.gripper["open"]                # the jaws open, as they come down over the object
        self.ik.pose(q)                                  # the scratch copy now holds the arm pointing down, jaws along x
        tool = s.site_xpos[self.site].copy()
        centre = tool
        if self.arm.gripper["mode"] == "single":           # the object stands beside the fixed jaw (grasp_offset); here the jaws close along x
            centre = tool - np.array([float(self.arm.gripper.get("fixed_side", 1)) * (self.grasp_width(name) / 2 + FIXED_JAW_GAP), 0, 0])
        r = 0.6 * min(self.radius(name), 0.5 * self.grasp_width(name) + 0.01)
        robot = {b for b in range(m.nbody) if m.body(b).name.startswith(self.arm.prefix)}
        geomid = np.zeros(1, np.int32)
        best = 1.0
        for dx, dy in ((0, 0), (r, 0), (-r, 0), (0, r), (0, -r)):
            start = centre + np.array([dx, dy, 0.0])
            for _ in range(8):                           # skip what is not the robot (objects above, the scene)
                dist = mujoco.mj_ray(m, s, start, np.array([0.0, 0.0, 1.0]), None, 1, -1, geomid)
                if dist < 0:
                    break
                g = int(geomid[0])
                solid = m.geom_contype[g] or m.geom_conaffinity[g]   # visual-only meshes do not stop anything
                if solid and int(m.geom_bodyid[g]) in robot and g not in self.pads:
                    best = min(best, float(start[2] + dist - tool[2]))
                    break
                start = start + np.array([0.0, 0.0, dist + 1e-4])
        s.qpos[jaws] = was
        self._palms[key] = best
        return best

    def stack_hang(self, name: str, held: float = 0.0) -> float:
        """How far below the tool the bottom of `name` hangs, held `held` above its bottom (default: as low as the jaws go).
        A thing taller than the palm sticks up into the hand and is held higher than asked."""
        return max(2 * self.half(name) - self.palm(name), 0.004, held)

    def workspace_sample(self, rng) -> tuple[float, float]:
        """A random reachable spot on the table, in table coordinates."""
        rx, ry = self.workspace.sample(rng)
        x, y = self._abs((rx, ry), self.hand.base)
        return (float(x), float(y))

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
        if not (pr.get("shape") in ("cylinder", "round") or hx <= hy):
            yaw = yaw + np.pi / 2
        if self.arm.gripper["mode"] != "single":
            return float(yaw)
        return self._jaw_side(name, float(yaw))

    def _jaw_side(self, name: str, yaw: float) -> float:
        """A single moving jaw: which way round the arm can hold its jaws over this object. Closing along yaw or along
        yaw + pi is the same line, but it decides on which side the fixed jaw stands, and the plan puts the fixed jaw beside
        the thing (grasp_offset). The wrist cannot always turn half a turn, so ask the arm, and use the way it can do."""
        o = self.obj_pos(name)
        key = (name, round(float(o[0]), 3), round(float(o[1]), 3), round(yaw, 3), tuple(np.round(self.hand.base, 3)))
        cache = self.__dict__.setdefault("_jaw_cache", {})
        if key not in cache:
            if len(cache) > 256:
                cache.clear()
            at = np.array([o[0], o[1], o[2] + 0.03])
            self.ik.set_base(self.hand.base)
            tilt = self.workspace.tilt_for(self._rel(at, self.hand.base), float(at[2]))
            q, _, _ = self.ik.solve(at, yaw, self.ik.q_down, iters=150, tilt=tilt)
            jaws = self.ik.pose(q)[1] @ self.ik.c_axis
            cache[key] = yaw if jaws @ np.array([np.cos(yaw), np.sin(yaw), 0.0]) >= 0 else yaw + np.pi
        return cache[key]

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

    def size_needed(self, name: str) -> dict:
        """How big this arm would have to be to pick this object up where it lies: big enough for the jaws to open around it
        and to reach it, not so big that the object is inside the ring it cannot bend down to. Sizes are relative to the
        arm as described (1.0). `needed` is None when no size works (too thin to pinch, or out of the arm's sweep)."""
        s = float(self.arm.scale)
        opening = float(self.arm.max_opening or 0.08) / s
        width, thick = self.grasp_width(name), 2 * self.half(name)
        r_min, r_max = self.workspace.r_min / s, self.workspace.r_max / s
        base1 = np.asarray(self.base, float)[:2] - np.array([0.0, BASE_INSET * (s - 1)])   # where the base stands at size 1
        obj = self.obj_pos(name)[:2]

        def dist(k):                                        # a bigger arm stands further in (base_xy): distance at size k
            return float(np.hypot(*(obj - base1 - np.array([0.0, BASE_INSET * (k - 1)]))))
        reach, most = dist(1.0) / r_max, dist(1.0) / max(r_min, 1e-6)
        for _ in range(8):                                  # solve k * r_max = dist(k) (and the same for r_min)
            reach, most = dist(reach) / r_max, dist(most) / max(r_min, 1e-6)
        rel = obj - base1
        grasp = (width + GRIP_MARGIN) / opening
        if self.mobile:                                     # the base drives up to it: only the jaws decide
            reach, most = 0.0, 99.0
        k = max(grasp, reach)
        why = ""
        if thick < MIN_THICKNESS:
            why = f"it is only {thick * 1000:.0f} mm thick: no gripper pinches that from a table"
        elif not self.mobile and abs(np.arctan2(rel[0], rel[1])) > self.workspace.pan:
            why = "it is beside the arm, where the base cannot turn"
        elif k > most:
            why = "an arm with jaws that wide could not bend down so close to its base"
        return {"name": self.layout.props[int(name[5:])]["name"], "width_cm": round(width * 100, 1),
                "grasp": round(grasp, 2), "reach": round(reach, 2), "most": round(most, 2),
                "needed": None if why else round(k, 2), "why": why}

    def refusal(self, name: str, goal_xy=None, on: str | None = None) -> str:
        """Why the arm cannot move this object (to this spot, or on top of the object `on`), in plain words; empty when it can."""
        ok, why = self.can_grasp(name)
        if not ok:
            return why
        if self.mobile:
            if self.standoff(self.obj_pos(name)[:2], ignore=name) is None:
                return "there is no free spot for me to stand next to it"
            if goal_xy is not None and self.standoff(goal_xy, ignore=name) is None:
                return "there is no free spot for me to stand next to that place"
        else:
            if not self.reachable(self.obj_pos(name)[:2]):
                return "it is out of my reach"
            if goal_xy is not None and not self.reachable(goal_xy):
                return "the spot is out of my reach"
        if on is not None and goal_xy is not None:
            off = self.grasp_offset(name)[:2]
            top = float(self.obj_pos(on)[2] + self.half(on))
            ceiling = self.path_ceiling(self.obj_pos(name)[:2] + off, np.asarray(goal_xy, float)[:2] + off, name)
            if ceiling - self.stack_hang(name) - top < STACK_CLEAR:
                return f"I can't lift it high enough over the {self.layout.props[int(on[5:])]['name']} from here"
        return ""

    def grasp_offset(self, name: str) -> np.ndarray:
        """Where the tool must stand relative to the object's centre (horizontal). Zero for a parallel gripper;
        a single moving jaw closes against its fixed pad, so the fixed pad has to touch the object's side."""
        if self.arm.gripper["mode"] != "single":
            return np.zeros(3)
        y = self.grasp_yaw(name)
        side = float(self.arm.gripper.get("fixed_side", 1))
        return np.array([np.cos(y), np.sin(y), 0.0]) * side * (self.grasp_width(name) / 2 + FIXED_JAW_GAP)

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
        rel = self.workspace.clamp(np.array([*self._rel(t, hs.base), t[2]]), Z_MIN)
        return np.array([*self._abs(rel[:2], hs.base), rel[2]])

    def command(self, action, hs: HandState | None = None) -> np.ndarray:
        """The controller half of a tick: the wanted tool motion -> joint setpoints (hs.q is updated and returned).
        action = (vx, vy, vz, grip in {0,1}[, yaw of the jaws]); the same for every back-end."""
        hs = hs if hs is not None else self.hand
        a = np.asarray(action, dtype=float)
        if hs.drive is not None:                    # a mobile base is driving: the arm holds still
            hs.grip = bool(a[3] > 0.5)
            self._drive_tick(hs)
            return hs.q
        self.ik.set_base(hs.base)
        v = np.clip(a[:3], -VMAX, VMAX)
        if len(a) > 4:
            hs.yaw = float(a[4])
        hs.target = self._clamp(hs.target + v * CTRL_DT, hs)
        hs.grip = bool(a[3] > 0.5)
        if len(self.arm.tilts) > 1:                 # lean out only as far as this spot needs, a little at a time
            want = self.workspace.tilt_for(self._rel(hs.target, hs.base), float(hs.target[2]))
            hs.tilt += float(np.clip(want - hs.tilt, -TILT_RATE * CTRL_DT, TILT_RATE * CTRL_DT))
        q, _, _ = self.ik.solve(hs.target, hs.yaw, hs.q, tilt=hs.tilt)
        hs.q = hs.q + np.clip(q - hs.q, -JOINT_SPEED * CTRL_DT, JOINT_SPEED * CTRL_DT)
        return hs.q

    def advance(self, d, hs: HandState, q=None, grip=None):
        """The physics half of a tick: servo targets (`q`, default hs.q; `grip` closed or not, default hs.grip) into
        the actuators, step the world for one control period, and find out what the pads hold."""
        q = hs.q if q is None else q
        grip = hs.grip if grip is None else grip
        for act, val in zip(self.act, q):
            d.ctrl[act] = val
        for act, val in zip(self.base_act, hs.base):
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
        """Where to hold the tool to look at the table. Folded in just in front of the base; with `away_from` (x, y), held high
        out to the side of the base opposite to that spot, so the arm does not hide what is looked at or pass low over it."""
        r = self.workspace.r_min + 0.03
        if away_from is None:
            return np.array([*self._abs((0.0, r), self.hand.base), 0.06])
        side = -1.0 if self._rel(away_from, self.hand.base)[0] >= 0 else 1.0   # high and to the side: nothing low to sweep through
        return np.array([*self._abs((side * 0.20, 0.10), self.hand.base), self.carry_height()])

    def observe(self, name: str):
        """Look at an object: its table position (x, y), or None when it cannot be seen. The simulation is its own truth."""
        return self.obj_pos(name)[:2].copy()

    def see_tool(self):
        """Where the tool really is, by looking: the simulation's joints are exact, so it is where they say."""
        return self.hand_pos()[:2].copy()

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
        hs.tilt = 0.0
        if teleport:
            self.ik.set_base(hs.base)
            for a, v in zip(self.ik.base_qadr, hs.base):
                d.qpos[a] = v
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
