"""The robot arm of the simulation, described by a small JSON so it can be swapped for the arm you have.

An *arm descriptor* (`echotwin/robot/arms/<name>.json`) says where the arm's MJCF file is, which of its joints are the
arm and which actuators drive the gripper, where the tool point is (between the pads) and which of its axes point
down and close. `compose` attaches the arm to the scene with MuJoCo's `MjSpec`; `ArmIK` turns "put the tool here,
pointing down, jaws across this direction" into joint targets.

    ARM=builtin              a small parallel-jaw arm written in this file (default; works offline)
    ARM=so_arm100            the SO-ARM100 from MuJoCo Menagerie (python -m echotwin.robot.arm --download so_arm100)
    ARM=path/to/arm.json     your own (see docs/ARMS.md)

    python -m echotwin.robot.arm --list | --download so_arm100 | --check [name]
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

import mujoco

REPO = Path(__file__).resolve().parents[2]
ARMS_DIR = Path(__file__).with_name("arms")
PREFIX = "arm_"
BUILTIN_MJCF = """
<mujoco model="builtin_arm">
  <compiler angle="radian"/>
  <default>
    <joint damping="0.3" armature="0.01"/>
    <geom contype="1" conaffinity="1" friction="1 0.01 0.001" rgba="0.82 0.84 0.88 1"/>
    <position kp="60" kv="3" forcerange="-4 4"/>
  </default>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.045 0.015" pos="0 0 0.015" rgba="0.25 0.27 0.3 1"/>
      <body name="turret" pos="0 0 0.03">
        <joint name="pan" axis="0 0 1" range="-1.9 1.9"/>
        <geom type="cylinder" size="0.03 0.035" pos="0 0 0.035" rgba="0.3 0.32 0.36 1"/>
        <body name="upper" pos="0 0 0.07">
          <joint name="lift" axis="-1 0 0" range="-0.35 1.75"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.16" size="0.016"/>
          <body name="fore" pos="0 0 0.16">
            <joint name="elbow" axis="-1 0 0" range="-0.1 2.9"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.18" size="0.013"/>
            <body name="wrist" pos="0 0 0.18">
              <joint name="wflex" axis="-1 0 0" range="-2.6 2.6"/>
              <geom type="box" size="0.02 0.016 0.014" pos="0 0 0.012" rgba="0.3 0.32 0.36 1"/>
              <body name="hand" pos="0 0 0.03">
                <joint name="roll" axis="0 0 1" range="-3.0 3.0"/>
                <geom type="box" size="0.03 0.016 0.012" pos="0 0 0.012" rgba="0.3 0.32 0.36 1"/>
                <body name="fl" pos="-0.045 0 0.03">
                  <joint name="fl" type="slide" axis="1 0 0" range="0 0.047"/>
                  <geom name="pad_l" type="box" size="0.004 0.012 0.025" pos="0 0 0.025" friction="1.6 0.02 0.002" condim="4" rgba="0.15 0.15 0.15 1"/>
                </body>
                <body name="fr" pos="0.045 0 0.03">
                  <joint name="fr" type="slide" axis="-1 0 0" range="0 0.047"/>
                  <geom name="pad_r" type="box" size="0.004 0.012 0.025" pos="0 0 0.025" friction="1.6 0.02 0.002" condim="4" rgba="0.15 0.15 0.15 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="pan" joint="pan"/>
    <position name="lift" joint="lift"/>
    <position name="elbow" joint="elbow"/>
    <position name="wflex" joint="wflex"/>
    <position name="roll" joint="roll"/>
    <position name="fl" joint="fl" kp="400" kv="12" forcerange="-14 14" ctrlrange="0 0.06"/>
    <position name="fr" joint="fr" kp="400" kv="12" forcerange="-14 14" ctrlrange="0 0.06"/>
  </actuator>
</mujoco>
"""


class ArmError(ValueError):
    pass


@dataclass
class ArmSpec:
    name: str
    mjcf: str                       # a path (relative to the repo) or "builtin"
    base_body: str
    joints: list[str]               # the arm joints, base to wrist
    actuators: list[str]            # their position actuators, same order
    gripper: dict                   # {"mode": "parallel"|"single", "actuators": [...], "open": [...], "closed": [...], "fixed_side": 1}
    tool: dict                      # {"body", "pos", "point_axis", "close_axis"}
    pads: list[str]                 # substrings of the pad geom names
    home: list[float]               # joint values to start from
    prefix: str = PREFIX
    mount_yaw_deg: float = 0.0
    max_opening: float | None = None
    download: dict | None = None
    real: dict | None = None        # servo ids and calibration for a real arm (see feetech.py)
    source: Path | None = None
    scale: float = 1.0              # the arm k times its described size (see sized and resize)
    pad_contact: dict | None = None  # how the pads touch: {"friction": [slide, spin, roll], "condim": 4, ...} (see pad_tune)
    tilts: tuple = (0.0,)           # radians the gripper may lean outward from straight down, smallest first (tilts_deg)

    @property
    def mjcf_path(self) -> Path | None:
        if self.mjcf == "builtin":
            return None
        p = Path(self.mjcf)
        return p if p.is_absolute() else REPO / p

    def sized(self, scale: float) -> "ArmSpec":
        """The same arm at `scale` times its described size: tool point, jaw travel (parallel jaws slide, so their
        travel is a length; a single jaw turns, so its angles stay) and the opening it can hold."""
        k = float(scale) / self.scale
        g = dict(self.gripper)
        if g["mode"] == "parallel":
            g["open"], g["closed"] = [v * k for v in g["open"]], [v * k for v in g["closed"]]
        return replace(self, scale=float(scale), gripper=g, tool={**self.tool, "pos": [v * k for v in self.tool["pos"]]},
                       max_opening=(self.max_opening or 0.08) * k)

    def missing_files(self) -> list[Path]:
        if self.mjcf == "builtin":
            return []
        base = self.mjcf_path.parent
        need = [self.mjcf_path] + [base / f for f in (self.download or {}).get("files", [])]
        return [p for p in need if not p.exists()]


# ---------------- loading ----------------
def list_arms() -> list[str]:
    return sorted(p.stem for p in ARMS_DIR.glob("*.json"))


def describe_arms() -> list[dict]:
    """The arms that can be chosen: [{name, about, ready}]. `ready` is False when files must be downloaded first."""
    out = []
    for name in list_arms():
        try:
            arm = load(name)
            about = json.loads(arm.source.read_text(encoding="utf-8")).get("about", "")
            out.append({"name": name, "about": about, "ready": not arm.missing_files()})
        except (ArmError, OSError, ValueError):
            continue
    return out


def load(name: str | None = None) -> ArmSpec:
    """An arm by name (echotwin/robot/arms/<name>.json), by path, or 'auto' (the ARM variable, else builtin)."""
    name = (name or os.environ.get("ARM") or "auto").strip()
    if name == "auto":
        name = "builtin"
    path = Path(name)
    if not path.suffix:
        path = ARMS_DIR / f"{name}.json"
    elif not path.is_absolute():
        path = REPO / path
    if not path.exists():
        raise ArmError(f"No arm descriptor {name!r}. Known arms: {', '.join(list_arms())}, or a path to a .json file.")
    return parse(json.loads(path.read_text(encoding="utf-8")), path)


def parse(d: dict, source: Path | None = None) -> ArmSpec:
    for key in ("name", "mjcf", "base_body", "joints", "actuators", "gripper", "tool", "pads", "home"):
        if key not in d:
            raise ArmError(f"The arm descriptor has no '{key}'.")
    if len(d["joints"]) != 5 or len(d["actuators"]) != 5:
        raise ArmError("The arm needs 5 joints and 5 actuators (base rotation, three bending joints, wrist roll).")
    g = d["gripper"]
    n = len(g.get("actuators", []))
    if g.get("mode") not in ("parallel", "single") or not n or len(g.get("open", [])) != n or len(g.get("closed", [])) != n:
        raise ArmError("gripper: 'mode' (parallel or single) and matching 'actuators', 'open' and 'closed' lists are needed.")
    for key in ("body", "pos", "point_axis", "close_axis"):
        if key not in d["tool"]:
            raise ArmError(f"tool: '{key}' is missing.")
    if len(d["home"]) != 5:
        raise ArmError("home: five joint values are needed.")
    return ArmSpec(name=d["name"], mjcf=d["mjcf"], base_body=d["base_body"], joints=list(d["joints"]),
                   actuators=list(d["actuators"]), gripper=dict(g), tool=dict(d["tool"]), pads=list(d["pads"]),
                   home=[float(v) for v in d["home"]], prefix=d.get("prefix", PREFIX),
                   mount_yaw_deg=float(d.get("mount_yaw_deg", 0.0)), max_opening=d.get("max_opening"),
                   download=d.get("download"), real=d.get("real"), source=source, pad_contact=d.get("pad_contact"),
                   tilts=_tilts(d.get("tilts_deg", [0])))


def _tilts(deg) -> tuple:
    """tilts_deg -> radians, smallest first, always starting with straight down (0)."""
    try:
        vals = sorted({0.0, *(float(v) for v in deg)})
    except (TypeError, ValueError):
        raise ArmError("tilts_deg: a list of angles in degrees, like [0, 25, 45].")
    if vals[0] < 0 or vals[-1] > 80:
        raise ArmError("tilts_deg: angles between 0 and 80 degrees.")
    return tuple(float(np.radians(v)) for v in vals)


# ---------------- putting the arm into a scene ----------------
def child_spec(arm: ArmSpec) -> "mujoco.MjSpec":
    if arm.mjcf == "builtin":
        kid = mujoco.MjSpec.from_string(BUILTIN_MJCF)
    else:
        missing = arm.missing_files()
        if missing:
            raise ArmError(f"The arm '{arm.name}' is missing {len(missing)} file(s), e.g. {missing[0]}. "
                           f"Fetch it with: python -m echotwin.robot.arm --download {arm.name}")
        kid = mujoco.MjSpec.from_file(str(arm.mjcf_path))
    if arm.pad_contact:
        pad_tune(kid, arm.pads, arm.pad_contact)
    if arm.scale != 1.0:
        resize(kid, arm.scale, jaws=set(arm.gripper["actuators"]))
    return kid


PAD_KEYS = ("friction", "condim", "solref", "solimp", "margin")


def pad_tune(spec: "mujoco.MjSpec", pads: list[str], contact: dict) -> int:
    """Set how the gripper's pads touch things (the geoms whose names contain one of `pads`), from the arm descriptor's
    "pad_contact": friction (slide, spin, roll), condim (4 adds spin friction, so a pinched thing does not swivel),
    solref, solimp, margin. The arm's own file stays as it is. Returns how many pads were changed."""
    unknown = set(contact) - set(PAD_KEYS)
    if unknown:
        raise ArmError(f"pad_contact: unknown key(s) {sorted(unknown)}; use {', '.join(PAD_KEYS)}.")
    n = 0
    for g in spec.geoms:
        if g.name and any(p in g.name for p in pads):
            for key, val in contact.items():
                setattr(g, key, val if key in ("condim", "margin") else np.asarray(val, float))
            n += 1
    if not n:
        raise ArmError(f"pad_contact: no geom name contains {pads}.")
    return n


def resize(spec: "mujoco.MjSpec", k: float, jaws: set | frozenset = frozenset()) -> None:
    """Make an arm k times bigger (or smaller) and keep it moving the same way.

    Lengths times k, masses times k**3, inertias times k**5. A turning joint needs torques k**5 times bigger for the same
    motion (inertia), a sliding one forces k**4 times bigger (mass times a k times longer travel): gains, damping, armature
    and force limits follow, so the scaled arm has the same speed and stiffness relative to its size. The jaws (`jaws`: the
    gripper's actuator names) are the exception: how hard they squeeze is about the object, not the arm, so the force at their
    tip only grows with k (a big gripper squeezing a light round thing k**4 times harder pops it out)."""
    def nan(v):
        return v is None or np.isnan(np.asarray(v, float)).any()

    for f in spec.frames:
        f.pos = f.pos * k
    for b in spec.bodies:
        b.pos = b.pos * k
        if b.explicitinertial:
            b.mass, b.ipos = b.mass * k ** 3, b.ipos * k
            b.inertia = b.inertia * k ** 5
            if not nan(b.fullinertia):
                b.fullinertia = b.fullinertia * k ** 5
    for g in spec.geoms:
        g.pos, g.size = g.pos * k, g.size * k
        if not nan(g.fromto[0]):
            g.fromto = g.fromto * k
        if not nan(g.mass):
            g.mass = g.mass * k ** 3
    for s in spec.sites:
        s.pos, s.size = s.pos * k, s.size * k
    for m in spec.meshes:
        m.scale = m.scale * k
    slide = {}
    for j in spec.joints:
        sl = j.type == mujoco.mjtJoint.mjJNT_SLIDE
        slide[j.name] = sl
        if sl:
            j.range = j.range * k
        torque = k ** 4 if sl else k ** 5
        j.damping, j.armature, j.frictionloss = j.damping * (k ** 3 if sl else torque), \
            j.armature * (k ** 3 if sl else torque), j.frictionloss * torque
    for a in spec.actuators:
        sl = slide.get(a.target, False)
        gain = k ** 3 if sl else k ** 5                      # N per m (or Nm per rad) for the same motion
        a.gainprm[0] *= gain
        a.biasprm[1] *= gain
        if a.biasprm[2] < 0:                                 # an explicit damping; a positive value is a damping ratio
            a.biasprm[2] *= gain
        if a.name in jaws:                                   # the squeeze holds the object, whose weight does not change:
            a.forcerange = a.forcerange * (k if sl else k ** 2)       # force at the jaw tip grows with k (a turning jaw: torque k**2)
        else:
            a.forcerange = a.forcerange * (k ** 4 if sl else k ** 5)
        if sl:
            a.ctrlrange = a.ctrlrange * k


BASE_JOINTS = ("robot_x", "robot_y", "robot_yaw")       # a mobile base: drives along x and y, turns about z


def add_mobile_base(scene: "mujoco.MjSpec", robot) -> "mujoco.MjsBody":
    """A base that drives in any direction (two slides and a turn, held by position servos), seen from above as its
    footprint. It bumps into the objects and the furniture (collision bit 4, see scene.BUMPS) but not into the surface
    it drives on or the arm on top of it. The planner keeps it clear of things; driving into one anyway pushes it."""
    fx, fy = (v / 2 for v in robot.footprint)
    h = max(robot.mount_height, 0.02)
    b = scene.worldbody.add_body(name="robot_base", pos=[0, 0, 0])
    for name, kind, axis in zip(BASE_JOINTS, (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE),
                                ([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        b.add_joint(name=name, type=kind, axis=axis, damping=5.0 if kind == mujoco.mjtJoint.mjJNT_SLIDE else 0.5)
    b.add_geom(name="robot_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=[fx, fy, h / 2], pos=[0, 0, h / 2],
               rgba=[0.22, 0.24, 0.28, 1], contype=4, conaffinity=4, density=600)
    for k, ang in enumerate((90, 210, 330)):             # three wheels, for the look
        a = np.radians(ang)
        b.add_geom(name=f"robot_wheel_{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.035, 0.012, 0],
                   pos=[0.8 * fx * np.cos(a), 0.8 * fy * np.sin(a), 0.035], euler=[90, 0, ang + 90],
                   rgba=[0.1, 0.1, 0.1, 1], contype=4, conaffinity=4, density=300)
    for name, kp in zip(BASE_JOINTS, (4000.0, 4000.0, 120.0)):
        act = scene.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=kp, dampratio=1.0)
    return b


def compose(scene: "mujoco.MjSpec", arm: ArmSpec, base_xy=(0.0, 0.0), robot=None) -> None:
    """Attach the arm to `scene` with its base at base_xy on the table (z = 0) and add the `tool` site. With a mobile
    `robot` (robots.py) the arm is mounted on a base that drives; the base joints then say where it is (x, y, turn)."""
    kid = child_spec(arm)
    for name in (arm.base_body, arm.tool["body"], *arm.joints):
        if kid.body(name) is None and kid.joint(name) is None:
            raise ArmError(f"The arm file has no body or joint called '{name}'.")
    if robot is not None and robot.mobile:
        frame = add_mobile_base(scene, robot).add_frame(pos=[0, 0, robot.mount_height], euler=[0, 0, arm.mount_yaw_deg])
    else:
        frame = scene.worldbody.add_frame(pos=[base_xy[0], base_xy[1], 0.0], euler=[0, 0, arm.mount_yaw_deg])
    scene.attach(kid, frame=frame, prefix=arm.prefix)
    scene.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    scene.option.impratio = 10
    for b in scene.bodies:                      # the servos hold the weight of the arm, as on the real thing
        if b.name.startswith(arm.prefix):
            b.gravcomp = 1.0
    scene.body(arm.prefix + arm.tool["body"]).add_site(
        name="tool", pos=list(arm.tool["pos"]), size=[0.004, 0, 0], rgba=[1, 0.2, 0.2, 1])


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class ArmIK:
    """Damped least squares on a scratch copy of the data.

    Pose asked for: the tool point at `target`, the tool's `point_axis` straight down, its `close_axis`
    horizontal along `yaw`. 5 joints, 6 constraints: position wins, orientation is the compromise.
    """

    def __init__(self, model: "mujoco.MjModel", arm: ArmSpec, base_xy=(0.0, 0.0), mount: float = 0.0):
        self.m, self.arm = model, arm
        self.base = np.array([base_xy[0], base_xy[1], 0.0])         # targets are in table coordinates
        self.mount = float(mount)                                   # the arm's base this high above the surface
        self.scratch = mujoco.MjData(model)
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in BASE_JOINTS]
        self.base_qadr = [int(model.jnt_qposadr[j]) for j in ids] if min(ids) >= 0 else []   # a mobile base
        self.yaw = 0.0
        if self.base_qadr:
            self.set_base((base_xy[0], base_xy[1], 0.0))
        self.jid = [self._joint(n) for n in arm.joints]
        self.qadr = [model.jnt_qposadr[j] for j in self.jid]
        self.dadr = [model.jnt_dofadr[j] for j in self.jid]
        self.lo = model.jnt_range[self.jid, 0].copy()
        self.hi = model.jnt_range[self.jid, 1].copy()
        self.site = model.site("tool").id
        self.p_axis = _unit(arm.tool["point_axis"])
        self.c_axis = _unit(arm.tool["close_axis"])
        self.q_home = np.array(arm.home, float)
        self.q_down = self.q_home.copy()

    def set_base(self, pose):
        """Where a mobile base stands (x, y, turn): the solver works on a copy of the robot standing there."""
        if not self.base_qadr:
            return
        for a, v in zip(self.base_qadr, pose):
            self.scratch.qpos[a] = v
        self.base, self.yaw = np.array([pose[0], pose[1], 0.0]), float(pose[2])

    def _joint(self, name: str) -> int:
        j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, self.arm.prefix + name)
        if j < 0:
            raise ArmError(f"The arm has no joint '{name}'.")
        return j

    def seed_down(self, near_rel_xy, z=0.10):
        """A good starting pose: tool pointing down about `near_rel_xy` from the base (many iterations)."""
        q, ep, er = self._solve(self.base + np.array([near_rel_xy[0], near_rel_xy[1], z]), 0.0, self.q_home.copy(), 600)
        self.q_down = q
        return ep, er

    def pose(self, q) -> tuple[np.ndarray, np.ndarray]:
        s = self.scratch
        for a, v in zip(self.qadr, q):
            s.qpos[a] = v
        mujoco.mj_kinematics(self.m, s)
        return s.site_xpos[self.site].copy(), s.site_xmat[self.site].reshape(3, 3).copy()

    def _aim(self, target, yaw, tilt):
        """The tool's pointing direction and jaw-closing direction for a target: straight down, or leaning `tilt` radians
        outward from the base (like an arm reaching out); the closing direction stays square to the pointing one."""
        down = np.array([0.0, 0.0, -1.0])
        close = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        if not tilt:
            return down, close
        out = np.asarray(target, float)[:2] - self.base[:2]
        n = float(np.linalg.norm(out))
        out = out / n if n > 1e-6 else np.array([0.0, 1.0])
        point = np.cos(tilt) * down + np.sin(tilt) * np.array([out[0], out[1], 0.0])
        close = close - (close @ point) * point
        return point, close / np.linalg.norm(close)

    def _solve(self, target, yaw, q0, iters, tilt=0.0):
        s = self.scratch
        q = np.clip(np.asarray(q0, float), self.lo, self.hi)
        down, close = self._aim(target, yaw, tilt)
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        e_pos = np.ones(3)
        e_rot = np.ones(3)
        for _ in range(iters):
            for a, v in zip(self.qadr, q):
                s.qpos[a] = v
            mujoco.mj_kinematics(self.m, s)
            mujoco.mj_comPos(self.m, s)
            R = s.site_xmat[self.site].reshape(3, 3)
            e_pos = target - s.site_xpos[self.site]
            e_rot = 0.5 * (np.cross(R @ self.p_axis, down) + np.cross(R @ self.c_axis, close))
            if np.linalg.norm(e_pos) < 4e-4 and np.linalg.norm(e_rot) < 4e-3:
                break
            mujoco.mj_jacSite(self.m, s, jp, jr, self.site)
            J = np.vstack([jp[:, self.dadr], 0.3 * jr[:, self.dadr]])
            err = np.concatenate([e_pos, 0.3 * e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q = np.clip(q + 0.8 * dq, self.lo, self.hi)
        return q, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))

    def solve(self, target, yaw, q0, iters=60, tilt=0.0):
        """-> (joint values, position error in m, orientation error). Warm start first, then restarts.
        `tilt`: radians the tool leans outward from pointing straight down (see _aim)."""
        q, ep, er = self._solve(target, yaw, q0, 6, tilt)
        if ep < 2e-3 and er < 2e-2:
            return q, ep, er
        best = (ep + 0.05 * er, q, ep, er)
        for seed in (q0, self.q_down):
            q, ep, er = self._solve(target, yaw, seed, iters, tilt)
            if ep + 0.05 * er < best[0]:
                best = (ep + 0.05 * er, q, ep, er)
            if ep < 1e-3 and er < 1e-2:
                break
        return best[1], best[2], best[3]


class Workspace:
    """Where the tool can go, pointing down, relative to the base (forward = +y).

    Built once per arm by asking the IK, then cached: feasible[radius, height]. `clamp` pulls a wanted tool position
    to the nearest place the arm can really reach (inside the ring around the base, inside the pan range, below the
    ceiling for that radius).
    """
    RADII = np.arange(0.04, 0.50, 0.02)
    HEIGHTS = np.array([0.006, 0.02, 0.05, 0.09, 0.13, 0.17, 0.22])
    _cache: dict = {}

    def __init__(self, ik: "ArmIK", pan_half_range: float, scale: float = 1.0, tilts=(0.0,)):
        self.pan = pan_half_range
        self.RADII, self.HEIGHTS = self.RADII * scale, self.HEIGHTS * scale    # the grid is as big as the arm
        self.tilts = tuple(tilts)
        self.oks = np.zeros((len(self.tilts), len(self.RADII), len(self.HEIGHTS)), bool)     # per tilt
        for k, tilt in enumerate(self.tilts):
            q = ik.q_down
            for i, r in enumerate(self.RADII):
                for j, z in enumerate(self.HEIGHTS):
                    q2, ep, er = ik.solve(ik.base + np.array([0.0, r, z]), 0.0, q, iters=200, tilt=tilt)
                    self.oks[k, i, j] = ep < 2e-3 and er < 0.08
                    if self.oks[k, i, j]:
                        q = q2
        self.ok = self.oks.any(axis=0)                        # reachable with some tilt
        # the ceiling of each column: the highest row reached, except pointing straight down, where it is found between
        # that row and the next (rows are 4 cm apart, and a row can be missed where the solver did not find the pose).
        # Not for a lean: hovering leant out over a thing knocks it, so the arm leans only where the grid says it must.
        self.tops = np.array([[float(self.HEIGHTS[np.nonzero(col)[0].max()]) if col.any() else 0.0 for col in o] for o in self.oks])
        for i, r in enumerate(self.RADII):
            if not self.oks[0, i].any():
                continue
            lo = self.tops[0, i]
            hi = float(self.HEIGHTS[self.HEIGHTS > lo][0]) if (self.HEIGHTS > lo).any() else lo
            for _ in range(5):                                # to 4 cm / 32
                if hi - lo < 1e-4:
                    break
                mid = 0.5 * (lo + hi)
                _, ep, er = ik.solve(ik.base + np.array([0.0, r, mid]), 0.0, ik.q_down, iters=200)
                lo, hi = (mid, hi) if ep < 2e-3 and er < 0.08 else (lo, mid)
            self.tops[0, i] = lo
        # one lean per distance from the base, from the table to the ceiling: going down to a thing or up from it the
        # lean does not change, so the open jaws do not swing over it. Straight down where that reaches about as high
        # as leaning; else the smallest lean that reaches the table and that height. None: no lean reaches the table.
        self.column_tilt: list = []
        for i in range(len(self.RADII)):
            top = float(self.tops[:, i].max())
            pick = None
            for k, tilt in enumerate(self.tilts):
                if self.oks[k, i, :2].any() and self.tops[k, i] >= top - 0.01:
                    pick = tilt
                    break
            self.column_tilt.append(pick)
        low = self.ok[:, 1]                                   # table height
        if not low.any():
            raise ArmError("The arm cannot reach the table top with its tool pointing down.")
        self.r_min = float(self.RADII[np.argmax(low)])
        self.r_max = float(self.RADII[len(low) - 1 - np.argmax(low[::-1])])

    @classmethod
    def for_arm(cls, ik: "ArmIK", arm: ArmSpec) -> "Workspace":
        key = (arm.name, str(arm.source), arm.mount_yaw_deg, round(arm.scale, 4), round(ik.mount, 4), arm.tilts)
        if key not in cls._cache:
            lo, hi = ik.lo[0], ik.hi[0]
            cls._cache[key] = cls(ik, float(min(abs(lo), abs(hi)) - 0.05), arm.scale, arm.tilts)
        return cls._cache[key]

    def z_max(self, r: float) -> float:
        """The highest the tool reaches at this distance from the base, pointing straight down or leaning out."""
        i = int(np.clip(np.searchsorted(self.RADII, r), 0, len(self.RADII) - 1))
        return float(self.tops[:, i].max())

    def tilt_for(self, rel_xy, z: float) -> float:
        """How far to lean out at this spot (relative to the base): the one lean of its distance from the base
        (column_tilt); where there is none, the smallest that reaches this height, the largest when none does."""
        if len(self.tilts) == 1:
            return self.tilts[0]
        i = int(np.clip(np.searchsorted(self.RADII, float(np.hypot(*rel_xy))), 0, len(self.RADII) - 1))
        if self.column_tilt[i] is not None:
            return self.column_tilt[i]
        j = int(np.clip(np.searchsorted(self.HEIGHTS, z), 0, len(self.HEIGHTS) - 1))
        for k, tilt in enumerate(self.tilts):
            if self.oks[k, i, j]:
                return tilt
        return self.tilts[-1]

    def reachable(self, rel_xy, z: float = 0.02) -> bool:
        """rel_xy: position relative to the base."""
        r = float(np.hypot(*rel_xy))
        ang = abs(np.arctan2(rel_xy[0], rel_xy[1]))
        return self.r_min - 0.005 <= r <= self.r_max + 0.005 and ang <= self.pan and z <= self.z_max(r) + 0.005

    def clamp(self, rel, z_min: float = 0.004):
        """A wanted tool position relative to the base -> the nearest reachable one."""
        x, y, z = float(rel[0]), float(rel[1]), float(rel[2])
        r = float(np.hypot(x, y))
        ang = float(np.arctan2(x, y))                          # 0 = straight ahead
        r2 = float(np.clip(r, self.r_min, self.r_max))
        ang2 = float(np.clip(ang, -self.pan, self.pan))
        if abs(y) < 1e-9 and abs(x) < 1e-9:
            ang2 = 0.0
        z2 = float(np.clip(z, z_min, max(self.z_max(r2), z_min)))
        return np.array([r2 * np.sin(ang2), r2 * np.cos(ang2), z2])

    def sample(self, rng, z: float = 0.02):
        """A random reachable spot on the table, relative to the base (x, y)."""
        for _ in range(1000):
            r = rng.uniform(self.r_min + 0.01, self.r_max - 0.01)
            ang = rng.uniform(-self.pan * 0.7, self.pan * 0.7)
            if self.reachable((r * np.sin(ang), r * np.cos(ang)), z):
                return (float(r * np.sin(ang)), float(r * np.cos(ang)))
        raise ArmError("no reachable spot found")


def fetch(arm: ArmSpec) -> list[Path]:
    """Download the files of an arm that comes from the internet (only when asked)."""
    d = arm.download
    if not d:
        print(f"'{arm.name}' needs no download.")
        return []
    base = arm.mjcf_path.parent
    url = f"https://raw.githubusercontent.com/{d['repo']}/{d['commit']}/{d['path']}"
    got = []
    for rel in d["files"]:
        dst = base / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{url}/{rel}", timeout=60) as r:
            dst.write_bytes(r.read())
        got.append(dst)
    print(f"{arm.name}: {len(got)} file(s) downloaded into {base} ({d['license']}, {d['credit']}).")
    return got


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="the arms that are described")
    ap.add_argument("--download", metavar="NAME", help="fetch the files of an arm (about 3 MB for so_arm100)")
    ap.add_argument("--check", nargs="?", const="auto", metavar="NAME", help="load an arm and test it in the simulator")
    a = ap.parse_args(argv)
    if a.download:
        fetch(load(a.download))
    if a.list or not (a.download or a.check):
        for n in list_arms():
            arm = load(n)
            miss = arm.missing_files()
            print(f"  {n:12s} {'ready' if not miss else f'missing {len(miss)} file(s): --download {n}'}")
        print(f"in use: {load().name}")
    if a.check:
        from . import armcheck
        return armcheck.main(a.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
