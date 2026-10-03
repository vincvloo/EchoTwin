"""World: MuJoCo model + object registry, stepping and grasping.

Grasping is a kinematic attach: closing the gripper around an object glues it to the hand
until release, then physics takes over again. Robust and good enough for trajectory data.
"""
import copy
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from .scene import HOME, Layout, build_xml

CTRL_HZ = 20
CTRL_DT = 1.0 / CTRL_HZ
VMAX = 0.25
Z_MAX = 0.45
FINGER_OPEN = 0.045


@dataclass
class HandState:
    grip: bool = False
    attached: str | None = None
    offset: tuple = (0.0, 0.0, 0.0)

    def clone(self) -> "HandState":
        return replace(self)


class World:
    def __init__(self, layout: Layout | None = None):
        self.layout = layout or Layout()
        self.build(self.layout)

    # ---------- model ----------
    def build(self, layout: Layout):
        new_layout = layout.copy()
        self.model = mujoco.MjModel.from_xml_string(build_xml(new_layout))  # raises before anything changes
        self.layout = new_layout
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.hand_mocap = m.body_mocapid[m.body("hand").id]
        self.finger_mocap = [m.body_mocapid[m.body(f"finger_{s}").id] for s in "lr"]
        self.obj_qadr, self.obj_dadr = {}, {}
        for i in range(len(self.layout.props)):  # everyday objects are free bodies
            self.obj_qadr[f"prop_{i}"] = m.jnt_qposadr[m.joint(f"prop_{i}").id]
            self.obj_dadr[f"prop_{i}"] = m.jnt_dofadr[m.joint(f"prop_{i}").id]
        self.hand = HandState()
        mujoco.mj_forward(m, self.data)

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
        """Everything the gripper can pick up."""
        return [f"prop_{i}" for i in range(len(self.layout.props))]

    def reset(self, layout: Layout | None = None):
        """Rebuild if sizes/texture changed, else just reset state (fast)."""
        layout = layout or self.layout
        self.build(layout)

    # ---------- queries ----------
    def hand_pos(self, d=None) -> np.ndarray:
        d = d or self.data
        return d.mocap_pos[self.hand_mocap].copy()

    def obj_pos(self, name: str, d=None) -> np.ndarray:
        d = d or self.data
        a = self.obj_qadr[name]
        return d.qpos[a:a + 3].copy()

    # ---------- stepping ----------
    def clone(self):
        return copy.copy(self.data), self.hand.clone()

    def step(self, action, d=None, hs: HandState | None = None):
        """Advance one control tick. action = (vx, vy, vz, grip in {0,1})."""
        d = d if d is not None else self.data
        hs = hs if hs is not None else self.hand
        m = self.model
        a = np.asarray(action, dtype=float)
        v = np.clip(a[:3], -VMAX, VMAX)
        p0 = d.mocap_pos[self.hand_mocap].copy()
        zmin = (self.half(hs.attached) + 0.002) if hs.attached else 0.012
        th = self.layout.table_half
        lo = np.array([-th[0] + 0.03, -th[1] + 0.03, zmin])
        hi = np.array([th[0] - 0.03, th[1] - 0.03, Z_MAX])
        p1 = np.clip(p0 + v * CTRL_DT, lo, hi)

        want = a[3] > 0.5
        if want and not hs.grip:
            hs.grip = True
            hs.attached = self._graspable(p1, d)
            if hs.attached:
                hs.offset = tuple(self.obj_pos(hs.attached, d) - p1)
        elif not want and hs.grip:
            hs.grip = False
            if hs.attached:
                d.qvel[self.obj_dadr[hs.attached]:self.obj_dadr[hs.attached] + 6] = 0
            hs.attached = None

        if hs.attached:
            opening = self.half(hs.attached) + 0.006
        else:
            opening = 0.012 if hs.grip else FINGER_OPEN
        n = max(1, int(round(CTRL_DT / m.opt.timestep)))
        for i in range(n):
            p = p0 + (p1 - p0) * (i + 1) / n
            d.mocap_pos[self.hand_mocap] = p
            for k, s in zip(self.finger_mocap, (-1, 1)):
                d.mocap_pos[k] = p + np.array([s * opening, 0, 0.008])
            if hs.attached:
                qa, da = self.obj_qadr[hs.attached], self.obj_dadr[hs.attached]
                d.qpos[qa:qa + 3] = p + np.array(hs.offset)
                d.qvel[da:da + 6] = 0
            mujoco.mj_step(m, d)

    def _graspable(self, p, d) -> str | None:
        best, bd = None, 1e9
        for n in self.things():
            o = self.obj_pos(n, d)
            h, r = self.half(n), self.radius(n)
            dxy = np.linalg.norm(o[:2] - p[:2])
            if dxy < r + 0.014 and abs(o[2] - p[2]) < h + 0.014 and dxy < bd:
                best, bd = n, dxy
        return best

    def settle(self, ticks: int = 20):
        for _ in range(ticks):
            self.step((0, 0, 0, 1 if self.hand.grip else 0))


__all__ = ["World", "HandState", "CTRL_DT", "CTRL_HZ", "VMAX", "HOME"]
