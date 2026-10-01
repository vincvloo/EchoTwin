"""World: MuJoCo model + object/zone registry, stepping, grasping and goal checks.

Grasping is a kinematic attach: closing the gripper around a cube glues it to the hand
until release, then physics takes over again. Robust and good enough for trajectory data.
"""
import copy
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from .scene import (CUBE_HALF, DEFAULT_ZONES, HOME, OBJECT_NAMES, TABLE_HALF, WS_HALF,
                    ZONE_LABEL, Layout, build_xml)

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
        self.layout = layout.copy()
        self.model = mujoco.MjModel.from_xml_string(build_xml(self.layout))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.hand_mocap = m.body_mocapid[m.body("hand").id]
        self.finger_mocap = [m.body_mocapid[m.body(f"finger_{s}").id] for s in "lr"]
        self.obj_qadr = {n: m.jnt_qposadr[m.joint(f"obj_{n}").id] for n in OBJECT_NAMES}
        self.obj_dadr = {n: m.jnt_dofadr[m.joint(f"obj_{n}").id] for n in OBJECT_NAMES}
        for i in range(len(self.layout.props)):  # everyday objects are free bodies too
            self.obj_qadr[f"prop_{i}"] = m.jnt_qposadr[m.joint(f"prop_{i}").id]
            self.obj_dadr[f"prop_{i}"] = m.jnt_dofadr[m.joint(f"prop_{i}").id]
        self.zone_body = {z: m.body(f"zone_{z}").id for z in self.layout.zones}
        self.hand = HandState()
        mujoco.mj_forward(m, self.data)

    def present(self, name: str) -> bool:
        if name.startswith("prop_"):
            return True
        return self.layout.objects[name].get("present", True)

    def half(self, name: str) -> float:
        """Half height (grasp height above the table)."""
        if name.startswith("prop_"):
            return self.layout.props[int(name[5:])]["size"][2]
        return self.layout.objects[name]["half"]

    def radius(self, name: str) -> float:
        if name.startswith("prop_"):
            hx, hy, _ = self.layout.props[int(name[5:])]["size"]
            return max(hx, hy)
        return self.layout.objects[name]["half"]

    def things(self) -> list[str]:
        """Everything the gripper can pick up."""
        return self.objects() + [f"prop_{i}" for i in range(len(self.layout.props))]

    def objects(self) -> list[str]:
        return [n for n in OBJECT_NAMES if self.present(n)]

    # ---------- layout ----------
    def randomise(self, rng: np.random.Generator, keep_zones: bool = True) -> Layout:
        lay = self.layout.copy()
        lay.texture = self.layout.texture
        if not keep_zones:
            lay.zones = {k: dict(v) for k, v in DEFAULT_ZONES.items()}
        placed = []
        for n in OBJECT_NAMES:
            if not lay.objects[n].get("present", True):
                continue
            for _ in range(200):
                p = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.17, 0.03)])
                if all(np.linalg.norm(p - q) > 0.09 for q in placed) and not any(
                        self._in_rect(p, z["pos"], z["half"], margin=0.03) for z in lay.zones.values()):
                    break
            placed.append(p)
            lay.objects[n]["pos"] = (float(p[0]), float(p[1]))
        return lay

    def reset(self, layout: Layout | None = None):
        """Rebuild if zones/sizes/texture changed, else just reset state (fast)."""
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

    def zone_pos(self, zone: str) -> np.ndarray:
        return np.array(self.layout.zones[zone]["pos"], dtype=float)

    @staticmethod
    def _in_rect(p, c, half, margin=0.0) -> bool:
        return abs(p[0] - c[0]) <= half[0] + margin and abs(p[1] - c[1]) <= half[1] + margin

    def in_zone(self, name: str, zone: str, d=None, hs: HandState | None = None) -> bool:
        hs = hs or self.hand
        if not self.present(name) or hs.attached == name:
            return False
        p = self.obj_pos(name, d)
        z = self.layout.zones[zone]
        return self._in_rect(p, z["pos"], z["half"]) and p[2] < self.half(name) * 1.6

    def zone_of(self, name: str, d=None, hs=None) -> str | None:
        for z in self.layout.zones:
            if self.in_zone(name, z, d, hs):
                return z
        return None

    def describe_positions(self) -> dict:
        out = {}
        for n in self.objects():
            p = self.obj_pos(n)
            out[n] = {"x": round(float(p[0]), 3), "y": round(float(p[1]), 3), "zone": self.zone_of(n)}
        return out

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
        lo = np.array([-TABLE_HALF[0] + 0.03, -TABLE_HALF[1] + 0.03, zmin])
        hi = np.array([TABLE_HALF[0] - 0.03, TABLE_HALF[1] - 0.03, Z_MAX])
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

    def label(self, zone: str) -> str:
        return ZONE_LABEL.get(zone, zone)


__all__ = ["World", "HandState", "CTRL_DT", "CTRL_HZ", "VMAX", "HOME", "CUBE_HALF", "WS_HALF"]
