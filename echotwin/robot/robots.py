"""Robots: a base and an arm, described in a small JSON file (echotwin/robot/robots/<name>.json), so any robot is a file.

    base   {"kind": "fixed"}: the arm stands at the front edge of the surface (the table-top arm)
           {"kind": "mobile", "footprint_cm": [w, d], "mount_height_cm": h, "speed_cm_s": v, "turn_deg_s": t}:
           a base that drives in any direction (x, y and turning), the arm mounted on top of it, h above the surface
    arm    an arm descriptor (echotwin/robot/arms, docs/ARMS.md), at `arm_size` times its size

The base only drives between arm moves (drive, stop, pick), so every skill works the same on both kinds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

ROBOTS_DIR = Path(__file__).with_name("robots")


class RobotError(Exception):
    pass


@dataclass
class RobotSpec:
    name: str
    about: str = ""
    kind: str = "fixed"                 # fixed | mobile
    footprint: tuple = (0.26, 0.26)     # metres, the mobile base seen from above
    mount_height: float = 0.0           # metres from the surface to where the arm is mounted
    speed: float = 0.20                 # m/s
    turn: float = 1.57                  # rad/s
    arm: str = "builtin"
    arm_size: float = 1.0
    source: Path | None = field(default=None, compare=False)

    @property
    def mobile(self) -> bool:
        return self.kind == "mobile"

    def with_arm(self, arm: str | None = None, size: float | None = None) -> "RobotSpec":
        return replace(self, arm=self.arm if arm is None else arm, arm_size=self.arm_size if size is None else float(size))


def list_robots() -> list[str]:
    return sorted(p.stem for p in ROBOTS_DIR.glob("*.json"))


def parse(d: dict, source: Path | None = None) -> RobotSpec:
    if "name" not in d or "base" not in d:
        raise RobotError("A robot descriptor needs a 'name' and a 'base'.")
    b = d["base"]
    kind = b.get("kind")
    if kind not in ("fixed", "mobile"):
        raise RobotError("base.kind is 'fixed' or 'mobile'.")
    if kind == "fixed":
        return RobotSpec(d["name"], d.get("about", ""), "fixed", arm=d.get("arm", "builtin"),
                         arm_size=float(d.get("arm_size", 1.0)), source=source)
    try:
        fp = tuple(float(v) / 100 for v in b.get("footprint_cm", (26, 26)))
        spec = RobotSpec(d["name"], d.get("about", ""), "mobile", footprint=fp,
                         mount_height=float(b.get("mount_height_cm", 12)) / 100, speed=float(b.get("speed_cm_s", 20)) / 100,
                         turn=float(b.get("turn_deg_s", 90)) * 3.14159265 / 180, arm=d.get("arm", "builtin"),
                         arm_size=float(d.get("arm_size", 1.0)), source=source)
    except (TypeError, ValueError) as e:
        raise RobotError(f"The mobile base has an odd value: {e}")
    if len(fp) != 2 or min(fp) <= 0 or spec.speed <= 0 or spec.turn <= 0 or spec.mount_height < 0:
        raise RobotError("footprint_cm needs two positive sizes; speed and turn are positive; mount height is not negative.")
    return spec


def load(name: str | None = None) -> RobotSpec:
    """A robot by name (echotwin/robot/robots/<name>.json) or path. None: the table-top arm."""
    name = (name or "table_arm").strip()
    path = Path(name)
    if not path.suffix:
        path = ROBOTS_DIR / f"{name}.json"
    if not path.exists():
        raise RobotError(f"No robot {name!r}. Known robots: {', '.join(list_robots())}, or a path to a .json file.")
    return parse(json.loads(path.read_text(encoding="utf-8")), path)


def for_surface(surface: dict | None) -> str:
    """The robot that fits where the things are: a table-top arm on a table, a mobile one anywhere else."""
    return "table_arm" if (surface or {}).get("kind", "table") == "table" else "mobile_arm"
