"""Test an arm in the simulator: python -m echotwin.robot.arm --check [NAME].

Loads the descriptor, builds a table with the arm, prints how far it reaches, and tries to pick up and place a 4 cm cube
and a 6 cm cylinder. A failed pick is reported, not hidden: a single-jaw arm may need tuning (docs/ARMS.md).
"""
from __future__ import annotations

import numpy as np

from . import arm as A
from .features import prop_skills as PS
from .scene import Layout
from .world import World

CASES = {"cube 4 cm": ("box", (0.02, 0.02, 0.02)), "cylinder 6 cm": ("cylinder", (0.03, 0.03, 0.04))}


def try_move(arm, shape: str, half: tuple) -> tuple[bool, str]:
    lay = Layout()
    lay.props = [{"name": "thing", "shape": shape, "pos": (0.0, -0.05), "yaw": 0.0, "size": half, "rgb": (0.8, 0.3, 0.3)},
                 {"name": "mark", "shape": "box", "pos": (0.2, 0.0), "yaw": 0.0, "size": (0.02, 0.02, 0.02), "rgb": (0.2, 0.3, 0.8)}]
    w = World(lay, arm)
    w.settle(20)
    goal = np.array([0.13, 0.0])
    task = {"object": "prop_0", "name": "thing", "goal": goal, "h": half[2]}
    res = PS.imagine(w, task, PS.waypoints(w, task, dict(PS.DEFAULTS)), PS.DEFAULTS["speed"])
    return bool(res["ok"]), res["text"]


def main(name: str | None = None) -> int:
    arm = A.load(None if name in (None, "auto") else name)
    miss = arm.missing_files()
    if miss:
        print(f"{arm.name}: missing files, run: python -m echotwin.robot.arm --download {arm.name}")
        return 1
    w = World(Layout(), arm)
    ws = w.workspace
    print(f"{arm.name}: tool pointing down reaches {ws.r_min * 100:.0f} to {ws.r_max * 100:.0f} cm from its base, "
          f"up to {max(ws.HEIGHTS[ws.ok.any(axis=0)]) * 100:.0f} cm high")
    fails = 0
    for label, (shape, half) in CASES.items():
        ok, text = try_move(arm, shape, half)
        fails += not ok
        print(f"  {label:14s} {'ok' if ok else 'FAILED'}: {text}")
    return 1 if fails else 0
