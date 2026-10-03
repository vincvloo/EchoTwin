"""What an object is, for a skill: how wide it is to grip, how tall, how long, how heavy. Not what it is called.

A skill learned on a mug carries over to anything of a similar size; the shape label (flat, box, cylinder, round)
is only the drawing. Sizes come from the twin (a guess until the scan is metric, PR10); the mass is estimated from
the size (`scene.prop_mass`) until a real arm can weigh it.
"""
from __future__ import annotations

import numpy as np

from ..scene import prop_mass

SCALE_CM = 3.0          # a difference of this much in width, height or length is "clearly different" (distance 1.0)
FLAT_H, TALL_H, SMALL_W = 0.025, 0.08, 0.045
CLASSES = ("flat", "small", "medium", "tall")
CLASS_LABEL = {"flat": "flat things", "small": "small things", "medium": "medium things", "tall": "tall things"}
CLASS_WORDS = {"flat": "a flat thing", "small": "a small thing", "medium": "a medium-sized thing", "tall": "a tall thing"}
# typical sizes (m) to rebuild a measurement for old demos that only stored a shape label and the half height
_LEGACY_WIDTH = {"flat": 0.06, "box": 0.045, "cylinder": 0.065, "round": 0.05}


def measure(world, name: str) -> dict:
    """Width of the grip (the narrow side), height, length (the long horizontal side) in metres, mass in kg."""
    pr = world.layout.props[int(name[5:])]
    hx, hy, hz = pr["size"]
    return {"width": float(world.grasp_width(name)), "height": float(2 * hz), "length": float(2 * max(hx, hy)),
            "mass": prop_mass(pr)}


def from_task(task: dict) -> dict:
    """The measurement stored with a task; old tasks get a typical one for their shape."""
    m = task.get("m")
    if m:
        return m
    h = 2 * float(task.get("h", 0.02))
    w = _LEGACY_WIDTH.get(task.get("shape", "box"), 0.05)
    return {"width": w, "height": h, "length": w, "mass": 0.0, "legacy": True}


def vector(m: dict) -> np.ndarray:
    return np.array([m["width"], m["height"], m["length"]]) * 100.0 / SCALE_CM


def distance(a: dict, b: dict) -> float:
    """About 1.0 when width, height or length differs by 3 cm."""
    return float(np.linalg.norm(vector(a) - vector(b)))


def size_class(m: dict) -> str:
    if m["height"] < FLAT_H:
        return "flat"
    if m["height"] >= TALL_H:
        return "tall"
    return "small" if m["width"] < SMALL_W else "medium"


def describe(m: dict) -> str:
    """For speech: 'a tall thing, 6 cm wide and 10 cm tall'."""
    return f"{CLASS_WORDS[size_class(m)]}, {m['width'] * 100:.0f} cm wide and {m['height'] * 100:.0f} cm tall"
