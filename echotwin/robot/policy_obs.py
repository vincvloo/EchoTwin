"""What a learned policy sees, and what it does. One definition for the demos, for training and for running.

Observation (18 numbers), all things a real arm with a camera could have:

    0-2   tool position (m)               3   commanded grip (0 open, 1 closed)
    4-6   object position (m)             7-8 goal (x, y)
    9     carrying (the pads hold it)     10-12 object width, height, length (m)
    13    stack (1 when the goal is "on top of")        14-15 sin, cos of the jaws' yaw
    16    time since the move started / 400 ticks       17 top of the thing it goes on, when stacking (m)

Action (5 numbers), at 20 Hz: tool velocity vx, vy, vz (m/s), grip (0 or 1), yaw of the jaws (rad).

Numpy only: the robot environment has no torch, and the training environment has no MuJoCo.
"""
from __future__ import annotations

import numpy as np

OBS_DIM, ACT_DIM = 18, 5
T_SCALE = 400.0
OBS_NAMES = ("hand.x", "hand.y", "hand.z", "grip", "obj.x", "obj.y", "obj.z", "goal.x", "goal.y", "carrying",
             "width", "height", "length", "stack", "sin_yaw", "cos_yaw", "t", "ref_top")
ACT_NAMES = ("vx", "vy", "vz", "grip", "yaw")


def build(state10, m: dict, stack: bool, yaw: float, t: int, ref_top: float = 0.0) -> np.ndarray:
    """One observation from the 10 numbers of `dataset.state_vector`, the object's measurements and the context."""
    s = np.asarray(state10, float)
    return np.array([*s[:10], m["width"], m["height"], m["length"], 1.0 if stack else 0.0, np.sin(yaw), np.cos(yaw),
                     t / T_SCALE, ref_top], np.float32)


def from_world(world, task: dict, t: int) -> np.ndarray:
    """The observation now, from a back-end (any object that has the robot contract of backend.py)."""
    me = task["object"]
    hand, obj = world.hand_pos(), world.obj_pos(me)
    state = [*hand, 1.0 if world.hand.grip else 0.0, *obj, task["goal"][0], task["goal"][1], 1.0 if world.hand.held == me else 0.0]
    ref_top = 0.0
    if task.get("stack"):
        ref_top = float(world.obj_pos(task["ref"])[2] + world.half(task["ref"]))
    m = task.get("m") or _m_from_world(world, me)
    return build(state, m, bool(task.get("stack")), world.grasp_yaw(me), t, ref_top)


def _m_from_world(world, me: str) -> dict:
    from .features import measure
    return measure.measure(world, me)


def from_episode(ep: dict) -> tuple[np.ndarray, np.ndarray, bool]:
    """(observations (N, 18), actions (N, 5), legacy) of a stored episode (`Dataset` JSON).

    Old episodes have 4-number actions (no yaw), no yaw in the state and no top of the reference object: those are filled with
    0 and `legacy` is True. They still teach the shape of the motion."""
    from .features import measure
    task, frames = ep["task"], ep["frames"]
    m = measure.from_task(task)
    A = np.array([f["action"] for f in frames], float)
    legacy = A.shape[1] < ACT_DIM or bool(m.get("legacy"))
    if A.shape[1] < ACT_DIM:
        A = np.hstack([A, np.zeros((len(A), ACT_DIM - A.shape[1]))])
    obs = np.stack([build(f["state"], m, bool(task.get("stack")), 0.0 if legacy else float(A[i, 4]), i) for i, f in enumerate(frames)])
    return obs, A.astype(np.float32), legacy
