"""A learned policy that runs without torch: a small network that predicts a chunk of the next actions.

Trained by `train_policy.py` (torch, perception environment), saved as a numpy `.npz`, run here with numpy only. ACT's two ideas are
kept: the network predicts a *chunk* of the next K actions, and at run time the chunks predicted at overlapping steps are *ensembled*
(newer chunks count less, `exp(-m * age)`), which smooths the motion and absorbs a single bad prediction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import policy_obs as O
from .world import VMAX

ENSEMBLE_M = 0.05
GOAL_HOLD = 10            # ticks with the goal met and the jaws open before the move counts as finished
MAX_TICKS = 600


class ChunkPolicy:
    """params: W0, b0, W1, b1, ... (x @ W + b, ReLU between), obs_mean, obs_std, act_mean, act_std, K."""

    def __init__(self, params: dict):
        self.p = {k: np.asarray(v) for k, v in params.items()}
        self.K = int(self.p["K"])
        self.n_layers = len([k for k in self.p if k.startswith("W")])
        self.reset()

    @classmethod
    def load(cls, path) -> "ChunkPolicy":
        z = np.load(Path(path))
        return cls({k: z[k] for k in z.files})

    def reset(self):
        self.chunks: list[tuple[int, np.ndarray]] = []       # (tick the chunk starts at, (K, 5) actions)
        self.t = 0

    def predict(self, obs) -> np.ndarray:
        """The next K actions for one observation, (K, 5), in real units."""
        x = (np.asarray(obs, np.float32) - self.p["obs_mean"]) / self.p["obs_std"]
        for i in range(self.n_layers):
            x = x @ self.p[f"W{i}"] + self.p[f"b{i}"]
            if i < self.n_layers - 1:
                x = np.maximum(x, 0.0)
        return x.reshape(self.K, O.ACT_DIM) * self.p["act_std"] + self.p["act_mean"]

    def act(self, obs) -> np.ndarray:
        """The action for this tick: the ensemble of every chunk that reaches it. Grip is 0 or 1, speed is capped."""
        self.chunks.append((self.t, self.predict(obs)))
        self.chunks = [(t0, c) for t0, c in self.chunks if self.t - t0 < self.K]
        ages = np.array([self.t - t0 for t0, _ in self.chunks], float)
        w = np.exp(-ENSEMBLE_M * (ages.max() - ages))                 # the oldest chunk counts most, as in ACT
        a = sum(wi * c[self.t - t0] for wi, (t0, c) in zip(w, self.chunks)) / w.sum()
        self.t += 1
        return finish(a)


def finish(a: np.ndarray) -> np.ndarray:
    """Make a raw output a valid action: jaws open or closed, tool speed within what the arm may do."""
    a = np.array(a, float)
    v = a[:3]
    n = np.linalg.norm(v)
    if n > VMAX:
        a[:3] = v * (VMAX / n)
    a[3] = 1.0 if a[3] > 0.5 else 0.0
    return a


class LearnedExecutor:
    """Drives one move with a policy. `action(world)` has the contract of `prop_skills.waypoint_action`: an action, or None when done."""

    def __init__(self, policy: ChunkPolicy, task: dict, max_ticks: int = MAX_TICKS):
        self.policy, self.task, self.max_ticks = policy, task, max_ticks
        self.policy.reset()
        self.t, self.done_for = 0, 0

    def action(self, world, d=None, hs=None):
        from .features import prop_skills as PS
        if self.t >= self.max_ticks:
            return None
        if self.t > 20 and not world.hand.grip and PS.goal_met(world, self.task):
            self.done_for += 1
            if self.done_for >= GOAL_HOLD:
                return None
        else:
            self.done_for = 0
        a = self.policy.act(O.from_world(world, self.task, self.t))
        self.t += 1
        return a
