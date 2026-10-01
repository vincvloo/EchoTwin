"""Differential-drive robot with noisy wheel odometry and a simple sonar-based wander behaviour."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..perception.gridmap import GridMap


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Robot:
    pose: np.ndarray                 # true pose (x, y, theta)
    radius: float = 0.12
    odo_noise: tuple = (0.03, 0.02, 0.03, 0.02)   # alpha1..4 (Thrun odometry model)

    def step(self, world: GridMap, v: float, w: float, dt: float, rng: np.random.Generator):
        """Move with command (v, w). Returns the odometry reading (drot1, dtrans, drot2)."""
        x, y, th = self.pose
        # Real wheels slip: the executed motion differs a bit from the command.
        v_real = v * (1 + rng.normal(0, 0.03))
        w_real = w * (1 + rng.normal(0, 0.05)) + rng.normal(0, 0.01)
        nth = th + w_real * dt
        nx = x + v_real * dt * np.cos(th + w_real * dt / 2)
        ny = y + v_real * dt * np.sin(th + w_real * dt / 2)
        # Bumper: do not drive into obstacles.
        if world.dist_at(np.array([nx]), np.array([ny]))[0] < self.radius:
            nx, ny = x, y
        new = np.array([nx, ny, wrap(nth)])
        # Encoders measure the motion that happened, plus their own noise.
        dx, dy = new[0] - x, new[1] - y
        dtrans = np.hypot(dx, dy)
        drot1 = wrap(np.arctan2(dy, dx) - th) if dtrans > 1e-4 else 0.0
        drot2 = wrap(new[2] - th - drot1)
        a1, a2, a3, a4 = self.odo_noise
        o_rot1 = drot1 + rng.normal(0, 0.5 * (a1 * abs(drot1) + a2 * dtrans))
        o_trans = dtrans + rng.normal(0, 0.5 * (a3 * dtrans + a4 * (abs(drot1) + abs(drot2))))
        o_rot2 = drot2 + rng.normal(0, 0.5 * (a1 * abs(drot2) + a2 * dtrans))
        self.pose = new
        return np.array([o_rot1, o_trans, o_rot2])


class Wander:
    """Reactive explorer that only uses the sonar readings (front, left, right, back)."""

    def __init__(self, rng, v=0.25, w=1.2, safe=0.45):
        self.rng, self.v, self.w, self.safe = rng, v, w, safe
        self.turn_dir, self.turn_left = 1, 0
        self.bias = 0.0

    def __call__(self, z):
        front = z[0]
        # Side readings only exist on the 4-sonar rig (front, left, right, back).
        left, right = (z[1], z[2]) if len(z) >= 3 else (4.0, 4.0)
        if self.turn_left > 0:
            self.turn_left -= 1
            return 0.0, self.turn_dir * self.w
        if front < self.safe:
            self.turn_dir = 1 if left > right else -1
            self.turn_left = int(self.rng.integers(3, 9))
            return 0.0, self.turn_dir * self.w
        # Slowly varying heading bias gives varied, exploring trajectories.
        self.bias = 0.9 * self.bias + 0.1 * self.rng.normal(0, 0.8)
        steer = self.bias + 0.3 * np.clip(left - right, -1, 1) * (min(left, right) < 0.35)
        return self.v, float(np.clip(steer, -self.w, self.w))
