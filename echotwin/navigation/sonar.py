"""Ultrasonic range sensor model (HC-SR04-like).

A real sonar returns the FIRST echo inside a cone. We approximate the cone with a
fan of rays and keep the shortest ray that produces an echo. A ray hitting a
smooth surface at a steep angle (> specular_angle) bounces away and returns no
echo, which is why real sonars report false long ranges along angled walls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..perception.gridmap import GridMap


@dataclass
class SonarRig:
    # Mounting poses on the robot: (dx, dy, dtheta) in metres / radians, robot frame.
    mounts: list[tuple[float, float, float]] = field(default_factory=lambda: [
        (0.10, 0.0, 0.0),            # front
        (0.0, 0.10, np.pi / 2),      # left
        (0.0, -0.10, -np.pi / 2),    # right
        (-0.10, 0.0, np.pi),         # back
    ])
    fov: float = np.deg2rad(25)      # effective beam width
    n_rays: int = 5
    min_range: float = 0.02
    max_range: float = 4.0
    specular_angle: float = np.deg2rad(40)
    sigma0: float = 0.01             # noise std at 0 m
    sigma_k: float = 0.01            # extra std per metre
    p_spurious: float = 0.02         # random short reading (crosstalk, people, ...)

    @property
    def n(self):
        return len(self.mounts)

    def _ray_offsets(self):
        if self.n_rays == 1:
            return np.zeros(1)
        return np.linspace(-self.fov / 2, self.fov / 2, self.n_rays)

    def expected(self, gmap: GridMap, poses: np.ndarray) -> np.ndarray:
        """Noise-free expected readings for poses (P, 3) -> (P, n_sensors)."""
        poses = np.atleast_2d(poses).astype(np.float32)
        P = poses.shape[0]
        m = np.asarray(self.mounts, dtype=np.float32)            # (S, 3)
        off = self._ray_offsets().astype(np.float32)              # (R,)
        x, y, th = poses[:, 0:1], poses[:, 1:2], poses[:, 2:3]
        c, s = np.cos(th), np.sin(th)
        sx = x + c * m[:, 0] - s * m[:, 1]                         # (P, S)
        sy = y + s * m[:, 0] + c * m[:, 1]
        sth = th + m[:, 2]
        rx = np.repeat(sx[..., None], len(off), -1)                # (P, S, R)
        ry = np.repeat(sy[..., None], len(off), -1)
        rth = sth[..., None] + off
        rng, inc = gmap.raycast(rx.ravel(), ry.ravel(), rth.ravel(), self.max_range)
        rng = rng.reshape(P, self.n, len(off))
        inc = inc.reshape(P, self.n, len(off))
        rng = np.where(inc < self.specular_angle, rng, self.max_range)   # no echo -> max
        return np.clip(rng.min(-1), self.min_range, self.max_range)

    def measure(self, world: GridMap, pose: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One noisy reading from the TRUE world (which may differ from the map)."""
        z = self.expected(world, pose[None])[0]
        z = z + rng.normal(0, self.sigma0 + self.sigma_k * z)
        spur = rng.random(self.n) < self.p_spurious
        z[spur] = rng.uniform(self.min_range, np.maximum(z[spur], self.min_range + 1e-3))
        return np.clip(z, self.min_range, self.max_range)

    def log_likelihood(self, z: np.ndarray, z_exp: np.ndarray) -> np.ndarray:
        """Beam-model mixture (Thrun et al., Probabilistic Robotics ch. 6).

        z: (S,) measured, z_exp: (P, S) expected -> (P,) log-likelihood.
        """
        w_hit, w_short, w_max, w_rand = 0.75, 0.10, 0.10, 0.05
        sigma = 0.05 + 0.03 * z_exp          # looser than the true noise: map errors
        p_hit = np.exp(-0.5 * ((z - z_exp) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        lam = 1.0
        p_short = np.where(z <= z_exp, lam * np.exp(-lam * z), 0.0)
        p_max = (z >= self.max_range - 0.05).astype(float) * 20.0   # point mass, spread over 5 cm
        p_rand = 1.0 / self.max_range
        p = w_hit * p_hit + w_short * p_short + w_max * p_max + w_rand * p_rand
        return np.log(p + 1e-12).sum(-1)
