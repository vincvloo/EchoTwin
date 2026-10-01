"""Augmented Monte Carlo Localization (particle filter) for sparse sonar readings."""
from __future__ import annotations

import numpy as np

from ..perception.gridmap import GridMap
from .robot import wrap
from .sonar import SonarRig


class MCL:
    def __init__(self, gmap: GridMap, rig: SonarRig, rng: np.random.Generator,
                 n_max=3000, n_min=600, alpha=(0.08, 0.05, 0.08, 0.05),
                 temper=0.5, clearance=0.1):
        self.map, self.rig, self.rng = gmap, rig, rng
        self.n_max, self.n_min = n_max, n_min
        self.alpha = alpha
        self.temper = temper            # < 1 flattens the likelihood: sonar beams are not independent
        self.clearance = clearance
        self.w_slow = self.w_fast = None
        self.fx, self.fy = gmap.free_cells(clearance)
        self.global_init()

    # ------------------------------------------------------------ initialisation
    def _random_poses(self, n):
        i = self.rng.integers(0, len(self.fx), n)
        jitter = self.rng.uniform(-0.5, 0.5, (n, 2)) * self.map.res
        th = self.rng.uniform(-np.pi, np.pi, n)
        return np.column_stack([self.fx[i] + jitter[:, 0], self.fy[i] + jitter[:, 1], th])

    def global_init(self):
        self.p = self._random_poses(self.n_max)
        self.w = np.full(self.n_max, 1.0 / self.n_max)

    # ------------------------------------------------------------ filter steps
    def predict(self, odo):
        rot1, trans, rot2 = odo
        a1, a2, a3, a4 = self.alpha
        n = len(self.p)
        r1 = rot1 + self.rng.normal(0, a1 * abs(rot1) + a2 * trans + 0.01, n)
        tr = trans + self.rng.normal(0, a3 * trans + a4 * (abs(rot1) + abs(rot2)) + 0.005, n)
        r2 = rot2 + self.rng.normal(0, a1 * abs(rot2) + a2 * trans + 0.01, n)
        th = self.p[:, 2]
        self.p[:, 0] += tr * np.cos(th + r1)
        self.p[:, 1] += tr * np.sin(th + r1)
        self.p[:, 2] = wrap(th + r1 + r2)

    def update(self, z):
        z_exp = self.rig.expected(self.map, self.p)
        ll = self.rig.log_likelihood(z, z_exp) * self.temper
        # Particles inside walls or in unscanned space are impossible.
        d = self.map.dist_at(self.p[:, 0], self.p[:, 1])
        r, c = self.map.to_cell(self.p[:, 0], self.p[:, 1])
        ok = self.map.inside(r, c)
        known = np.zeros(len(self.p), bool)
        known[ok] = self.map.known[r[ok], c[ok]]
        ll = np.where((d > 0.05) & known, ll, -1e9)
        m = ll.max()
        lik = np.exp(ll - m)
        w = self.w * lik
        # Track average likelihood (augmented MCL) to detect being lost / kidnapped.
        avg = m + np.log((self.w * lik).sum() + 1e-300)   # log of mean likelihood
        if self.w_slow is None:
            self.w_slow = self.w_fast = avg
        self.w_slow += 0.05 * (avg - self.w_slow)
        self.w_fast += 0.5 * (avg - self.w_fast)
        s = w.sum()
        self.w = w / s if s > 0 and np.isfinite(s) else np.full(len(w), 1.0 / len(w))

    def resample(self):
        n_eff = 1.0 / np.sum(self.w ** 2)
        if n_eff > 0.5 * len(self.p):
            return False
        spread = self.spread()
        n_new = self.n_min if spread < 0.35 else self.n_max          # fewer particles once converged
        p_rand = max(0.0, 1.0 - np.exp(self.w_fast - self.w_slow))    # lost -> inject random particles
        p_rand = min(p_rand, 0.1)
        n_rand = self.rng.binomial(n_new, p_rand)
        # Low-variance (systematic) resampling.
        k = n_new - n_rand
        cs = np.cumsum(self.w)
        u = (self.rng.random() + np.arange(k)) / k
        idx = np.minimum(np.searchsorted(cs, u), len(self.w) - 1)
        kept = self.p[idx]
        self.p = np.vstack([kept, self._random_poses(n_rand)]) if n_rand else kept
        self.w = np.full(len(self.p), 1.0 / len(self.p))
        return True

    # ------------------------------------------------------------ estimates
    def spread(self):
        mx = np.average(self.p[:, 0], weights=self.w)
        my = np.average(self.p[:, 1], weights=self.w)
        return float(np.sqrt(np.average((self.p[:, 0] - mx) ** 2 + (self.p[:, 1] - my) ** 2, weights=self.w)))

    def estimate(self):
        """Pose of the densest particle cluster (robust when the belief is multimodal)."""
        b = 0.25
        kx = np.floor(self.p[:, 0] / b).astype(int)
        ky = np.floor(self.p[:, 1] / b).astype(int)
        keys = kx * 100003 + ky
        uniq, inv = np.unique(keys, return_inverse=True)
        mass = np.bincount(inv, weights=self.w)
        in_best = inv == np.argmax(mass)
        bx = np.average(self.p[in_best, 0], weights=self.w[in_best] + 1e-12)
        by = np.average(self.p[in_best, 1], weights=self.w[in_best] + 1e-12)
        sel = np.hypot(self.p[:, 0] - bx, self.p[:, 1] - by) < 0.5
        w = self.w[sel]
        x = np.average(self.p[sel, 0], weights=w)
        y = np.average(self.p[sel, 1], weights=w)
        th = np.arctan2(np.average(np.sin(self.p[sel, 2]), weights=w),
                        np.average(np.cos(self.p[sel, 2]), weights=w))
        return np.array([x, y, th]), float(w.sum())
