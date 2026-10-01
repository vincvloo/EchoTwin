"""2D occupancy grid with a precomputed distance field for fast sonar ray casting.

Conventions
-----------
occ[row, col] is True where a cell is occupied. Row 0 is the lowest y value
(y grows with the row index), column 0 is the lowest x value. ``origin`` is the
world position (x, y) of the lower-left corner of cell (0, 0), in metres.
This matches the ROS map_server convention once the image is flipped vertically.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage


@dataclass
class GridMap:
    occ: np.ndarray            # bool (H, W)
    res: float                 # metres per cell
    origin: tuple[float, float]
    known: np.ndarray | None = None  # bool (H, W): cells observed by the scan (free or occupied)

    def __post_init__(self):
        self.occ = self.occ.astype(bool)
        if self.known is None:
            self.known = np.ones_like(self.occ)
        # Distance (m) from every cell centre to the nearest occupied cell.
        self.dist = ndimage.distance_transform_edt(~self.occ).astype(np.float32) * self.res
        # Gradient of the distance field = direction away from the nearest surface.
        gy, gx = np.gradient(self.dist)
        norm = np.hypot(gx, gy) + 1e-9
        self.nx = (gx / norm).astype(np.float32)
        self.ny = (gy / norm).astype(np.float32)
        self.H, self.W = self.occ.shape

    # ------------------------------------------------------------------ geometry
    @property
    def extent(self):
        x0, y0 = self.origin
        return (x0, x0 + self.W * self.res, y0, y0 + self.H * self.res)

    def to_cell(self, x, y):
        c = np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(np.int64)
        r = np.floor((np.asarray(y) - self.origin[1]) / self.res).astype(np.int64)
        return r, c

    def inside(self, r, c):
        return (r >= 0) & (r < self.H) & (c >= 0) & (c < self.W)

    def dist_at(self, x, y):
        """Distance to nearest obstacle; 0 outside the map (outside counts as wall)."""
        r, c = self.to_cell(x, y)
        ok = self.inside(r, c)
        out = np.zeros(np.shape(r), dtype=np.float32)
        out[ok] = self.dist[r[ok], c[ok]]
        return out

    def free_cells(self, clearance=0.0):
        """World coordinates of free, known cells at least `clearance` m from obstacles."""
        mask = (~self.occ) & self.known & (self.dist > clearance)
        r, c = np.nonzero(mask)
        x = self.origin[0] + (c + 0.5) * self.res
        y = self.origin[1] + (r + 0.5) * self.res
        return x, y

    def raycast(self, x, y, theta, max_range, hit_eps=None, iters=48):
        """Vectorised sphere tracing on the distance field.

        Returns (range, incidence_angle). Incidence is the angle between the ray and
        the surface normal at the hit point (0 = perpendicular hit). Rays that hit
        nothing return max_range and incidence 0.
        """
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        theta = np.asarray(theta, dtype=np.float32)
        x, y, theta = np.broadcast_arrays(x, y, theta)
        hit_eps = hit_eps or self.res * 0.75
        dx, dy = np.cos(theta), np.sin(theta)
        t = np.zeros(x.shape, dtype=np.float32)
        active = np.ones(x.shape, dtype=bool)
        hit = np.zeros(x.shape, dtype=bool)
        min_step = self.res * 0.5
        for _ in range(iters):
            if not active.any():
                break
            px = x[active] + dx[active] * t[active]
            py = y[active] + dy[active] * t[active]
            d = self.dist_at(px, py)
            h = d < hit_eps
            idx = np.flatnonzero(active)
            hit[idx[h]] = True
            step = np.maximum(d - hit_eps * 0.5, min_step)
            t[idx[~h]] += step[~h]
            done = h | (t[idx] > max_range)
            active[idx[done]] = False
        # Rays still active after the iteration budget are treated as long misses.
        rng = np.where(hit, np.minimum(t, max_range), max_range)
        # Surface normal just in front of the hit point.
        back = np.maximum(rng - self.res, 0)
        r, c = self.to_cell(x + dx * back, y + dy * back)
        ok = self.inside(r, c) & hit
        cosang = np.ones(x.shape, dtype=np.float32)
        cosang[ok] = np.abs(dx[ok] * self.nx[r[ok], c[ok]] + dy[ok] * self.ny[r[ok], c[ok]])
        inc = np.arccos(np.clip(cosang, 0, 1))
        return rng, inc

    # ------------------------------------------------------------------ io
    def save(self, stem: str | Path):
        """Write ROS map_server compatible <stem>.png + <stem>.yaml."""
        stem = Path(stem)
        img = np.full(self.occ.shape, 205, np.uint8)          # unknown = grey
        img[self.known & ~self.occ] = 254                      # free = white
        img[self.occ] = 0                                      # occupied = black
        Image.fromarray(np.flipud(img)).save(stem.with_suffix(".png"))
        meta = {
            "image": stem.with_suffix(".png").name,
            "resolution": float(self.res),
            "origin": [float(self.origin[0]), float(self.origin[1]), 0.0],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196,
        }
        stem.with_suffix(".yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    @classmethod
    def load(cls, yaml_path: str | Path) -> "GridMap":
        yaml_path = Path(yaml_path)
        meta = yaml.safe_load(yaml_path.read_text())
        img = np.asarray(Image.open(yaml_path.parent / meta["image"]).convert("L"), dtype=np.float32)
        img = np.flipud(img) / 255.0
        p_occ = img if meta.get("negate", 0) else 1.0 - img
        occ = p_occ > meta.get("occupied_thresh", 0.65)
        free = p_occ < meta.get("free_thresh", 0.196)
        return cls(occ=occ, res=float(meta["resolution"]),
                   origin=(meta["origin"][0], meta["origin"][1]), known=occ | free)

    def with_extra(self, extra_occ: np.ndarray) -> "GridMap":
        """Copy of this map with additional occupied cells (e.g. unmapped clutter)."""
        return GridMap(occ=self.occ | extra_occ, res=self.res, origin=self.origin, known=self.known)
