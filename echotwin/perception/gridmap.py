"""A 2D grid over the floor plan, used to group labelled points into objects and to draw the object map.

Conventions
-----------
occ[row, col] is True where a cell holds points between the floor and head height (walls, furniture). Row 0 is the
lowest y value (y grows with the row index), column 0 is the lowest x value. ``origin`` is the world position (x, y)
of the lower-left corner of cell (0, 0), in metres. ``known`` marks cells the photos saw.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


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
        self.H, self.W = self.occ.shape

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

    def save(self, stem: str | Path):
        """Write the floor plan as <stem>.png (black = occupied, white = free, grey = not seen)."""
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        img = np.full(self.occ.shape, 205, np.uint8)
        img[self.known & ~self.occ] = 254
        img[self.occ] = 0
        Image.fromarray(np.flipud(img)).save(stem.with_suffix(".png"))
