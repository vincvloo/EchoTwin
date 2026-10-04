"""Find an object in a camera frame by looking, and say where it is on the table.

No detector: the robot knows roughly where the object should be, so it looks in a window around that spot, takes the
colour of the table from the border of the window, and finds the blob that differs from it. `PlaneMap` then turns the blob
into table coordinates (metres), at the object's middle height so that a tall thing seen from above is not placed with its
perspective shift. Works on the render of the simulation's camera and, with a calibrated homography, on a real camera.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

THRESH = 12.0
L_FREE = 30.0             # a shadow only changes lightness: up to this much does not count, objects differ in colour
MIN_AREA_FRACTION = 0.08  # a blob smaller than this part of the expected object is noise


@dataclass
class Found:
    px: tuple[float, float]       # blob centre (x, y) in the frame
    area: float                   # pixels
    box: tuple[int, int, int, int]
    found: bool = True


def locate(bgr: np.ndarray, expected_px, size_px: float, thresh: float = THRESH, max_area: float | None = None,
           others=()) -> Found | None:
    """The blob nearest `expected_px` (x, y) that stands out from the table around it, or None.
    `size_px`: about how wide the object looks, in pixels (sets the window). `others`: where other known objects are expected
    (x, y): a blob nearer to one of them than to this one is theirs."""
    h, w = bgr.shape[:2]
    r = int(max(2.0 * size_px, 28))
    cx, cy = int(round(expected_px[0])), int(round(expected_px[1]))
    x0, y0, x1, y1 = max(0, cx - r), max(0, cy - r), min(w, cx + r), min(h, cy + r)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    win = bgr[y0:y1, x0:x1]
    lab = cv2.cvtColor(win, cv2.COLOR_BGR2LAB).astype(np.float32)
    ring = np.concatenate([lab[:3].reshape(-1, 3), lab[-3:].reshape(-1, 3), lab[:, :3].reshape(-1, 3), lab[:, -3:].reshape(-1, 3)])
    table = np.median(ring, axis=0)
    d = lab - table
    dist = np.hypot(d[..., 1], d[..., 2]) + 0.8 * np.maximum(np.abs(d[..., 0]) - L_FREE, 0)   # colour, plus big lightness changes
    fg = (dist > thresh).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(fg, connectivity=8)
    best, score = None, 0.0
    ex, ey = cx - x0, cy - y0
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA_FRACTION * size_px ** 2 or (max_area is not None and area > max_area):
            continue
        dd = float(np.hypot(cents[i][0] - ex, cents[i][1] - ey))
        if dd > 1.6 * size_px + 8:                              # too far from where it should be: something else
            continue
        if any(np.hypot(cents[i][0] + x0 - o[0], cents[i][1] + y0 - o[1]) < np.hypot(cents[i][0] - ex, cents[i][1] - ey) for o in others):
            continue                                            # it belongs to a neighbour
        s = area / (1.0 + (dd / max(size_px, 1.0)) ** 2)        # big and close wins
        if s > score:
            best, score = i, s
    if best is None:
        return None
    bx, by, bw, bh = (int(stats[best, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    return Found(px=(float(cents[best][0] + x0), float(cents[best][1] + y0)), area=float(stats[best, cv2.CC_STAT_AREA]),
                 box=(bx + x0, by + y0, bw, bh))


class PlaneMap:
    """Pixels <-> positions on the table (metres, x right, y away, z up), through a pinhole camera or a homography."""

    def __init__(self, to_table, to_pixel):
        self._to_table, self._to_pixel = to_table, to_pixel

    def to_table(self, px, z: float = 0.0) -> np.ndarray:
        """The table (x, y) of pixel `px` for a point at height `z`."""
        return self._to_table(np.asarray(px, float), z)

    def to_pixel(self, xy, z: float = 0.0) -> np.ndarray:
        return self._to_pixel(np.asarray(xy, float), z)

    def metres_to_pixels(self, m: float, xy, z: float = 0.0) -> float:
        a = self.to_pixel(xy, z)
        b = self.to_pixel(np.asarray(xy, float) + [m, 0.0], z)
        return float(np.linalg.norm(a - b))

    @classmethod
    def from_camera(cls, pos, xyaxes, fovy_deg: float, size) -> "PlaneMap":
        """A MuJoCo-style camera: `xyaxes` are the image's right and up directions in the world, it looks along -z of that frame."""
        pos = np.asarray(pos, float)
        right, up = np.asarray(xyaxes[:3], float), np.asarray(xyaxes[3:], float)
        right, up = right / np.linalg.norm(right), up / np.linalg.norm(up)
        forward = -np.cross(right, up)
        W, H = size
        f = (H / 2) / np.tan(np.radians(fovy_deg) / 2)
        cx, cy = W / 2, H / 2

        def to_table(px, z):
            d = forward + (px[0] - cx) / f * right - (px[1] - cy) / f * up
            lam = (z - pos[2]) / d[2]
            return (pos + lam * d)[:2]

        def to_pixel(xy, z):
            v = np.array([xy[0], xy[1], z]) - pos
            zc = v @ forward
            return np.array([cx + f * (v @ right) / zc, cy - f * (v @ up) / zc])
        return cls(to_table, to_pixel)

    @classmethod
    def from_homography(cls, H: np.ndarray) -> "PlaneMap":
        """`H` (3 x 3) takes table metres (x, y, 1) at z = 0 to pixels. Height is ignored (a flat-plane camera model)."""
        Hinv = np.linalg.inv(H)

        def to_table(px, z):
            p = Hinv @ np.array([px[0], px[1], 1.0])
            return p[:2] / p[2]

        def to_pixel(xy, z):
            p = H @ np.array([xy[0], xy[1], 1.0])
            return p[:2] / p[2]
        return cls(to_table, to_pixel)
