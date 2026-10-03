"""Metric scale and table plane from the marker, using the points VGGT gave for each photo.

VGGT has no units, but the marker is a square of known size lying on the table. For each photo that shows it, the four
corners are placed on the table plane (fitted to the points inside the marker, corners from the camera rays), and the
side length in VGGT units gives `scale = marker size / side`. Several photos should agree; if they do not, the scale
is not trusted and the caller keeps the camera-height guess. Numpy only (the detection is in `marker.py`).
"""
from __future__ import annotations

import numpy as np

MAX_SPREAD = 0.05            # the photos must agree on the side length within 5 % (std / median)
MIN_PHOTOS = 2
MIN_POINTS = 20


def _fit_plane(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = p.mean(0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][2]
    return c, n / np.linalg.norm(n)


def corners_3d(point_map: np.ndarray, corners_px: np.ndarray, extr: np.ndarray, intr: np.ndarray):
    """The marker's 4 corners in world (VGGT) coordinates from one photo, or None.

    point_map: (H, W, 3) world point of every pixel; corners_px: (4, 2) as (x, y) in that map's pixels;
    extr: (3, 4) camera-from-world; intr: (3, 3)."""
    H, W = point_map.shape[:2]
    tl, tr, br, bl = corners_px
    uv = np.linspace(0.15, 0.85, 9)
    u, v = np.meshgrid(uv, uv)
    pix = ((1 - u)[..., None] * (1 - v)[..., None] * tl + u[..., None] * (1 - v)[..., None] * tr
           + (u * v)[..., None] * br + ((1 - u) * v)[..., None] * bl).reshape(-1, 2)
    cols = np.clip(np.round(pix[:, 0]).astype(int), 0, W - 1)
    rows = np.clip(np.round(pix[:, 1]).astype(int), 0, H - 1)
    inside = point_map[rows, cols]
    inside = inside[np.isfinite(inside).all(1)]
    if len(inside) < MIN_POINTS:
        return None
    c, n = _fit_plane(inside)
    R, t = extr[:, :3], extr[:, 3]
    cam = -R.T @ t
    out = []
    for x, y in corners_px:
        d = R.T @ (np.linalg.inv(intr) @ np.array([x, y, 1.0]))
        den = float(d @ n)
        if abs(den) < 1e-6:
            return None
        out.append(cam + d * (float((c - cam) @ n) / den))
    return np.array(out)


def _sides(c: np.ndarray) -> np.ndarray:
    return np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)


def estimate(frames: list[dict], size_m: float) -> dict | None:
    """frames: [{"point_map", "corners", "extr", "intr"}, ...] for the photos where the marker was seen.

    Returns the scale (metres per VGGT unit), how many photos gave a reading, how well they agree, and the marker's
    origin (its centre), x axis (along its top edge) and normal (towards the cameras), all in VGGT world coordinates."""
    cs, sides, cams = [], [], []
    for f in frames:
        c = corners_3d(f["point_map"], f["corners"], f["extr"], f["intr"])
        if c is None:
            continue
        s = _sides(c)
        if s.min() <= 0 or s.max() / s.min() > 1.3:               # not a square: a bad reading
            continue
        cs.append(c)
        sides.append(float(s.mean()))
        cams.append(-f["extr"][:, :3].T @ f["extr"][:, 3])
    if not cs:
        return None
    med = float(np.median(sides))
    spread = float(np.std(sides) / med) if len(sides) > 1 else 1.0
    c = np.median(np.array(cs), axis=0)
    x = ((c[1] - c[0]) + (c[2] - c[3])) / 2
    y = ((c[3] - c[0]) + (c[2] - c[1])) / 2
    x, y = x / np.linalg.norm(x), y / np.linalg.norm(y)
    n = np.cross(x, y)
    n /= np.linalg.norm(n)
    if (np.mean(cams, axis=0) - c.mean(0)) @ n < 0:                # the normal points to where the photos were taken
        n = -n
    return {"scale": size_m / med, "side_units": med, "photos_seen": len(cs), "spread": spread,
            "reliable": len(cs) >= MIN_PHOTOS and spread <= MAX_SPREAD,
            "origin": c.mean(0).tolist(), "x_axis": x.tolist(), "normal": n.tolist()}


def from_photos(photos: list[np.ndarray], point_maps: np.ndarray, extr: np.ndarray, intr: np.ndarray,
                size_m: float) -> dict | None:
    """Find the marker in each photo (any size, RGB or grey) and estimate. `point_maps`: (S, H, W, 3), `extr`:
    (S, 3, 4), `intr`: (S, 3, 3), at the model's resolution; photo corners are scaled down to it."""
    from . import marker
    H, W = point_maps.shape[1:3]
    frames = []
    for i, img in enumerate(photos):
        c = marker.detect(img)
        if c is None:
            continue
        h, w = img.shape[:2]
        c = (c + 0.5) * np.array([W / w, H / h]) - 0.5
        frames.append({"point_map": point_maps[i], "corners": c, "extr": extr[i], "intr": intr[i]})
    res = estimate(frames, size_m)
    if res is not None:
        res["photos_total"] = len(photos)
    return res
