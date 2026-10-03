"""The table texture of a 3D scan: paint the table top from the photos.

The 3D scan knows where the table plane is (the marker, or the surface it found) and where each photo was taken. For
every texel of the window the twin uses, project it into each photo, and take the median colour over the photos that
see it. Things standing on the table hide it in some photos but not in others, so the median mostly shows the bare table.
The marker itself is painted over. Runs in the robot env (numpy and OpenCV only).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from echotwin.scene import to_twin

from ..features.everyday import TEX_PX

MIN_VIEWS = 2
HALF = (TEX_PX[0] // 2, TEX_PX[1] // 2)      # the median is taken at half size, then scaled up


def _plane_height(scene: dict, surface: dict | None) -> float | None:
    """z of the table top in the map frame: 0 with the marker, else the top of the surface the scan found."""
    cal = scene.get("calibration") or {}
    if cal.get("source") == "marker":
        return 0.0
    if surface is not None and surface.get("height"):
        return float(surface.get("base_z", 0.0)) + float(surface["height"])
    return None


def window_points(scene: dict, z: float, size=HALF) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Map-frame position (x, y, z) of every texel centre, shape (H, W, 3), and the window (cx, cy, w, h) in metres."""
    (cx, cy), items, surface = to_twin.pick_window(scene)
    real_w, real_h = to_twin.window_size(items, surface)
    W, H = size
    xs = cx + ((np.arange(W) + 0.5) / W - 0.5) * real_w
    ys = cy + (0.5 - (np.arange(H) + 0.5) / H) * real_h           # texture row 0 is the far edge (+y)
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X, Y, np.full_like(X, z)], axis=-1), (cx, cy, real_w, real_h)


def build(scene: dict, folder: Path, cloud: str = "cloud", out_name: str = "table.png") -> str | None:
    """Write `folder/table.png` and return its name, or None when the scan cannot give a texture."""
    cal = scene.get("calibration") or {}
    M = cal.get("cloud_to_map")
    cams, pix = Path(folder) / f"{cloud}.cams.npz", Path(folder) / f"{cloud}.pix.npz"
    if not (M and cams.exists() and pix.exists()):
        return None
    (_, _), items, surface = to_twin.pick_window(scene)
    z = _plane_height(scene, surface)
    if z is None:
        return None
    P, (cx, cy, real_w, real_h) = window_points(scene, z)
    c, p = np.load(cams), np.load(pix)
    extr, intr, Rup = c["extrinsic"], c["intrinsic"], c["up_rotation"]
    mh, mw = (int(v) for v in p["model_hw"])
    Minv = np.linalg.inv(np.array(M, float))
    flat = P.reshape(-1, 3)
    world = ((Minv[:3, :3] @ flat.T + Minv[:3, 3:4]).T) @ Rup                # map -> cloud -> VGGT world
    H, W = P.shape[:2]
    stack = []
    for i in range(len(extr)):
        key = f"frame_{i}"
        if key not in p:
            continue
        photo = p[key]                                                       # RGB, up to 1280 px
        ph, pw = photo.shape[:2]
        R, t = extr[i][:, :3], extr[i][:, 3]
        cam = world @ R.T + t
        ok = cam[:, 2] > 1e-6
        uv = (cam @ intr[i].T)
        u = uv[:, 0] / np.where(ok, uv[:, 2], 1) * (pw / mw)
        v = uv[:, 1] / np.where(ok, uv[:, 2], 1) * (ph / mh)
        ok &= (u >= 1) & (u < pw - 1) & (v >= 1) & (v < ph - 1)
        col = cv2.remap(photo, u.reshape(H, W).astype(np.float32), v.reshape(H, W).astype(np.float32), cv2.INTER_LINEAR)
        stack.append((col, ok.reshape(H, W)))
    if len(stack) < MIN_VIEWS:
        return None
    cols = np.stack([c for c, _ in stack]).astype(np.float32)               # (N, H, W, 3)
    oks = np.stack([o for _, o in stack])                                   # (N, H, W)
    cols[~oks] = np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(cols, axis=0)
    seen = oks.sum(0) >= MIN_VIEWS
    if not seen.any():
        return None
    bgr = cv2.cvtColor(np.nan_to_num(med, nan=128).astype(np.uint8), cv2.COLOR_RGB2BGR)
    gaps = (~seen).astype(np.uint8) * 255
    if cal.get("source") == "marker":                                       # paint over the printed marker
        half = float(cal.get("marker_size_m", 0.10)) * 0.65
        px, py = (0.0 - cx) / real_w * W + W / 2, H / 2 - (0.0 - cy) / real_h * H
        r = half / real_w * W
        cv2.rectangle(gaps, (int(px - r), int(py - r)), (int(px + r), int(py + r)), 255, -1)
    if gaps.any():
        bgr = cv2.inpaint(bgr, gaps, 5, cv2.INPAINT_TELEA)
    bgr = cv2.resize(bgr, TEX_PX, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(Path(folder) / out_name), bgr)
    return out_name
