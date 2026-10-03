"""Put a reconstructed point cloud in a clean frame, and rasterise it into a floor plan.

Steps: read the points -> put gravity on +z (known up axis, or the largest plane) -> find the floor (RANSAC) and
level it -> align the walls to the axes -> rasterise a height band into a 2D grid. The object map
(objects.py) groups labelled points on that grid.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

from .gridmap import GridMap

UP = {"y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),   # glTF / ARCore: +Y up -> +Z up
      "z": np.eye(3),
      "x": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
      "-y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
      "-z": np.diag([1, -1, -1]),
      "-x": np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])}
VOXEL_ABOVE = 1_000_000     # downsample clouds larger than this before RANSAC


def _read_geometry(path: str, n: int, seed: int) -> np.ndarray:
    """All points of a mesh, point cloud or multi-part scene, in the file's frame.

    Scenes (e.g. .glb with one mesh per room chunk) are flattened with their node transforms.
    Meshes are sampled by area over the whole scene; point clouds are used as they are.
    Colours and textures are ignored.
    """
    geom = trimesh.load(path)
    parts = geom.dump() if isinstance(geom, trimesh.Scene) else [geom]
    clouds = [np.asarray(g.vertices) for g in parts
              if isinstance(g, trimesh.PointCloud) or len(getattr(g, "faces", [])) == 0]
    meshes = [g for g in parts if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    if meshes:
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        clouds.append(trimesh.sample.sample_surface(mesh, n, seed=seed)[0])
    if not clouds:
        raise ValueError(f"{path}: no vertices found")
    pts = np.vstack(clouds).astype(float)
    return pts[np.isfinite(pts).all(1)]


def voxel_downsample(pts: np.ndarray, voxel: float = 0.02) -> np.ndarray:
    """Keep one point (the centroid) per occupied voxel."""
    keys = np.floor((pts - pts.min(0)) / voxel).astype(np.int64)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((len(cnt), 3))
    np.add.at(out, inv, pts)
    return out / cnt[:, None]


def detect_up(pts: np.ndarray, rng: np.random.Generator, iters=300, thresh=0.02, sample=50_000) -> str:
    """Up axis of the file = the axis along which the largest flat plane's normal points.

    The largest plane can be the floor or the ceiling, so the sign comes from the contents:
    furniture sits on the floor, so the slab 0.1-0.8 m from the floor end holds more points than
    the same slab under the ceiling end.
    Only the six axis directions are considered; small tilt is fixed later by `level_floor`.
    """
    p = pts[rng.choice(len(pts), min(sample, len(pts)), replace=False)]
    best_n, best_nrm, best_a = 0, None, None
    for _ in range(iters):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        if np.linalg.norm(nrm) < 1e-9:
            continue
        nrm /= np.linalg.norm(nrm)
        if np.abs(nrm).max() < 0.9:          # only planes close to an axis
            continue
        k = int((np.abs((p - a) @ nrm) < thresh).sum())
        if k > best_n:
            best_n, best_nrm, best_a = k, nrm, a
    if best_nrm is None:
        raise ValueError("no axis-aligned plane found; pass --up explicitly")
    axis = int(np.argmax(np.abs(best_nrm)))
    h = p[:, axis]
    lo, hi = np.quantile(h, [0.005, 0.995])
    near_lo = ((h > lo + 0.1) & (h < lo + 0.8)).sum()
    near_hi = ((h < hi - 0.1) & (h > hi - 0.8)).sum()
    if abs(near_lo - near_hi) > 0.1 * max(near_lo, near_hi):
        sign = "" if near_lo > near_hi else "-"
    else:                                    # empty room: assume the largest plane is the floor
        d = h - best_a[axis]
        sign = "" if (d > thresh).sum() >= (d < -thresh).sum() else "-"
    return sign + "xyz"[axis]


def load_points(path: str, n: int = 800_000, up: str | None = "y", seed: int = 0,
                voxel: float = 0.02, return_info: bool = False, scale: float = 1.0):
    """Load a scan as Z-up points. `up=None` (or "auto") detects the up axis.

    `scale` multiplies the coordinates first (video reconstructions have no metric units), so
    the voxel size and all later thresholds are in metres.
    Clouds over VOXEL_ABOVE points are voxel-downsampled to `voxel` metres first.
    With `return_info`, also returns {"up", "R", "n_raw", "n"}; R maps file frame -> Z-up.
    """
    pts = _read_geometry(path, n, seed) * scale
    n_raw = len(pts)
    if n_raw > VOXEL_ABOVE and voxel:
        pts = voxel_downsample(pts, voxel)
    if up in (None, "auto"):
        up = detect_up(pts, np.random.default_rng(seed))
    R = UP[up]
    pts = pts @ R.T
    if return_info:
        return pts, {"up": up, "R": R, "n_raw": n_raw, "n": len(pts)}
    return pts


def level_floor(pts: np.ndarray, rng: np.random.Generator, iters=400, thresh=0.015):
    """RANSAC the floor plane among the lowest points, rotate it flat, put it at z=0."""
    low = pts[pts[:, 2] <= np.quantile(pts[:, 2], 0.25)]     # <=: a perfectly flat floor is the quantile
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = low[rng.choice(len(low), 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        if np.linalg.norm(nrm) < 1e-9:
            continue
        nrm /= np.linalg.norm(nrm)
        if abs(nrm[2]) < 0.9:            # must be roughly horizontal (gravity is known from ARCore)
            continue
        inl = np.abs((low - a) @ nrm) < thresh
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    fl = low[best]
    centroid = fl.mean(0)
    _, _, vt = np.linalg.svd(fl - centroid, full_matrices=False)
    nrm = vt[2] * np.sign(vt[2][2])
    # Rotation taking nrm onto +z (Rodrigues).
    v = np.cross(nrm, [0, 0, 1.0])
    s, c = np.linalg.norm(v), nrm[2]
    if s < 1e-9:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
    out = (pts - centroid) @ R.T
    tilt = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return out, tilt, best_n / len(low), (centroid, R)


def align_walls(pts, band=(0.05, 2.0), step_deg=0.5, sample=60_000, seed=0):
    """Rotate about z so the dominant walls line up with the x/y axes (Manhattan-world assumption).

    For each candidate yaw, wall points projected on x and y form sharp histogram peaks when
    the walls are axis-aligned; we keep the yaw with the sharpest peaks.
    """
    rng = np.random.default_rng(seed)
    w = pts[(pts[:, 2] > band[0]) & (pts[:, 2] < band[1])][:, :2]
    if len(w) > sample:
        w = w[rng.choice(len(w), sample, replace=False)]
    best, best_score = 0.0, -1
    for deg in np.arange(0, 90, step_deg):
        a = np.deg2rad(deg)
        c, s = np.cos(a), np.sin(a)
        rx = w[:, 0] * c - w[:, 1] * s
        ry = w[:, 0] * s + w[:, 1] * c
        score = sum((np.histogram(v, bins=np.arange(v.min(), v.max() + 0.05, 0.05))[0].astype(float) ** 2).sum()
                    for v in (rx, ry))
        if score > best_score:
            best, best_score = deg, score
    # Put the longer side of the room along x (landscape), for nicer plots.
    a = np.deg2rad(best)
    rx = w[:, 0] * np.cos(a) - w[:, 1] * np.sin(a)
    ry = w[:, 0] * np.sin(a) + w[:, 1] * np.cos(a)
    if np.ptp(ry) > np.ptp(rx):
        best -= 90
    a = np.deg2rad(best)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return pts @ Rz.T, best, Rz


def points_to_grid(pts, res=0.03, band=(0.05, 0.35), floor_tol=0.03, min_pts=2, margin=0.3):
    """Occupied = points inside the height band. Known = scanned floor (+ obstacles)."""
    xy_min = pts[:, :2].min(0) - margin
    xy_max = pts[:, :2].max(0) + margin
    W, H = np.ceil((xy_max - xy_min) / res).astype(int)

    def raster(p):
        c = ((p[:, 0] - xy_min[0]) / res).astype(int)
        r = ((p[:, 1] - xy_min[1]) / res).astype(int)
        g = np.zeros((H, W), np.int32)
        np.add.at(g, (np.clip(r, 0, H - 1), np.clip(c, 0, W - 1)), 1)
        return g

    in_band = (pts[:, 2] > band[0]) & (pts[:, 2] < band[1])
    occ = raster(pts[in_band]) >= min_pts
    occ = ndimage.binary_closing(occ, iterations=1) | occ          # seal pin-holes in walls
    floor = raster(pts[np.abs(pts[:, 2]) < floor_tol]) > 0
    known = ndimage.binary_closing(floor, iterations=3)
    known = ndimage.binary_fill_holes(known | occ)
    known |= ndimage.binary_dilation(occ, iterations=1)
    # Keep only the largest connected free area. Enclosed pockets such as the inside of a sofa or
    # cabinet become 'unknown'.
    lab, n = ndimage.label(known & ~occ)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        main_free = lab == (1 + int(np.argmax(sizes)))
        known = main_free | (known & occ) | ndimage.binary_dilation(occ, iterations=1) & ndimage.binary_dilation(main_free, iterations=2)
    return GridMap(occ=occ, res=res, origin=(float(xy_min[0]), float(xy_min[1])), known=known)


def add_scale_args(ap):
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the scan coordinates by this factor (for scans without metric units)")


def resolve_scale(args) -> float:
    """Final scale factor from --scale; prints what it used."""
    if args.scale != 1.0:
        print(f"scale {args.scale:.5f}")
    return args.scale


