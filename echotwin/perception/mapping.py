"""Turn a phone scan (mesh or point cloud, metric scale) into a 2D occupancy grid
at the height of the robot's sonars.

Steps: sample points -> put gravity on +z -> find the floor (RANSAC) and level it
-> keep the horizontal band the sonars can see -> rasterise -> clean up.
"""
from __future__ import annotations

import argparse

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


def _ply_header(path: str) -> tuple[list[str], int, int, bool]:
    """(vertex property names, vertex count, header bytes, all-float binary LE) of a .ply file."""
    with open(path, "rb") as fh:
        if fh.read(3) != b"ply":
            return [], 0, 0, False
        fh.seek(0)
        lines = []
        while not lines or lines[-1] != b"end_header":
            line = fh.readline()
            if not line:
                return [], 0, 0, False
            lines.append(line.strip())
        size = fh.tell()
    elems = [l for l in lines if l.startswith(b"element")]
    if not elems or not elems[0].startswith(b"element vertex"):
        return [], 0, 0, False
    props = [l.split() for l in lines if l.startswith(b"property")]
    ok = (b"format binary_little_endian 1.0" in lines and len(elems) == 1
          and all(q[1] in (b"float", b"float32") for q in props))
    return [q[-1].decode() for q in props], int(elems[0].split()[-1]), size, ok


def read_splat(path: str, min_opacity=0.5, max_size=0.05):
    """Centres of a 3D Gaussian Splatting .ply, or None if the file is not one.

    Faint splats (opacity < min_opacity) and large ones (largest axis > max_size m) are dropped:
    they are floaters and background blobs, not surfaces.
    """
    props, n, off, ok = _ply_header(path)
    if not ok or not {"opacity", "scale_0", "scale_1", "scale_2"} <= set(props):
        return None
    a = np.fromfile(path, dtype="<f4", count=n * len(props), offset=off).reshape(n, len(props))
    ix = {q: i for i, q in enumerate(props)}
    opacity = 1 / (1 + np.exp(-a[:, ix["opacity"]].astype(float)))
    size = np.exp(a[:, [ix["scale_0"], ix["scale_1"], ix["scale_2"]]].astype(float)).max(1)
    return a[(opacity >= min_opacity) & (size <= max_size)][:, [ix["x"], ix["y"], ix["z"]]].astype(float)


def crop_densest(pts: np.ndarray, radius: float, cell=0.5) -> np.ndarray:
    """Keep points within `radius` m (horizontally) of the densest `cell` x `cell` column.

    For scans that also caught the rest of a hall or the view through windows."""
    xy = pts[:, :2]
    bins = [np.arange(xy[:, k].min(), xy[:, k].max() + cell, cell) for k in (0, 1)]
    H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins)
    i, j = np.unravel_index(H.argmax(), H.shape)
    c = np.array([xe[i] + cell / 2, ye[j] + cell / 2])
    return pts[np.hypot(*(xy - c).T) <= radius]


def _read_geometry(path: str, n: int, seed: int) -> np.ndarray:
    """All points of a mesh, point cloud, Gaussian splat or multi-part scene, in the file's frame.

    Scenes (e.g. .glb with one mesh per room chunk) are flattened with their node transforms.
    Meshes are sampled by area over the whole scene; point clouds are used as they are;
    splats contribute the centres of their opaque, small splats. Colours and textures are ignored.
    """
    if path.lower().endswith(".ply"):
        pts = read_splat(path)
        if pts is not None:
            return pts[np.isfinite(pts).all(1)]
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
    """Occupied = points inside the sonar band. Known = scanned floor (+ obstacles)."""
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
    # Keep only the largest connected free area (where the robot can drive). Enclosed pockets
    # such as the inside of a sofa or cabinet become 'unknown'.
    lab, n = ndimage.label(known & ~occ)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        main_free = lab == (1 + int(np.argmax(sizes)))
        known = main_free | (known & occ) | ndimage.binary_dilation(occ, iterations=1) & ndimage.binary_dilation(main_free, iterations=2)
    return GridMap(occ=occ, res=res, origin=(float(xy_min[0]), float(xy_min[1])), known=known)


def wall_faces(pts, band=(0.05, 1.0), bin_m=0.01, rel_height=0.15):
    """Positions of wall faces along x and y in a levelled, wall-aligned cloud.

    Peaks of a 1 cm histogram of the points in `band`. The distance between the two outermost
    peaks on an axis is the wall-to-wall length to compare with a tape measure.
    """
    from scipy.signal import find_peaks
    b = pts[(pts[:, 2] > band[0]) & (pts[:, 2] < band[1])]
    out = []
    for ax in (0, 1):
        edges = np.arange(b[:, ax].min(), b[:, ax].max() + bin_m, bin_m)
        h = np.pad(np.histogram(b[:, ax], edges)[0], 1)       # so a face in the end bin is a peak
        pk, _ = find_peaks(h, height=h.max() * rel_height, distance=max(1, round(0.05 / bin_m)))
        out.append(edges[pk - 1] + bin_m / 2)
    return out


def quality_report(pts, gmap: GridMap, floor_tol=0.03, wall_reach=0.06, min_gap=0.10) -> dict:
    """Numbers to judge a scan before using it.

    floor_cov: share of the room footprint (outline filled in) where floor points were seen.
    gaps: stretches of the room outline with no wall within `wall_reach`, longer than `min_gap`
    (doorways show up here too), as (length m, x, y of centre).
    """
    floor = np.zeros_like(gmap.occ)
    fp = pts[np.abs(pts[:, 2]) < floor_tol]
    r, c = gmap.to_cell(fp[:, 0], fp[:, 1])
    ok = gmap.inside(r, c)
    floor[r[ok], c[ok]] = True
    room = ndimage.binary_fill_holes(ndimage.binary_closing(floor | gmap.occ, iterations=3))
    lab, n = ndimage.label(room)
    if n > 1:
        room = lab == 1 + int(np.argmax(ndimage.sum(room, lab, np.arange(1, n + 1))))
    inner = room & ~gmap.occ
    cov = float((floor & inner).sum() / max(inner.sum(), 1))
    edge = room & ~ndimage.binary_erosion(room)
    near = ndimage.binary_dilation(gmap.occ, iterations=max(1, round(wall_reach / gmap.res)))
    lab, n = ndimage.label(edge & ~near, structure=np.ones((3, 3)))
    gaps = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        rr, cc = np.nonzero(lab[sl] == i)
        length = max(np.ptp(rr), np.ptp(cc)) * gmap.res + gmap.res
        if length >= min_gap:
            gaps.append((float(length), gmap.origin[0] + (sl[1].start + cc.mean() + 0.5) * gmap.res,
                         gmap.origin[1] + (sl[0].start + rr.mean() + 0.5) * gmap.res))
    gaps.sort(reverse=True)
    rows, cols = np.nonzero(room)
    size = ((np.ptp(cols) + 1) * gmap.res, (np.ptp(rows) + 1) * gmap.res) if len(rows) else (0.0, 0.0)
    return {"floor_cov": cov, "gaps": gaps, "room_size": size, "map_size": (gmap.W * gmap.res, gmap.H * gmap.res)}


def format_report(q: dict) -> str:
    g = q["gaps"]
    lines = [f"quality: floor coverage {q['floor_cov']:.0%} | room {q['room_size'][0]:.2f} x {q['room_size'][1]:.2f} m"
             f" | map {q['map_size'][0]:.2f} x {q['map_size'][1]:.2f} m",
             f"wall gaps on outline: {len(g)} ({sum(x[0] for x in g):.2f} m total, doorways included)"]
    lines += [f"  {L:.2f} m at x={x:.2f}, y={y:.2f}" for L, x, y in g[:5]]
    if q["floor_cov"] < 0.6:
        lines.append("  warning: floor coverage under 60 %; check --up and --band, or rescan lower")
    return "\n".join(lines)


def add_scale_args(ap):
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the scan coordinates by this factor (for scans without metric units)")
    ap.add_argument("--ref", type=float, nargs=5, metavar=("AX", "AY", "BX", "BY", "METRES"),
                    help="two map points (aligned frame, as printed by echotwin.perception.measure with the same "
                         "--scale) and their tape distance; multiplies --scale by METRES / |AB|")


def add_crop_arg(ap):
    ap.add_argument("--crop", type=float, default=0.0, metavar="R",
                    help="keep only points within R m of the densest area (scans of a bigger space); 0 = off")
    ap.add_argument("--floor-offset", type=float, default=0.0, metavar="H",
                    help="the robot drives on a surface H m above the detected floor (e.g. a raised platform)")


def resolve_scale(args) -> float:
    """Final scale factor from --scale and --ref; prints what it used."""
    scale = args.scale
    if args.ref:
        ax, ay, bx, by, metres = args.ref
        d = float(np.hypot(bx - ax, by - ay))
        if d <= 0 or metres <= 0:
            raise SystemExit("--ref: points must differ and METRES must be > 0")
        scale *= metres / d
        print(f"--ref: |AB| = {d:.4f} (scan units x {args.scale:g}), tape {metres:.3f} m -> scale {scale:.5f}")
    elif scale != 1.0:
        print(f"scale {scale:.5f}")
    return scale


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scan", help="mesh or point cloud exported from the phone (.glb/.obj/.ply)")
    ap.add_argument("-o", "--out", default="map", help="output stem -> <out>.png + <out>.yaml")
    ap.add_argument("--up", choices=["auto", *UP], default="auto",
                    help="up axis of the file (ARCore/glTF: y); default: detect from the largest plane")
    ap.add_argument("--res", type=float, default=0.03, help="cell size in metres")
    ap.add_argument("--band", type=float, nargs=2, default=(0.05, 0.35),
                    help="height band (m) the sonars can see: sensor height +/- half the beam spread")
    add_scale_args(ap)
    add_crop_arg(ap)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    pts, info = load_points(args.scan, up=args.up, return_info=True, scale=resolve_scale(args),
                            voxel=min(0.02, args.res / 2))          # never coarser than half a cell
    print(f"loaded {info['n_raw']:,} points -> {info['n']:,} used | up axis: {info['up']}"
          + (" (auto)" if args.up == "auto" else ""))
    pts, tilt, frac, _ = level_floor(pts, rng)
    if args.crop:
        pts = crop_densest(pts, args.crop)
        print(f"cropped to {args.crop:g} m around the densest area: {len(pts):,} points")
    if args.floor_offset:
        pts = pts - [0, 0, args.floor_offset]
    pts, yaw, _ = align_walls(pts)
    gmap = points_to_grid(pts, res=args.res, band=tuple(args.band))
    gmap.save(args.out)
    print(f"walls aligned: rotated {yaw:.1f} deg | floor tilt corrected: {tilt:.2f} deg | floor inliers: {frac:.0%} of low points")
    print(f"map: {gmap.W}x{gmap.H} cells @ {args.res*100:.0f} cm -> {args.out}.png / {args.out}.yaml")
    print(format_report(quality_report(pts, gmap)))


if __name__ == "__main__":
    main()
