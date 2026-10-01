"""End-to-end demo: phone scan -> occupancy grid -> simulated sonar robot -> particle filter.

    python -m echotwin.perception.sample_scan          # or use your own phone scan
    python -m echotwin.navigation.demo                  # map + one animated run + benchmark
    python -m echotwin.navigation.demo --scan my_room.ply --no-bench
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]

from echotwin.perception.gridmap import GridMap
from echotwin.perception.mapping import (add_crop_arg, add_scale_args, align_walls, crop_densest,
                                   format_report, level_floor, load_points, points_to_grid,
                                   quality_report, resolve_scale)
from echotwin.navigation.simulate import add_clutter, run_episode
from echotwin.navigation.sonar import SonarRig
from echotwin.navigation import viz


def build_maps(scan, truth_path, up, res, band, scale=1.0, crop=0.0, floor_offset=0.0):
    rng = np.random.default_rng(0)
    pts, info = load_points(str(scan), up=up, return_info=True, scale=scale, voxel=min(0.02, res / 2))
    print(f"loaded {info['n_raw']:,} points -> {info['n']:,} used | up axis: {info['up']}")
    pts, tilt, frac, (centroid, R) = level_floor(pts, rng)
    if crop:
        pts = crop_densest(pts, crop)
    pts = pts - [0, 0, floor_offset]
    pts, yaw, Rz = align_walls(pts)
    R = Rz @ R
    gmap = points_to_grid(pts, res=res, band=band)
    world = gmap
    if truth_path and Path(truth_path).exists():
        # Ground-truth room for the simulator, rasterised on the same grid as the map.
        tp = ((np.load(truth_path) * scale) @ info["R"].T - centroid) @ R.T
        tp = tp - [0, 0, floor_offset]
        tp = tp[(tp[:, 2] > band[0]) & (tp[:, 2] < band[1])]
        occ = np.zeros_like(gmap.occ)
        r, c = gmap.to_cell(tp[:, 0], tp[:, 1])
        ok = gmap.inside(r, c)
        occ[r[ok], c[ok]] = True
        occ |= ndimage.binary_closing(occ, iterations=1)
        world = GridMap(occ=occ, res=gmap.res, origin=gmap.origin, known=np.ones_like(occ))
    print(f"floor levelled (tilt {tilt:.2f} deg), walls aligned ({yaw:.1f} deg), map {gmap.W}x{gmap.H} cells at {res*100:.0f} cm")
    print(format_report(quality_report(pts, gmap)))
    return gmap, world


def load_objects(path, gmap):
    """Objects from object_map.py, or None if not given or built on a different map frame."""
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    m = data["map"]
    same = ((m["width"], m["height"]) == (gmap.W, gmap.H) and abs(m["res"] - gmap.res) < 1e-9
            and np.allclose(m["origin"], gmap.origin, atol=gmap.res / 2))
    if not same:
        print(f"objects: {path} was built on a different map (other flags or cloud); not drawn")
        return None
    print(f"objects: {len(data['objects'])} labels drawn in the GIF")
    return data["objects"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=ROOT / "data/sample_scan.ply")
    ap.add_argument("--truth", default=ROOT / "data/world_truth.npy",
                    help="ground-truth room points for the simulator (optional; defaults to the map itself)")
    ap.add_argument("--up", default="auto", help="up axis of the scan file (y, z, ...); default: auto-detect")
    ap.add_argument("--res", type=float, default=0.03)
    ap.add_argument("--band", type=float, nargs=2, default=(0.05, 0.35))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--objects", help="objects JSON from echotwin.perception.objects to label in the GIF (same scan and flags)")
    ap.add_argument("--clutter", type=int, default=0,
                    help="add N box obstacles to the simulated world that are not in the map (0 = off)")
    ap.add_argument("--out", default=ROOT / "out")
    add_scale_args(ap)
    add_crop_arg(ap)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    gmap, world = build_maps(a.scan, a.truth, a.up, a.res, tuple(a.band), resolve_scale(a), a.crop, a.floor_offset)
    if a.clutter:
        world = add_clutter(world, a.clutter, np.random.default_rng(11))
        print(f"world: {a.clutter} unmapped obstacles added (the filter does not see them)")
    # Without ground truth the scan edge is all we know (open plan, raised platform): the robot stops
    # there (bumper / cliff sensor) but the sonars see nothing, so it is not added to the world.
    drive = None if a.truth and Path(a.truth).exists() else world.with_extra(~gmap.known)
    gmap.save(out / "map")
    differs = []                                           # plot caption: how the world differs from the map
    if a.truth and Path(a.truth).exists():
        sample = Path(a.truth).name == "world_truth.npy"
        differs.append("glass, moved chair, 1.5 % scale drift" if sample else f"ground truth {Path(a.truth).name}")
    if a.clutter:
        differs.append(f"{a.clutter} unmapped obstacles")
    viz.plot_maps(gmap, world, out / "map_vs_world.png", ", ".join(differs))

    fx, _ = (drive or world).free_cells(0.35)
    if len(fx) == 0:                                  # e.g. a table top: nowhere with 35 cm clearance
        reason = "no free area with 35 cm clearance for the robot (map too small, e.g. a table top)"
        print(f"robot run skipped: {reason}")
        (out / "summary.json").write_text(json.dumps({"skipped": reason, "map_m": [
            round(gmap.W * gmap.res, 2), round(gmap.H * gmap.res, 2)]}))
        return
    rig = SonarRig()
    t0 = time.time()
    ep = run_episode(gmap, world, rig, seed=3, steps=a.steps, kidnap_at=int(a.steps * 0.6), snapshot_every=2, drive=drive)
    print(f"demo run: {time.time()-t0:.1f}s, converged at step {ep.convergence()}, "
          f"driven {ep.dist:.1f} m, kidnapped at step {int(a.steps*0.6)}")
    viz.animate(gmap, world, rig, ep, out / "demo_run.gif", objects=load_objects(a.objects, gmap))
    viz.plot_errors(ep, out / "demo_errors.png", kidnap_at=int(a.steps * 0.6))
    kid, k, dt = int(a.steps * 0.6), ep.convergence(), ep.t[1] - ep.t[0]
    pos, _ = ep.errors()
    after = pos[kid:]
    back = next((i for i in range(len(after) - 10) if (after[i:i + 10] < 0.3).all()), None)
    (out / "summary.json").write_text(json.dumps({             # for the web app's result tiles
        "converged_s": None if k is None else round(k * dt, 1),
        "error_cm": None if k is None else round(float(pos[k:kid if k < kid else None].mean() * 100), 1),
        "kidnap_recovery_s": None if back is None else round(back * dt, 1),
        "driven_m": round(ep.dist, 1), "map_m": [round(gmap.W * gmap.res, 2), round(gmap.H * gmap.res, 2)],
    }))

    if a.no_bench:
        return
    configs = {
        "4 sonars (F/L/R/B)": SonarRig(),
        "2 sonars (F/B)": SonarRig(mounts=[(0.10, 0, 0), (-0.10, 0, np.pi)]),
        "4 sonars, narrow 10 deg beam": SonarRig(fov=np.deg2rad(10), n_rays=3),
        "4 sonars, harsh (30 deg beam, 3x noise, 8 % ghost echoes)": SonarRig(
            fov=np.deg2rad(30), sigma0=0.02, sigma_k=0.03, p_spurious=0.08, specular_angle=np.deg2rad(30)),
    }
    rows = []
    for name, r in configs.items():
        res = []
        for s in range(a.runs):
            e = run_episode(gmap, world, r, seed=100 + s, steps=a.steps, drive=drive)
            k = e.convergence()
            pos, ang = e.errors()
            tail = slice(k, None) if k is not None else slice(len(pos), None)
            dist_to_conv = None
            if k is not None:
                tr = np.array(e.true)[: k + 1, :2]
                dist_to_conv = float(np.hypot(*np.diff(tr, axis=0).T).sum())
            res.append((k, dist_to_conv, pos[tail].mean() if k is not None else np.nan,
                        np.degrees(ang[tail]).mean() if k is not None else np.nan))
        ks = [x for x in res if x[0] is not None]
        row = dict(config=name, runs=a.runs, converged=len(ks),
                   median_dist_m=np.median([x[1] for x in ks]) if ks else np.nan,
                   median_time_s=np.median([x[0] for x in ks]) * 0.25 if ks else np.nan,
                   pos_err_cm=np.nanmean([x[2] for x in ks]) * 100 if ks else np.nan,
                   heading_err_deg=np.nanmean([x[3] for x in ks]) if ks else np.nan)
        rows.append(row)
        print(row)
    viz.write_benchmark(rows, out / "benchmark.md")


if __name__ == "__main__":
    main()
