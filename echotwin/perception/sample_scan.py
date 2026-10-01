"""Generate a synthetic 'phone scan' of a flat so the pipeline can be tested without a real scan.

Writes:
  data/sample_scan.ply     noisy point cloud, Y-up like ARCore, slightly tilted (what the phone gives)
  data/world_truth.npy     exact geometry of the real room, including things the camera missed or
                           that moved after the scan (glass partition, moved chair). Only the
                           simulator uses it, as ground truth. The scan also gets a 1.5 % scale
                           error, typical of phone tracking drift, so map and reality never match exactly.
"""
from pathlib import Path

import numpy as np
import trimesh

OUT = Path(__file__).resolve().parents[2] / "data"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)


def box(x0, x1, y0, y1, z0, z1):
    b = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return b


t = 0.1   # wall thickness
parts = [
    trimesh.Trimesh([[0, 0, 0], [7, 0, 0], [7, 5, 0], [0, 5, 0]], [[0, 1, 2], [0, 2, 3]]),  # floor (top surface only, as a phone sees it)
    box(-t, 7 + t, -t, 0, 0, 2.5), box(-t, 7 + t, 5, 5 + t, 0, 2.5),   # outer walls
    box(-t, 0, 0, 5, 0, 2.5), box(7, 7 + t, 0, 5, 0, 2.5),
    box(4.0, 4.1, 0, 3.0, 0, 2.5),                   # interior wall, doorway from y=3 to 5
    box(0.3, 2.3, 4.1, 4.9, 0, 0.8),                 # sofa
    box(1.5, 2.7, 1.5, 2.4, 0.71, 0.75),             # table top (above the sonar band)
    *[box(x, x + 0.05, y, y + 0.05, 0, 0.71)         # table legs (inside the band)
      for x in (1.55, 2.6) for y in (1.55, 2.3)],
    box(5.0, 6.2, 2.0, 2.8, 0, 0.9),                 # kitchen island
    box(6.4, 6.95, 3.4, 4.95, 0, 2.0),               # tall cabinet
    box(4.7, 5.4, 4.1, 4.8, 0, 0.45),                # armchair
    box(0.05, 0.4, 1.0, 2.0, 0.5, 1.5),              # wall shelf above the band (not an obstacle)
    box(4.1, 4.5, 0.1, 1.2, 0, 0.75),                # low sideboard
]
scene = trimesh.util.concatenate(parts)
pts, _ = trimesh.sample.sample_surface(scene, 900_000, seed=1)
pts += rng.normal(0, 0.008, pts.shape)               # depth noise, ~1 cm
# Things the phone scan does NOT contain but the robot will meet.
extras = trimesh.util.concatenate([
    box(5.0, 6.2, 0.95, 1.0, 0, 2.0),                # glass partition: invisible to the camera, solid to sound
    box(2.9, 3.35, 3.2, 3.65, 0, 0.45),              # chair moved after the scan
])
ex, _ = trimesh.sample.sample_surface(extras, 60_000, seed=2)


def to_phone_frame(p):
    """Z-up design frame -> ARCore-like frame: Y up, 1.5 deg tilt, arbitrary origin and yaw."""
    yaw = np.deg2rad(23)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    tilt = np.deg2rad(1.5)
    Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    p = (p - [3.2, 2.1, 1.35]) @ (Rx @ Rz).T          # phone starts ~1.35 m above floor
    return p @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T   # z-up -> y-up

truth, _ = trimesh.sample.sample_surface(scene, 900_000, seed=3)
trimesh.PointCloud(to_phone_frame(pts) * 1.015).export(OUT / "sample_scan.ply")
np.save(OUT / "world_truth.npy", to_phone_frame(np.vstack([truth, ex])).astype(np.float32))
print(f"wrote {OUT/'sample_scan.ply'} ({len(pts):,} points) and world_truth.npy")
