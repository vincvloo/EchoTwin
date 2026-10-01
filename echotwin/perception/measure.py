"""Scale check: wall-face positions along map x and y, to compare with tape-measured distances.

    python -m echotwin.perception.measure data/my_room.glb
    python -m echotwin.perception.measure data/my_room.glb --up y --band 0.05 1.0
    python -m echotwin.perception.measure data/video_room.ply --scale 0.5     # unscaled video scan, rough guess first

Uses the same levelling and wall alignment as mesh_to_grid, so x and y match the map axes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


from echotwin.perception.mapping import (UP, add_crop_arg, add_scale_args, align_walls, crop_densest,
                                   level_floor, load_points, resolve_scale, wall_faces)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan")
    ap.add_argument("--up", choices=["auto", *UP], default="auto")
    ap.add_argument("--band", type=float, nargs=2, default=(0.05, 1.0),
                    help="height band (m) used to find wall faces")
    add_scale_args(ap)
    add_crop_arg(ap)
    a = ap.parse_args(argv)
    pts, info = load_points(a.scan, up=a.up, return_info=True, scale=resolve_scale(a))
    pts, *_ = level_floor(pts, np.random.default_rng(0))
    if a.crop:
        pts = crop_densest(pts, a.crop)
    pts = pts - [0, 0, a.floor_offset]
    pts, *_ = align_walls(pts)
    print(f"up axis: {info['up']}")
    for name, f in zip("xy", wall_faces(pts, band=tuple(a.band))):
        span = f"{f[-1] - f[0]:.3f} m" if len(f) > 1 else "n/a"
        print(f"{name} faces at {np.round(f, 3).tolist()} | outermost span {span}")


if __name__ == "__main__":
    main()
