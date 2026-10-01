"""Pack map + ground truth into one PNG for the browser simulator (task T2).

Channels: R = map occupied, G = map known, B = ground-truth occupied. Row 0 of the image is the
top (highest y), as in map_server PNGs. Prints resolution and origin to put in the page.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
from echotwin.navigation.demo import build_maps

g, w = build_maps(ROOT / "data/sample_scan.ply", ROOT / "data/world_truth.npy", "y", 0.03, (0.05, 0.35))
img = np.zeros(g.occ.shape + (3,), np.uint8)
img[..., 0] = g.occ * 255
img[..., 1] = g.known * 255
img[..., 2] = w.occ * 255
out = ROOT / "out/web_maps.png"
Image.fromarray(np.flipud(img)).save(out, optimize=True)
print(f"{out}  resolution={g.res}  origin=({g.origin[0]:.4f}, {g.origin[1]:.4f})  size={g.W}x{g.H}")
