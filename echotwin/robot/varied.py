"""Varied objects for the skill check: what people photograph is not four shapes at four sizes.

Each draw is one object, the way a scan would build it: a box of any proportions, a cylinder, a ball or an egg, or a
mesh revolved from a silhouette (lathe.py, as the scan does for rounded and tapered things): a bottle, a cup, a tapered
glass, a vase, a tub, an egg. Sizes are everyday ones (3 to 7 cm across, 2 to 14 cm tall), at any rotation. The
shape label given with it is how the twin draws it; the robot reads the built geometry (features/geometry.py).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .features import lathe

MESH_DIR = Path("out/varied_meshes")

# silhouettes, half-width from the bottom up, widest = 1 (lathe.profile's convention)
PROFILES = {
    "bottle": [1, 1, 1, 1, 1, 0.97, 0.85, 0.6, 0.4, 0.35, 0.35, 0.35],
    "cup": list(np.linspace(0.82, 1.0, 12)),
    "glass": list(np.linspace(0.7, 1.0, 12)),
    "vase": [0.7, 0.9, 1.0, 0.95, 0.8, 0.6, 0.5, 0.5, 0.6, 0.7, 0.75, 0.75],
    "tub": list(np.linspace(1.0, 0.8, 12)),
    "egg": list(np.clip(np.sqrt(np.clip(1 - np.linspace(-0.9, 0.95, 12) ** 2, 0, 1)) * 1.02, 0.5, 1.0)),
}
KINDS = ("box", "cylinder", "round", "egg", "flat", *PROFILES)


def _mesh(kind: str, width: float, height: float) -> str:
    prof = np.asarray(PROFILES[kind], float)
    key = hashlib.sha1(f"{kind}|{width:.4f}|{height:.4f}".encode()).hexdigest()[:12]
    path = MESH_DIR / f"{kind}_{key}.obj"
    if not path.exists():
        MESH_DIR.mkdir(parents=True, exist_ok=True)
        v, uv, f = lathe.build(prof, width, width, height)
        path.write_text(lathe.to_obj(v, uv, f), encoding="utf-8")
    return str(path.resolve())


def draw(rng, name: str = "mover", k: float = 1.0) -> dict:
    """One random object as a Layout prop (without its position: set "pos")."""
    kind = str(rng.choice(KINDS))
    w = float(rng.uniform(0.03, 0.07)) * k
    yaw = float(rng.uniform(0.0, 180.0))
    rgb = tuple(float(v) for v in rng.uniform(0.2, 0.9, 3))
    if kind == "box":
        size, shape = (w, w * float(rng.uniform(1.0, 1.8)), float(rng.uniform(0.03, 0.12)) * k), "box"
    elif kind == "flat":
        size, shape = (w * float(rng.uniform(1.2, 2.2)), w, float(rng.uniform(0.015, 0.025)) * k), "flat"
    elif kind == "cylinder":
        size, shape = (w, w, float(rng.uniform(0.03, 0.12)) * k), "cylinder"
    elif kind == "round":
        size, shape = (w, w, w), "round"
    elif kind == "egg":
        size, shape = (w, w, w * float(rng.uniform(1.25, 1.6))), "round"
    else:                                           # a scanned rounded or tapered thing
        h = float(rng.uniform(0.04, 0.14)) * k
        return {"name": name, "shape": "cylinder", "pos": (0.0, 0.0), "yaw": yaw, "size": (w / 2, w / 2, h / 2),
                "rgb": rgb, "mesh": _mesh(kind, w, h), "mesh_scale": (1.0, 1.0, 1.0), "kind": kind}
    return {"name": name, "shape": shape, "pos": (0.0, 0.0), "yaw": yaw, "size": tuple(v / 2 for v in size),
            "rgb": rgb, "kind": kind}
