"""scene.json: what perception found, in one file that both halves read. See schema.md.

Numpy-free. Perception writes it (`build_scene`), the twin builder and the sonar side read it.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import catalog

FORMAT = "echotwin-scene"
VERSION = 1
SOURCES = ("yolo", "nvidia", "quick", "manual")
ON_MARGIN = 0.05           # an item counts as on a surface when its centre is this close to the surface outline
ON_HEIGHT = 0.10           # ... and its base is within this of the surface top


class SceneError(ValueError):
    pass


def build_scene(objects: list[dict], map_meta: dict | None = None, name: str = "scene",
                source: str = "yolo", extra: dict | None = None) -> dict:
    """Detected objects ({class, x, y, size_x, size_y, height, base_z?, photos?, points?}) -> a scene dict.

    Fills in shape / movable / surface from the catalog and which surface each object stands on.
    """
    out = []
    for i, o in enumerate(objects):
        cls = str(o.get("class", "object"))
        sx, sy = float(o.get("size_x", 0.1)), float(o.get("size_y", 0.1))
        h = o.get("height")
        info = catalog.classify(cls, sx, sy, h)
        out.append({
            "id": f"o{i + 1}", "class": cls, "label": o.get("label") or info.get("label") or cls, "source": o.get("source", source),
            "conf": o.get("conf"), "x": float(o["x"]), "y": float(o["y"]), "size_x": sx, "size_y": sy,
            "height": None if h is None else float(h), "base_z": float(o.get("base_z", 0.0)),
            "shape": info["shape"], "movable": info["movable"], "surface": info["surface"], "on": None,
            "traits": list(o.get("traits") or []), "photos": o.get("photos"), "points": o.get("points"),
            "views": o.get("views") or [],
        })
    _assign_support(out)
    doc = {"format": FORMAT, "version": VERSION, "name": name, "map": map_meta or {}, "objects": out}
    if extra:
        doc.update(extra)
    validate(doc)
    return doc


def _assign_support(objs: list[dict]):
    """Set o["on"] to the id of the surface a small thing stands on."""
    for o in objs:
        o["on"] = None
        if o["surface"]:
            continue
        best, best_d = None, 1e9
        for s in objs:
            if not s["surface"] or s is o or s["height"] is None:
                continue
            dx = abs(o["x"] - s["x"]) - (s["size_x"] / 2 + ON_MARGIN)
            dy = abs(o["y"] - s["y"]) - (s["size_y"] / 2 + ON_MARGIN)
            if dx > 0 or dy > 0:
                continue
            if abs(o["base_z"] - s["height"]) > ON_HEIGHT or o["base_z"] < 0.5 * s["height"]:
                continue
            d = max(dx, dy)
            if best is None or d < best_d:
                best, best_d = s["id"], d
        o["on"] = best


def validate(doc: dict):
    if doc.get("format") != FORMAT:
        raise SceneError(f"Not a scene file (format={doc.get('format')!r}).")
    if doc.get("version") != VERSION:
        raise SceneError(f"Unsupported scene version {doc.get('version')!r}.")
    ids = set()
    for o in doc.get("objects", []):
        for k in ("id", "class", "x", "y", "size_x", "size_y", "shape", "movable"):
            if k not in o:
                raise SceneError(f"Object {o.get('id', '?')} has no '{k}'.")
        if o["shape"] not in catalog.SHAPES:
            raise SceneError(f"Object {o['id']}: shape must be one of {', '.join(catalog.SHAPES)}.")
        if o["id"] in ids:
            raise SceneError(f"Duplicate object id {o['id']}.")
        ids.add(o["id"])
    for o in doc.get("objects", []):
        if o.get("on") and o["on"] not in ids:
            raise SceneError(f"Object {o['id']} stands on {o['on']}, which does not exist.")


def load(path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(doc)
    return doc


def save(doc: dict, path):
    validate(doc)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")
