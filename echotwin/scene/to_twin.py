"""scene.json -> robot twin file.

The robot works on a table, not a room. So this picks a table-sized window of the scene:
  1. a surface (dining table, couch, bed...) that has small things standing on it, or else
  2. the densest cluster of small movable things.
Everything movable inside the window becomes an object the gripper can move, real size and real place.
Furniture that reaches into the window (a chair next to the table, a big plant) becomes a fixed obstacle.
The window is scaled to the sim table the same way the one-photo importer does (`sim_scale`).

    python -m echotwin.scene.to_twin out/lounge_objects_scene.json --out data/lounge_twin.zip
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

from . import catalog, schema

# Sim table size in sim metres. Must equal 2 * TABLE_HALF in echotwin/robot/scene.py (a test checks this).
TABLE_SIM = (1.188, 0.84)
CLUSTER_RADIUS = 0.8           # metres: small things this close belong to one working area


class NoTable(schema.SceneError):
    pass


def pick_window(scene: dict) -> tuple[tuple[float, float], list[dict], dict | None]:
    """-> (centre x, y of the window, the movable items it works on, the surface or None)."""
    objs = scene["objects"]
    movable = [o for o in objs if o["movable"]]
    if not movable:
        raise NoTable("Nothing in this scene is small enough to move.")
    by_id = {o["id"]: o for o in objs}
    support: dict[str, list[dict]] = {}
    for o in movable:
        if o.get("on") in by_id:
            support.setdefault(o["on"], []).append(o)
    if support:
        sid = max(support, key=lambda k: len(support[k]))
        s, items = by_id[sid], support[sid]
        return (s["x"], s["y"]), items, s
    best = max(movable, key=lambda m: sum(_dist(m, n) <= CLUSTER_RADIUS for n in movable))
    items = [n for n in movable if _dist(best, n) <= CLUSTER_RADIUS]
    return (sum(i["x"] for i in items) / len(items), sum(i["y"] for i in items) / len(items)), items, None


def _holds(surface: dict, items: list[dict], margin: float = 0.05) -> bool:
    """Does this surface lie under at least one of the things to move? (Then it is the table they stand on,
    even when the detector did not link them.)"""
    return any(abs(i["x"] - surface["x"]) <= surface["size_x"] / 2 + margin
               and abs(i["y"] - surface["y"]) <= surface["size_y"] / 2 + margin for i in items)


def _dist(a: dict, b: dict) -> float:
    return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5


def window_size(items: list[dict], surface: dict | None) -> tuple[float, float]:
    """Real metres covered by the sim table (same margins as the one-photo importer)."""
    tw, th = TABLE_SIM
    xs = [i["x"] for i in items]
    ys = [i["y"] for i in items]
    span_x = max(0.5, (max(xs) - min(xs)) * 1.6 + 0.15)
    span_y = max(0.35, (max(ys) - min(ys)) * 1.6 + 0.15)
    if surface is not None:
        span_x, span_y = max(span_x, surface["size_x"]), max(span_y, surface["size_y"])
    real_w = max(span_x, span_y * tw / th)
    return real_w, real_w * th / tw


def scene_to_twin(scene: dict, name: str | None = None) -> dict:
    """-> a twin document (see echotwin/robot/twin_import/layout_file.py), real centimetres."""
    schema.validate(scene)
    (cx, cy), items, surface = pick_window(scene)
    real_w, real_h = window_size(items, surface)
    sim_scale = TABLE_SIM[0] / real_w
    props, obstacles, left_out = [], [], 0
    for o in scene["objects"]:
        if o["surface"] and (o is surface or _holds(o, items)):
            continue                                   # the table the objects stand on is the sim table
        dx, dy = o["x"] - cx, o["y"] - cy
        # something to move must be on the table; furniture only has to reach into the window
        reach_x = 0.0 if o["movable"] else o["size_x"] / 2
        reach_y = 0.0 if o["movable"] else o["size_y"] / 2
        if abs(dx) > real_w / 2 + reach_x or abs(dy) > real_h / 2 + reach_y:
            left_out += 1
            continue
        h = float(o.get("height") or 0.1)
        sx, sy = o["size_x"], o["size_y"]
        if not o["movable"]:                           # keep only the part of the furniture over the table
            x0, x1 = max(dx - sx / 2, -real_w / 2), min(dx + sx / 2, real_w / 2)
            y0, y1 = max(dy - sy / 2, -real_h / 2), min(dy + sy / 2, real_h / 2)
            dx, dy, sx, sy = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        entry = {"name": _name(o, scene), "shape": o["shape"],
                 "size_cm": [round(max(2.0, sx * 100), 1), round(max(2.0, sy * 100), 1),
                             round(max(1.0, h * 100), 1)],
                 "pos_cm": [round(dx * 100, 1), round(dy * 100, 1)], "yaw_deg": 0.0,
                 "color": catalog.color_for(o["id"] if o["label"] == "object" else o["class"])}
        if not o["movable"] and entry["shape"] == "cylinder":
            entry["shape"] = "box"                     # a clipped cylinder is not a cylinder any more
        (props if o["movable"] else obstacles).append(entry)
    return {"format": "phone-puppeteer-twin", "version": 1, "name": name or scene.get("name", "EchoTwin scene"),
            "sim_scale": round(sim_scale, 4), "objects": props, "obstacles": obstacles, "scene": [],
            "window_m": [round(real_w, 3), round(real_h, 3)], "left_out": left_out}


def _name(o: dict, scene: dict) -> str:
    same = [x for x in scene["objects"] if x["label"] == o["label"]]
    return o["label"] if len(same) == 1 else f"{o['label']} {same.index(o) + 1}"


def write_twin(doc: dict, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".zip":
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("twin.json", json.dumps(doc, indent=1))
    else:
        out.write_text(json.dumps(doc, indent=1), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="scene.json written by echotwin.perception.objects")
    ap.add_argument("--out", default="twin.zip", help="output .zip or .json")
    ap.add_argument("--name", help="twin name shown in the dashboard")
    a = ap.parse_args(argv)
    try:
        doc = scene_to_twin(schema.load(a.scene), a.name)
    except schema.SceneError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    write_twin(doc, Path(a.out))
    print(f"{len(doc['objects'])} objects to move, {len(doc['obstacles'])} fixed obstacles, "
          f"{doc['left_out']} left out, window {doc['window_m'][0]:.2f} x {doc['window_m'][1]:.2f} m "
          f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
