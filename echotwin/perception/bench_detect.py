"""Which detector finds and names the objects best? A small benchmark on the example photos.

    python -m echotwin.perception.bench_detect yolo11s-seg.pt yolo26s-seg.pt yoloe-26s-seg-pf.pt yoloe-26s-seg.pt:text

Each model is a weights file in models/ (git-ignored). Add ":text" to an open-vocabulary YOLOE model to give it a
text prompt list instead of its built-in names. Missing weights are only downloaded with --download.

What is measured, per model and scene (a scene = one folder of photos, see bench_data.json):
  recall       expected objects found: a detection with one of the item's accepted names, in enough photos
  false rate   share of detections whose name matches no expected item and is not on the ignore list
  false names  those names, when they show up in 2 or more photos
  ms / photo   time per photo after a warm-up, and peak GPU memory

It is a rough, scene-level, lenient benchmark written by hand for two scenes, not an academic one. The expected
lists and accepted names are in bench_data.json: read them before trusting a number.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from echotwin.perception import detectors

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).with_name("bench_data.json")
IMG_EXT = {".jpg", ".jpeg", ".png"}


# ---------------- scoring (no GPU, no models) ----------------
def score_scene(scene: dict, detections: dict[str, list[tuple[str, float]]], conf: float = 0.25) -> dict:
    """detections: {photo: [(name, confidence), ...]} -> metrics for one scene."""
    items = scene["items"]
    ignore = {n.lower() for n in scene.get("ignore", [])}
    accepted = {n.lower() for it in items for n in it["names"]}
    photos_with: dict[str, set[str]] = {}
    total = false = 0
    for photo, dets in detections.items():
        for name, c in dets:
            if c < conf:
                continue
            name = name.lower()
            photos_with.setdefault(name, set()).add(photo)
            total += 1
            if name not in accepted and name not in ignore:
                false += 1
    found, missing, seen = [], [], {}
    for it in items:
        names = {n.lower() for n in it["names"]}
        photos = set().union(*(photos_with.get(n, set()) for n in names)) if names else set()
        seen[it["id"]] = len(photos)
        (found if len(photos) >= it.get("min_photos", 2) else missing).append(it["id"])
    false_names = sorted(n for n, ps in photos_with.items() if len(ps) >= 2 and n not in accepted and n not in ignore)
    return {"recall": len(found) / len(items), "found": found, "missing": missing, "photos_seen": seen,
            "detections": total, "per_photo": total / max(1, len(detections)),
            "false_rate": false / total if total else 0.0, "false_names": false_names}


def summarize(results: dict[str, dict]) -> str:
    """{model: {"scenes": {scene: metrics}, "ms": ..., "gpu_mb": ...}} -> a Markdown table."""
    scenes = sorted({s for r in results.values() for s in r["scenes"]})
    head = "| Model | " + " | ".join(f"{s}: recall, false" for s in scenes) + " | ms / photo | GPU MB |"
    lines = [head, "|" + "---|" * (len(scenes) + 3)]
    for model, r in results.items():
        cells = [f"{r['scenes'][s]['recall']:.0%}, {r['scenes'][s]['false_rate']:.0%}" if s in r["scenes"] else "-"
                 for s in scenes]
        lines.append(f"| {model} | " + " | ".join(cells) + f" | {r['ms']:.0f} | {r['gpu_mb']:.0f} |")
    return "\n".join(lines)


# ---------------- models ----------------
class Detector:
    """One detector. predict(path) -> [(name, confidence)]."""

    def __init__(self, spec: str, download: bool = False, device: str | None = None):
        self.spec = spec
        self.model = detectors.load(spec, download=download)
        self.device = device

    def predict(self, img_path: Path, conf: float, imgsz: int = 640) -> list[tuple[str, float]]:
        res = self.model.predict(str(img_path), conf=conf, imgsz=imgsz, retina_masks=True, verbose=False,
                                 device=self.device)[0]
        if res.boxes is None:
            return []
        names = res.names
        return [(names[int(k)], float(c)) for k, c in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist())]


def run_model(spec: str, cfg: dict, conf: float, download: bool, scenes: list[str]) -> dict:
    import torch
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    det = Detector(spec, download)
    out = {"scenes": {}, "ms": 0.0, "gpu_mb": 0.0}
    times = []
    first = True
    for sname in scenes:
        scene = cfg["scenes"][sname]
        photos = sorted(p for p in (REPO / scene["folder"]).iterdir() if p.suffix.lower() in IMG_EXT)
        dets = {}
        for p in photos:
            if first:                                   # warm-up: the first call loads kernels
                det.predict(p, conf)
                first = False
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            dets[p.name] = det.predict(p, conf)
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        out["scenes"][sname] = score_scene(scene, dets, conf)
        out["scenes"][sname]["names"] = sorted({n for d in dets.values() for n, c in d if c >= conf})
    out["ms"] = 1000 * sum(times) / max(1, len(times))
    out["gpu_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else 0.0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="weights in models/ (add :text for YOLOE with the text prompt list)")
    ap.add_argument("--conf", type=float, default=0.25, help="minimum confidence")
    ap.add_argument("--scenes", nargs="+", help="scenes from bench_data.json (default: all)")
    ap.add_argument("--download", action="store_true", help="let Ultralytics download missing weights")
    ap.add_argument("--out", default="out/bench_detect.json", help="where to write the raw results")
    a = ap.parse_args(argv)
    cfg = json.loads(DATA.read_text(encoding="utf-8"))
    scenes = a.scenes or list(cfg["scenes"])
    results = {}
    for spec in a.models:
        print(f"== {spec}", flush=True)
        results[spec] = run_model(spec, cfg, a.conf, a.download, scenes)
        for s, m in results[spec]["scenes"].items():
            miss = ", ".join(f"{i} ({m['photos_seen'][i]} photos)" for i in m["missing"]) or "none"
            print(f"   {s}: recall {m['recall']:.0%} (missing: {miss}), false {m['false_rate']:.0%} {m['false_names']}",
                  flush=True)
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\n" + summarize(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
