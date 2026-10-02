"""Step 5: a vision model reviews what YOLO found, photo by photo.

    python -m echotwin.perception.review data/lounge.ply out/lounge_scene.json

Needs the cloud's .pix.npz (the photos) next to the cloud and AI_API_KEY in .env. For each of a few photos
that see the objects best, the objects are drawn as numbered yellow boxes and the vision model says what each
one really is. Answers from several photos are voted per object; one photo alone never removes an object. Results: the scene file is updated in place
(new names, shapes, movable flags, traits, false detections removed) and the marked photos are saved in
<scene>_review/ so people can check the review. Without a key nothing changes.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from echotwin.scene import review as R
from echotwin.scene import schema

MAX_FRAMES = 4
PAD = 0.35                 # box margin, as a fraction of the box size
SEND_SIDE = 768            # long side of the photo sent to the model


def select_frames(scene: dict, max_frames: int = MAX_FRAMES, want: int = 2) -> dict[int, list[tuple[int, list[float]]]]:
    """Pick the photos to send: {frame: [(object number, box), ...]}. Numbers are 1-based positions in the scene.

    Greedy: each pick is the photo that helps most objects that have fewer than `want` opinions so far, so every
    object is seen twice where the photos allow it (one photo alone never removes an object).
    """
    by_frame: dict[int, list[tuple[int, list[float]]]] = {}
    for n, o in enumerate(scene["objects"], 1):
        for v in o.get("views") or []:
            by_frame.setdefault(v["frame"], []).append((n, v["box"]))
    chosen: dict[int, list] = {}
    seen: dict[int, int] = {}
    while len(chosen) < max_frames and by_frame.keys() - chosen.keys():
        todo = by_frame.keys() - chosen.keys()
        best = max(todo, key=lambda f: (sum(seen.get(n, 0) < want for n, _ in by_frame[f]), len(by_frame[f])))
        chosen[best] = by_frame[best]
        for n, _ in by_frame[best]:
            seen[n] = seen.get(n, 0) + 1
    return chosen


def mark_frame(img: np.ndarray, marks: list[tuple[int, list[float]]]) -> Image.Image:
    """The photo with a numbered yellow box per object (boxes are fractions of the photo)."""
    im = Image.fromarray(np.ascontiguousarray(img)).convert("RGB")
    im.thumbnail((SEND_SIDE, SEND_SIDE))
    d = ImageDraw.Draw(im)
    w, h = im.size
    for n, (x0, y0, x1, y1) in marks:
        px, py = PAD * (x1 - x0), PAD * (y1 - y0)         # the points often cover only part of the object
        x0, y0, x1, y1 = max(0.0, x0 - px), max(0.0, y0 - py), min(1.0, x1 + px), min(1.0, y1 + py)
        box = [x0 * w, y0 * h, x1 * w, y1 * h]
        d.rectangle(box, outline=(0, 0, 0), width=5)
        d.rectangle(box, outline=(255, 220, 40), width=2)
        d.rectangle([box[0], box[1], box[0] + 22 + 8 * (n > 9), box[1] + 20], fill=(255, 220, 40))
        d.text((box[0] + 5, box[1] + 4), str(n), fill=(0, 0, 0))
    return im


def jpeg(im: Image.Image, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def review_scene(scene: dict, frames: dict[int, np.ndarray], out_dir: Path | None = None,
                 max_frames: int = MAX_FRAMES, cfg: dict | None = None, ask=R.ask_json) -> dict | None:
    """Run the review on a scene (changed in place). Returns the report, or None when no review happened."""
    cfg = cfg or R.config()
    if not cfg["key"]:
        print("No AI_API_KEY: keeping the detector's names.")
        return None
    names = {n: o["label"] for n, o in enumerate(scene["objects"], 1)}
    ids = {n: o["id"] for n, o in enumerate(scene["objects"], 1)}
    answers: dict[str, list[dict]] = {}
    for f, marks in select_frames(scene, max_frames).items():
        if f not in frames:
            continue
        im = mark_frame(frames[f], marks)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            im.save(out_dir / f"photo_{f:02d}.jpg", quality=85)
        seen = sorted({n for n, _ in marks})
        got = R.parse_answer(ask(jpeg(im), R.build_prompt([(n, names[n]) for n in seen]), cfg))
        print(f"photo {f}: {len(got)} of {len(seen)} objects answered")
        for n, entry in got.items():
            if n in seen:                                   # ignore numbers we did not ask about
                answers.setdefault(ids[n], []).append(entry)
    if not answers:
        print("The vision model gave no usable answer: keeping the detector's names.")
        return None
    verdicts = {oid: R.vote(es) for oid, es in answers.items()}
    return R.apply_review(scene, verdicts, model=cfg["models"][0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cloud", help=".ply from reconstruct.py (its .pix.npz must be next to it)")
    ap.add_argument("scene", help="<stem>_scene.json from objects.py; updated in place")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="photos to send (one request each)")
    a = ap.parse_args(argv)
    scene = schema.load(a.scene)
    pix = np.load(Path(a.cloud).with_suffix(".pix.npz"))
    frames = {int(k.split("_")[1]): pix[k] for k in pix.files if k.startswith("frame_")}
    before = len(scene["objects"])
    rep = review_scene(scene, frames, Path(a.scene).with_suffix("").with_name(Path(a.scene).stem + "_review"),
                       a.max_frames)
    if rep is None:
        return 0
    schema.save(scene, a.scene)
    for old, new in rep["renamed"]:
        print(f"  renamed: {old} -> {new}")
    for r in rep["removed"]:
        print(f"  removed: {r['class']} ({r['id']})")
    print(f"reviewed {rep['reviewed']} of {before} objects with {rep['model']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
