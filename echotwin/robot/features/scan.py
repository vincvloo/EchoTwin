"""F0 tier 1: scan the place. Frames -> sheet pose per frame -> fused objects + stitched table texture.

Pipeline: decode -> sharpness + sheet + pose per frame -> orientation kept consistent across the sweep
(camera continuity) -> keep up to 8 sharp frames spread over viewing angles -> warp each top-down ->
detect blobs -> fuse (median position, vote on presence) -> stitch texture -> Layout for the sim.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import SCANS
from ..scene import CUBE_HALF, DEFAULT_ZONES, OBJECT_NAMES, SCALE, WS_HALF, ZONE_LABEL, Layout
from . import vision as V

MAX_VIEWS = 8


class NoSheet(Exception):
    pass


def _mm_to_sim(x, y):
    return x / 1000 * SCALE, y / 1000 * SCALE


def select_frames(frames: list[dict], k: int = MAX_VIEWS) -> list[dict]:
    """Sharpest frame per azimuth bin, then fill with the sharpest remaining."""
    if len(frames) <= k:
        return frames
    az = np.array([np.arctan2(f["pose"]["cam"][1], f["pose"]["cam"][0]) if f.get("pose") else 0 for f in frames])
    bins = np.linspace(az.min(), az.max() + 1e-6, k + 1)
    chosen = []
    for i in range(k):
        idx = [j for j in range(len(frames)) if bins[i] <= az[j] < bins[i + 1]]
        if idx:
            chosen.append(max(idx, key=lambda j: frames[j]["sharp"]))
    rest = sorted(set(range(len(frames))) - set(chosen), key=lambda j: -frames[j]["sharp"])
    chosen += rest[:k - len(chosen)]
    return [frames[j] for j in sorted(chosen)]


def run_scan(jpegs: list[bytes], progress=lambda *a: None) -> tuple[Layout, dict]:
    t0 = time.time()
    scan_id = time.strftime("%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"
    out = SCANS / scan_id
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, jb in enumerate(jpegs):
        img = V.decode(jb)
        if img is None:
            continue
        f = {"i": i, "img": img, "sharp": V.sharpness(img), "corners": V.find_sheet(img)}
        f["cands"] = V.pose_candidates(f["corners"], img.shape) if f["corners"] is not None else []
        frames.append(f)
    with_sheet = [f for f in frames if f["cands"]]
    progress("frames", {"received": len(frames), "with_sheet": len(with_sheet)})

    # keep the orientation consistent: the first view defines "front", later views follow the camera
    prev = None
    for f in with_sheet:
        f["pose"] = V.choose_pose(f["cands"], prev)
        prev = f["pose"]["cam"]

    # drop blurry frames (relative to this sweep), then pick views spread over angles
    if len(with_sheet) > 10:  # video sweeps only; hand-taken photos are all kept
        cut = np.percentile([f["sharp"] for f in with_sheet], 30)
        with_sheet = [f for f in with_sheet if f["sharp"] >= cut] or with_sheet
    views = select_frames(with_sheet)
    fallback = False
    if not views and len(frames) > 1:
        raise NoSheet(f"I couldn't find the white A4 sheet in any of the {len(frames)} photos. "
                      "Put the sheet on a darker surface, keep all four corners in view, and scan again.")
    if not views and frames:  # single-photo fallback: assume top-down
        best = max(frames, key=lambda f: f["sharp"])
        best["fallback"] = True
        views, fallback = [best], True

    per_view, acc, wsum = [], np.zeros((V.CANVAS_H, V.CANVAS_W, 3)), np.zeros((V.CANVAS_H, V.CANVAS_W))
    thumbs = []
    for n, f in enumerate(views):
        if f.get("fallback"):
            warped, valid = V.topdown_fallback(f["img"])
            weight = 1.0
        else:
            warped, valid = V.warp(f["img"], f["pose"]["img_pts"])
            C = f["pose"]["cam"]
            weight = float(C[2] / np.linalg.norm(C)) ** 2  # steeper views are sharper top-down
        dets = V.detect(warped, valid)
        per_view.append(dets)
        wm = (valid > 0).astype(np.float64) * weight
        acc += warped * wm[..., None]
        wsum += wm
        # thumbnail with the sheet outline for the dashboard
        th = f["img"].copy()
        if f["corners"] is not None:
            cv2.polylines(th, [f["corners"].astype(np.int32)], True, (0, 255, 120), 3)
        s = 320 / max(th.shape[:2])
        cv2.imwrite(str(out / f"view_{n}.jpg"), cv2.resize(th, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 75])
        thumbs.append(f"/scans/{scan_id}/view_{n}.jpg")
        progress("view", {"n": n + 1, "of": len(views), "thumb": thumbs[-1], "found": sorted(dets)})

    # fuse
    nv = max(1, len(per_view))
    fused = {}
    for label in ("red", "blue", "yellow", "green", "tray"):
        obs = [d[label] for d in per_view if label in d]
        if not obs:
            continue
        xs, ys, ar = (np.array([o[k] for o in obs]) for k in ("x", "y", "area"))
        fused[label] = {"x": float(np.median(xs)), "y": float(np.median(ys)), "area": float(np.median(ar)),
                        "seen_in_n_views": len(obs), "confidence": round(len(obs) / nv, 2)}
    min_views = 1 if nv <= 2 else 2
    unsure = [k for k, v in fused.items() if v["seen_in_n_views"] < min_views]

    # texture
    tex = np.full((V.CANVAS_H, V.CANVAS_W, 3), (70, 95, 125), np.float64)  # table colour where unseen
    ok = wsum > 0
    tex[ok] = acc[ok] / wsum[ok][:, None]
    tex = tex.astype(np.uint8)
    tex_path = out / "table.png"
    cv2.imwrite(str(tex_path), tex)
    ann = V.annotate(tex, fused, {k: f"{v['seen_in_n_views']}/{nv}" for k, v in fused.items()})
    cv2.imwrite(str(out / "twin.jpg"), ann, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # layout for the sim
    lay = Layout()
    lay.texture = str(tex_path.resolve())
    for z in ("green", "tray"):
        if z in fused:
            x, y = _mm_to_sim(fused[z]["x"], fused[z]["y"])
            side = float(np.clip(np.sqrt(fused[z]["area"]) / 2 / 1000 * SCALE, 0.05, 0.09))
            lay.zones[z] = {"pos": (x, y), "half": (side, side)}
        else:
            lay.zones[z] = dict(DEFAULT_ZONES[z])
    placed = []
    for o in OBJECT_NAMES:
        if o in fused:
            x, y = _mm_to_sim(fused[o]["x"], fused[o]["y"])
            h = float(np.clip(np.sqrt(fused[o]["area"]) / 2 / 1000 * SCALE, 0.016, 0.03))
            p = np.array([x, y])
            for q in placed:  # never start the sim with interpenetrating cubes
                d = p - q
                if np.linalg.norm(d) < 2 * CUBE_HALF + 0.01:
                    p = q + (d / (np.linalg.norm(d) + 1e-9)) * (2 * CUBE_HALF + 0.012)
            p = np.clip(p, [-2 * WS_HALF[0] + 0.05, -2 * WS_HALF[1] + 0.05], [2 * WS_HALF[0] - 0.05, 2 * WS_HALF[1] - 0.05])
            placed.append(p)
            lay.objects[o] = {"pos": (float(p[0]), float(p[1])), "half": h, "present": True}
        else:
            lay.objects[o] = {"pos": (0.0, 0.0), "half": CUBE_HALF, "present": False}

    summary = {
        "id": scan_id, "frames": len(frames), "with_sheet": sum(1 for f in frames if f["cands"]),
        "views": len(views), "fallback": fallback, "objects": fused, "unsure": unsure,
        "thumbs": thumbs, "twin": f"/scans/{scan_id}/twin.jpg", "texture": f"/scans/{scan_id}/table.png",
        "seconds": round(time.time() - t0, 2),
    }
    summary["greeting"] = greeting(fused, unsure, fallback)
    (out / "scan.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return lay, summary


def greeting(fused: dict, unsure: list, fallback: bool) -> str:
    blocks = [o for o in OBJECT_NAMES if o in fused and o not in unsure]
    zones = [ZONE_LABEL[z] for z in ("green", "tray") if z in fused and z not in unsure]
    if not blocks and not zones:
        return "I couldn't find anything on the sheet. Can you scan again, a bit slower?"
    parts = []
    if blocks:
        parts.append("a " + (", a ".join(blocks[:-1]) + " and a " + blocks[-1] if len(blocks) > 1 else blocks[0])
                     + (" block" if len(blocks) == 1 else " block"))
    if zones:
        parts.append("the " + " and the ".join(zones))
    s = ("I've mapped your table" if not fallback else "I had to guess from one photo") + ". I see " + \
        (", " if len(blocks) > 1 else " and ").join(parts) + "."
    for u in unsure:
        s += f" I think there's a {ZONE_LABEL.get(u, u + ' block')} too, but I'm not sure."
    missing = [ZONE_LABEL[z] for z in ("green", "tray") if z not in fused]
    if missing:
        s += f" I couldn't see the {' or the '.join(missing)}, so I'll use the usual spot."
    return s + " What should I do?"
