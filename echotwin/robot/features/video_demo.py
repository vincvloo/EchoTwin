"""Learn from a video of a human hand doing the task.

The A4 sheet gives a top-down view of every frame. We track the three coloured blocks, find the one that
moved, which zone it ended in, and its path. That becomes a waypoint plan the robot replays in the twin;
the replay is recorded as a normal demo (source "video"). Heights and the grasp itself are not visible in
a single video, so the approach, grasp and carry height are filled in by the sim.
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ..scene import OBJECT_NAMES, SCALE
from . import vision as V

MOVED_MIN_MM = 40
MAX_JUMP_MM = 90


def frames_from_video(data: bytes, fps: float = 6.0, max_frames: int = 150) -> tuple[list[np.ndarray], float]:
    """Decode an uploaded video file and subsample it."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(data)
        path = f.name
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    out, i = [], 0
    while len(out) < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            s = 1280 / max(img.shape[:2])
            out.append(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img)
        i += 1
    cap.release()
    Path(path).unlink(missing_ok=True)
    return out, src_fps / step


def _zone_match_cost(dets: dict, zones_mm: dict) -> float:
    cost, n = 0.0, 0
    for z, (x, y) in zones_mm.items():
        if z in dets:
            cost += np.hypot(dets[z]["x"] - x, dets[z]["y"] - y)
            n += 1
    return cost / n if n else 1e9


def analyse(frames: list[np.ndarray], fps: float, zones_mm: dict | None = None) -> dict:
    """-> {object, target, start, end, path[(t, x_mm, y_mm)], duration, frames_used} or {error}."""
    tracks = {c: [] for c in OBJECT_NAMES}
    zone_obs = {"green": [], "tray": []}
    prev_cam, used = None, 0
    for i, img in enumerate(frames):
        corners = V.find_sheet(img)
        if corners is None:
            continue
        cands = V.pose_candidates(corners, img.shape)
        if not cands:
            continue
        if prev_cam is None and zones_mm:
            # first frame: pick the orientation that puts the zones where the scan saw them
            best = None
            for c in cands:
                w, valid = V.warp(img, c["img_pts"])
                cost = _zone_match_cost(V.detect(w, valid), zones_mm)
                if best is None or cost < best[0]:
                    best = (cost, c, w, valid)
            pose, warped, valid = best[1], best[2], best[3]
        else:
            pose = V.choose_pose(cands, prev_cam)
            warped, valid = V.warp(img, pose["img_pts"])
        prev_cam = pose["cam"]
        dets = V.detect(warped, valid)
        used += 1
        t = i / fps
        for c in OBJECT_NAMES:
            if c in dets:
                tracks[c].append((t, dets[c]["x"], dets[c]["y"]))
        for z in zone_obs:
            if z in dets:
                zone_obs[z].append((dets[z]["x"], dets[z]["y"]))
    if used < 3:
        return {"error": "I couldn't see the sheet in the video. Keep the whole sheet in view."}

    zones = {z: tuple(np.median(np.array(v), axis=0)) for z, v in zone_obs.items() if v}
    if zones_mm:
        zones = {**zones_mm, **{k: v for k, v in zones.items() if k not in zones_mm}}

    # which block moved?
    best = None
    for c, tr in tracks.items():
        if len(tr) < 4:
            continue
        a = np.array(tr)
        start, end = np.median(a[:3, 1:], axis=0), np.median(a[-3:, 1:], axis=0)
        d = float(np.linalg.norm(end - start))
        if best is None or d > best[0]:
            best = (d, c, start, end, a)
    if best is None or best[0] < MOVED_MIN_MM:
        return {"error": "I didn't see any block move. Try again, a bit slower."}
    _, obj, start, end, a = best
    if not zones:
        return {"error": "I couldn't see the zones. Scan the table first."}

    # a blue block dropped on the blue tray merges with it: it vanishes near a zone for the rest of the video
    video_end = (len(frames) - 1) / fps
    last = a[-1, 1:]
    near = min(zones, key=lambda z: np.hypot(last[0] - zones[z][0], last[1] - zones[z][1]))
    if video_end - a[-1, 0] > 1.0 and np.hypot(last[0] - zones[near][0], last[1] - zones[near][1]) < 130:
        end = np.array(zones[near], dtype=float)
        a = np.vstack([a, [a[-1, 0] + 0.4, *end]])

    # clean the path: drop jumps (hand occlusion, skin mistaken for colour)
    path = [(a[0, 0], *start)]
    for t, x, y in a[1:]:
        if np.hypot(x - path[-1][1], y - path[-1][2]) < MAX_JUMP_MM:
            path.append((t, x, y))
    path.append((a[-1, 0], *end))
    p = np.array(path)
    dist_start = np.hypot(p[:, 1] - start[0], p[:, 2] - start[1])
    dist_end = np.hypot(p[:, 1] - end[0], p[:, 2] - end[1])
    i0 = int(np.argmax(dist_start > 15)) if (dist_start > 15).any() else 0
    i1 = len(p) - 1 - int(np.argmax(dist_end[::-1] > 15)) if (dist_end > 15).any() else len(p) - 1
    carried = p[max(i0 - 1, 0):min(i1 + 2, len(p))]

    target = min(zones, key=lambda z: np.hypot(end[0] - zones[z][0], end[1] - zones[z][1]))
    zdist = float(np.hypot(end[0] - zones[target][0], end[1] - zones[target][1]))
    return {
        "object": obj, "target": target, "in_zone": zdist < 60,
        "start": [float(v) for v in start], "end": [float(v) for v in end],
        "path": carried.tolist(), "duration": float(max(0.5, carried[-1, 0] - carried[0, 0])),
        "frames_used": used, "frames": len(frames),
    }


def to_sim(res: dict) -> dict:
    """Convert the analysis to sim coordinates (metres)."""
    f = lambda x, y: (x / 1000 * SCALE, y / 1000 * SCALE)
    pts = [f(x, y) for _, x, y in res["path"]]
    length = float(sum(np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)))
    return {"object": res["object"], "target": res["target"], "start": f(*res["start"]), "end": f(*res["end"]),
            "path": pts, "speed": length / res["duration"]}
