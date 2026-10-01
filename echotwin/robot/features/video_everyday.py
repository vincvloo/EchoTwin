"""Learn from a video of your hand moving an everyday object (hand-held phone, plain table, no sheet).

Only the start and the end of the video matter:
  1. pick a clean frame near the start and near the end (most objects visible, sharp, no hand)
  2. detect the objects in both (everyday segmentation) and match them by colour and size
  3. the camera moved between the two: line up the objects that stayed put (rotation + translation);
     the one that no longer lines up is the object you moved
  4. where did it go? next to another object, or a move in some direction
The result is expressed as a relation ("next to the glass"), so it transfers to the twin even though the
video's camera and the twin's camera differ.
"""
import itertools

import cv2
import numpy as np

from . import everyday as E
from . import vision as V

PITCH = 50.0
NEAR_GAP = 0.06      # metres between object edges that still counts as "next to"
MIN_MOVE = 0.03


def _items(bgr, pitch):
    res = E.analyse(bgr, pitch)
    for it in res["items"]:
        x, y, w, h = it["box"]
        lab = cv2.cvtColor(res["bgr"], cv2.COLOR_BGR2LAB)
        it["lab"] = np.array(cv2.mean(lab, mask=it["mask"].astype(np.uint8))[:3])
        it["r"] = float(max(it["size"][0], it["size"][1]) / 2)
    return res


def _clean_frame(frames, pitch):
    """Best frame in a list: most objects, then sharpest."""
    best = None
    for f in frames:
        res = _items(f, pitch)
        res["cam"] = E.Camera(res["bgr"].shape, pitch)
        score = (len(res["items"]), V.sharpness(f))
        if best is None or score > best[0]:
            best = (score, f, res)
    return best[1], best[2]


def _norm_r(items):
    med = float(np.median([i["r"] for i in items])) or 1.0
    return [i["r"] / med for i in items]


def _appearance_ok(a, b, ra, rb) -> bool:
    """Same kind of thing? Colour close, relative size within a factor 1.8 (camera distance changes)."""
    colour = np.linalg.norm(a["lab"] - b["lab"])
    size = max(ra, rb) / max(1e-3, min(ra, rb))
    return colour < 38 and size < 1.8


def fit_similarity(P, Q):
    """2D scale + rotation + translation mapping P onto Q (Umeyama)."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    mp, mq = P.mean(0), Q.mean(0)
    Pc, Qc = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(Qc.T @ Pc / len(P))
    D = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    var = (Pc ** 2).sum() / len(P)
    sc = float(np.trace(np.diag(S) @ D) / var) if var > 1e-9 else 1.0
    return sc, R, mq - sc * R @ mp


def analyse(frames: list[np.ndarray], pitch: float = PITCH) -> dict:
    n = len(frames)
    if n < 4:
        return {"error": "That video is too short. Show me the whole move."}
    k = max(3, n // 6)
    f0, start = _clean_frame(frames[:k], pitch)
    f1, end = _clean_frame(frames[-k:], pitch)
    A, B = start["items"], end["items"]
    if len(A) < 2 or len(B) < 2:
        return {"error": "I need to see at least two things on the table, at the start and at the end."}
    ra, rb = _norm_r(A), _norm_r(B)
    ok = {(i, j) for i in range(len(A)) for j in range(len(B)) if _appearance_ok(A[i], B[j], ra[i], rb[j])}
    PA = [np.array(a["xy"]) for a in A]
    QB = [np.array(b["xy"]) for b in B]
    # RANSAC over pairs of correspondences: the camera motion that lines up the most objects
    best = None
    cands = sorted(ok)
    for (i1, j1), (i2, j2) in itertools.combinations(cands, 2):
        if i1 == i2 or j1 == j2:
            continue
        sc, R, t = fit_similarity([PA[i1], PA[i2]], [QB[j1], QB[j2]])
        if not 0.4 < sc < 2.5:
            continue
        tol = 0.035 * sc
        inl = []
        used = set()
        for i in range(len(A)):
            pi = sc * R @ PA[i] + t
            js = [(np.linalg.norm(pi - QB[j]), j) for j in range(len(B)) if (i, j) in ok and j not in used]
            if js:
                d, j = min(js)
                if d < tol:
                    inl.append((i, j))
                    used.add(j)
        err = sum(np.linalg.norm(sc * R @ PA[i] + t - QB[j]) for i, j in inl)
        # on a tie prefer the least camera change: any 2 points can be lined up by some zoom + rotation
        rot = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        key = (len(inl), -(err + 0.05 * abs(np.log(sc)) + 0.002 * rot))
        if best is None or key > best[0]:
            best = (key, inl)
    if best is None or len(best[1]) < 2:
        return {"error": "I couldn't line up the start and the end of the video. Keep the same things in view."}
    inl = best[1]
    sc, R, t = fit_similarity([PA[i] for i, _ in inl], [QB[j] for _, j in inl])
    resid = float(np.mean([np.linalg.norm(sc * R @ PA[i] + t - QB[j]) for i, j in inl]))
    # the moved object: a start object with a look-alike at the end that is NOT where the camera motion predicts
    still_i = {i for i, _ in inl}
    still_j = {j for _, j in inl}
    moved = None
    for i in range(len(A)):
        if i in still_i:
            continue
        pi = sc * R @ PA[i] + t
        for j in range(len(B)):
            if j in still_j or (i, j) not in ok:
                continue
            disp = float(np.linalg.norm(pi - QB[j]) / sc)  # in start-frame metres
            if disp > MIN_MOVE and (moved is None or disp < moved[2]):
                moved = (i, j, disp)
    if moved is None:
        return {"error": "I didn't see anything move. Try again, and move one thing clearly.",
                "lined_up": len(inl)}
    m, jm, disp = moved
    qm = QB[jm]
    near = None
    for i, j in inl:
        gap = float((np.linalg.norm(QB[j] - qm) / sc) - A[m]["r"] - A[i]["r"])
        if gap < NEAR_GAP and (near is None or gap < near[1]):
            near = (i, gap)
    move = R.T @ (qm - (sc * R @ PA[m] + t)) / sc  # displacement in the start frame's axes
    ax = 0 if abs(move[0]) >= abs(move[1]) else 1
    direction = (int(np.sign(move[0])), 0) if ax == 0 else (0, int(np.sign(move[1])))
    return {
        "moved": m, "ref": near[0] if near else None, "gap": near[1] if near else None,
        "direction": direction, "distance": disp, "residual": resid, "scale": sc, "lined_up": len(inl),
        "start_items": A, "end_items": B, "start_frame": f0, "end_frame": f1, "pairs": inl + [(m, jm)],
    }


def map_to_twin(res: dict, props: list[dict]) -> dict | None:
    """Video objects (start frame) -> twin props, by colour and relative size."""
    A = res["start_items"]
    if not props:
        return None
    ra = np.mean([a["r"] for a in A])
    rp = np.mean([max(p["size"][0], p["size"][1]) for p in props])
    cost = {}
    for i, a in enumerate(A):
        for j, p in enumerate(props):
            colour = np.linalg.norm(np.array(a["rgb"]) - np.array(p["rgb"])) * 2
            size = abs(a["r"] / ra - max(p["size"][0], p["size"][1]) / rp)
            cost[(i, j)] = colour + size
    mapping, used = {}, set()
    for (i, j), c in sorted(cost.items(), key=lambda kv: kv[1]):
        if i in mapping or j in used:
            continue
        mapping[i] = j
        used.add(j)
    if res["moved"] not in mapping:
        return None
    mover = mapping[res["moved"]]
    if res["ref"] is not None and res["ref"] in mapping:
        return {"prop": mover, "goal": ("near", mapping[res["ref"]], "next to")}
    dist = 0.08 if res["distance"] < 0.06 else 0.2
    return {"prop": mover, "goal": ("dir", res["direction"], dist)}


def frames_from_video_ends(data: bytes, per_end: int = 8, part: float = 0.2) -> list[np.ndarray]:
    """Decode only the first and last 20% of a video (a 4K phone video is slow to decode in full)."""
    import tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(data)
        path = f.name
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = list(np.linspace(0, n * part, per_end).astype(int)) + list(np.linspace(n * (1 - part), n - 2, per_end).astype(int))
    out = []
    for k in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, img = cap.read()
        if ok:
            s = 960 / max(img.shape[:2])
            out.append(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img)
    cap.release()
    Path(path).unlink(missing_ok=True)
    return out


def frames_everyday_ok(frames: list[np.ndarray]) -> bool:
    """True when no A4 sheet is visible (so the block tracker does not apply)."""
    for f in frames[:: max(1, len(frames) // 6)][:6]:
        c = V.find_sheet(f)
        if c is not None and V.pose_candidates(c, f.shape):
            return False
    return True
