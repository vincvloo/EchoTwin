"""F4 geometry: find the A4 sheet, recover the camera pose from it, warp to top-down, find colour blobs.

Canvas convention (used by the scan texture too): 1 px = 1 mm, sheet centred, canvas covers twice the
sheet in each direction (594 x 420 px). Sheet coordinates: x right, y back (away from the first camera
view), origin at the sheet centre.
"""
import json

import cv2
import numpy as np

from ..config import DATA

SHEET_W, SHEET_H = 297.0, 210.0
CANVAS_W, CANVAS_H = int(2 * SHEET_W), int(2 * SHEET_H)
SHEET_OBJ = np.array([[-SHEET_W / 2, -SHEET_H / 2, 0], [SHEET_W / 2, -SHEET_H / 2, 0],
                      [SHEET_W / 2, SHEET_H / 2, 0], [-SHEET_W / 2, SHEET_H / 2, 0]], dtype=np.float64)

# HSV ranges (OpenCV: H 0-180). Override on site with data/vision.json.
COLOURS = {
    "red": [((0, 110, 70), (9, 255, 255)), ((168, 110, 70), (180, 255, 255))],
    "yellow": [((18, 110, 110), (36, 255, 255))],
    "blue": [((95, 110, 50), (130, 255, 255))],
    "green": [((40, 70, 50), (88, 255, 255))],
}
BLOCK_MIN_MM2 = 60
ZONE_MIN_MM2 = 500
BLUE_TRAY_MIN_MM2 = 1400
SHEET_MARGIN_MM = 30
MAX_FIT_ERR = 0.012  # corner fit error / sheet size in the image


def load_config():
    p = DATA / "vision.json"
    if p.exists():
        try:
            cfg = json.loads(p.read_text())
            for k, v in cfg.get("colours", {}).items():
                COLOURS[k] = [tuple(map(tuple, r)) for r in v]
            globals().update({k: v for k, v in cfg.items() if k.isupper()})
        except Exception as e:
            print("[vision] bad data/vision.json:", e)


load_config()


def decode(jpeg: bytes, max_side: int = 1280) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def sharpness(bgr) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = 480 / g.shape[1]
    g = cv2.resize(g, None, fx=s, fy=s)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def camera_matrix(shape) -> np.ndarray:
    h, w = shape[:2]
    f = 0.78 * max(w, h)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def find_sheet(bgr) -> np.ndarray | None:
    """Corners (4x2, image px) of the largest bright, low-saturation quadrilateral."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    area_img = bgr.shape[0] * bgr.shape[1]
    best, best_area = None, 0
    for smax, vmin in ((60, 150), (80, 125), (45, 185), (100, 105)):
        m = ((hsv[..., 1] < smax) & (hsv[..., 2] > vmin)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
            a = cv2.contourArea(c)
            if a < 0.02 * area_img or a > 0.97 * area_img:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in (0.015, 0.025, 0.035, 0.05, 0.07):
                q = cv2.approxPolyDP(hull, eps * peri, True)
                if len(q) == 4:
                    break
            if len(q) != 4:
                continue
            # a 4-gon that fills most of its hull is a sheet; a blob is not
            if cv2.contourArea(q) < 0.8 * cv2.contourArea(hull):
                continue
            if a > best_area:
                best, best_area = q.reshape(4, 2).astype(np.float64), a
        if best is not None:
            break
    if best is None:
        return None
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    pts = best.reshape(-1, 1, 2).astype(np.float32)
    cv2.cornerSubPix(g, pts, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return pts.reshape(4, 2).astype(np.float64)


def pose_candidates(corners, shape) -> list[dict]:
    """All metric-consistent assignments of image corners to sheet corners (normally two, 180 deg apart)."""
    K = camera_matrix(shape)
    c = corners.mean(0)
    ang = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
    ordered = corners[np.argsort(ang)]
    out = []
    for seq in (ordered, ordered[::-1]):
        for k in range(4):
            ip = np.roll(seq, k, axis=0)
            ok, rvec, tvec = cv2.solvePnP(SHEET_OBJ, ip, K, None, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(SHEET_OBJ, rvec, tvec, K, None)
            err = float(np.linalg.norm(proj.reshape(4, 2) - ip, axis=1).mean())
            R, _ = cv2.Rodrigues(rvec)
            C = (-R.T @ tvec).ravel()
            if C[2] <= 0:
                continue
            out.append({"img_pts": ip, "rvec": rvec, "tvec": tvec, "err": err, "cam": C, "K": K})
    if not out:
        return []
    e0 = min(o["err"] for o in out)
    # a real A4 sheet fits a 297x210 rectangle almost exactly; windows, screens and rugs do not
    if e0 > MAX_FIT_ERR * np.sqrt(cv2.contourArea(corners.astype(np.float32))):
        return []
    return sorted([o for o in out if o["err"] <= 1.6 * e0 + 1.5], key=lambda o: o["err"])


def choose_pose(cands: list[dict], prev_cam: np.ndarray | None) -> dict | None:
    if not cands:
        return None
    if prev_cam is None:  # first view defines "front": the camera is in front of the sheet (y < 0)
        return min(cands, key=lambda o: (o["cam"][1] >= 0, o["err"]))
    a0 = np.arctan2(prev_cam[1], prev_cam[0])
    return min(cands, key=lambda o: abs(np.angle(np.exp(1j * (np.arctan2(o["cam"][1], o["cam"][0]) - a0)))))


def canvas_pts(obj_xy) -> np.ndarray:
    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    return np.c_[obj_xy[:, 0] + SHEET_W, SHEET_H - obj_xy[:, 1]]


def canvas_to_mm(u, v):
    return float(u - SHEET_W), float(SHEET_H - v)


def warp(bgr, img_pts) -> tuple[np.ndarray, np.ndarray]:
    """Top-down canvas of the table and a validity mask (inside the photo, in front of the camera)."""
    H = cv2.getPerspectiveTransform(img_pts.astype(np.float32), canvas_pts(SHEET_OBJ[:, :2]).astype(np.float32))
    out = cv2.warpPerspective(bgr, H, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full(bgr.shape[:2], 255, np.uint8), H, (CANVAS_W, CANVAS_H),
                                flags=cv2.INTER_NEAREST)
    Hi = np.linalg.inv(H)
    uu, vv = np.meshgrid(np.arange(CANVAS_W), np.arange(CANVAS_H))
    w = Hi[2, 0] * uu + Hi[2, 1] * vv + Hi[2, 2]
    wc = Hi[2] @ np.array([SHEET_W, SHEET_H, 1.0])
    valid[(w * wc) <= 0] = 0
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
    return out, valid


def topdown_fallback(bgr) -> tuple[np.ndarray, np.ndarray]:
    """No sheet found: assume a roughly top-down photo whose frame is the sheet."""
    if bgr.shape[0] > bgr.shape[1]:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    h, w = bgr.shape[:2]
    src = np.array([[0, h], [w, h], [w, 0], [0, 0]], dtype=np.float64)
    return warp(bgr, src)


def detect(warped, valid) -> dict:
    """Blobs on the sheet area -> {label: {x, y, area}} in sheet mm. Labels: red, blue, yellow, green, tray."""
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    region = np.zeros(valid.shape, np.uint8)
    m = SHEET_MARGIN_MM
    region[int(SHEET_H / 2 - m):int(SHEET_H * 1.5 + m), int(SHEET_W / 2 - m):int(SHEET_W * 1.5 + m)] = 255
    region &= valid
    blobs = {}
    for name, ranges in COLOURS.items():
        mask = np.zeros(valid.shape, np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask &= region
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        found = []
        for i in range(1, n):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a >= BLOCK_MIN_MM2:
                x, y = canvas_to_mm(*cents[i])
                found.append({"x": x, "y": y, "area": a,
                              "bbox": [int(v) for v in stats[i, :4]]})
        blobs[name] = sorted(found, key=lambda b: -b["area"])
    out = {}
    for c in ("red", "yellow"):
        if blobs[c]:
            out[c] = blobs[c][0]
    if blobs["green"] and blobs["green"][0]["area"] >= ZONE_MIN_MM2:
        out["green"] = blobs["green"][0]
    blue = blobs["blue"]
    trays = [b for b in blue if b["area"] >= BLUE_TRAY_MIN_MM2]
    blocks = [b for b in blue if b["area"] < BLUE_TRAY_MIN_MM2]
    if trays:
        out["tray"] = trays[0]
    if blocks:
        out["blue"] = blocks[0]
    elif len(trays) > 1:  # a big blue block (or parallax-smeared): second-largest is the block
        out["blue"] = trays[1]
    return out


def annotate(canvas, dets: dict, confidences: dict | None = None) -> np.ndarray:
    img = canvas.copy()
    tl = (int(SHEET_W / 2), int(SHEET_H / 2))
    cv2.rectangle(img, tl, (int(SHEET_W * 1.5), int(SHEET_H * 1.5)), (255, 255, 255), 1)
    colours = {"red": (40, 40, 230), "yellow": (30, 210, 240), "blue": (230, 90, 30), "green": (60, 200, 60),
               "tray": (230, 120, 30)}
    for name, b in dets.items():
        u, v = int(b["x"] + SHEET_W), int(SHEET_H - b["y"])
        r = int(max(8, np.sqrt(b["area"]) / 1.6))
        col = colours.get(name, (255, 255, 255))
        cv2.circle(img, (u, v), r, (0, 0, 0), 4)
        cv2.circle(img, (u, v), r, col, 2)
        label = name if name != "tray" else "blue tray"
        if confidences and name in confidences:
            label += f" {confidences[name]}"
        cv2.putText(img, label, (u + r + 3, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (u + r + 3, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img
