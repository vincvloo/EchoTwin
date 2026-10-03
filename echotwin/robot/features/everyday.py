"""Everyday objects: map a real table with ordinary things from ONE photo.

Geometry comes from classic vision (reliable, offline); names come from the AI (optional):
  1. the table surface = the region matching the colour at the bottom centre of the photo
  2. objects = things inside the table's outline that are not table-coloured
  3. camera model from the phone's tilt at capture time (or 45 deg): pixels -> table plane,
     giving each object's position, footprint and height
  4. the AI names the numbered boxes ("set-of-marks" prompting); without AI: "object 1", ...
Scale is approximate (camera height assumed), which is fine for a twin you look at.
"""
import cv2
import numpy as np

from ..scene import TABLE_ASPECT

CAM_HEIGHT = 0.45        # metres above the table, assumed
MAX_OBJECTS = 8
TEX_PX = (1188, 840)     # texture size, same aspect as the sim table


def _prep(bgr, max_side=960):
    s = max_side / max(bgr.shape[:2])
    return cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr.copy()


def segment(bgr) -> tuple[np.ndarray, list[dict]]:
    """-> (table mask, objects [{box, mask, area}]) in the given image's pixels."""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    # the table's colour: try several sample patches (a cable or a seam can sit under any single one)
    # and keep the one that yields the biggest connected table surface
    table, best = None, -1
    for fy, fx in ((0.88, 0.5), (0.5, 0.5), (0.7, 0.25), (0.7, 0.75), (0.88, 0.2), (0.88, 0.8), (0.3, 0.5)):
        y0, x0 = int(h * fy), int(w * fx)
        patch = lab[max(0, y0 - h // 20):y0 + h // 20, max(0, x0 - w // 10):x0 + w // 10].reshape(-1, 3)
        ref = np.median(patch, axis=0)
        d = np.sqrt(((lab[..., 0] - ref[0]) * 0.45) ** 2 + (lab[..., 1] - ref[1]) ** 2 + (lab[..., 2] - ref[2]) ** 2)
        cand = cv2.morphologyEx((d < 13).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand)
        if lbl[y0, x0] == 0:
            continue
        area = int(stats[lbl[y0, x0], cv2.CC_STAT_AREA])
        if area > best:
            table, best = (lbl == lbl[y0, x0]).astype(np.uint8), area
    if table is None:
        table = np.ones((h, w), np.uint8)
    cnts, _ = cv2.findContours(table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(table)
    if cnts:
        cv2.fillPoly(hull, [cv2.convexHull(max(cnts, key=cv2.contourArea))], 1)
    # textured things (wrappers with print, glass edges) count too, not just off-colour ones
    edges = cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    obj = ((hull > 0) & ((table == 0) | (edges > 0))).astype(np.uint8)
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    obj = cv2.dilate(obj, np.ones((9, 9), np.uint8))  # glue fragments of one object (glass rims)
    obj = cv2.erode(obj, np.ones((5, 5), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(obj)
    # far edge of the table, per column: things touching it are behind the table, not on it
    has = hull.any(axis=0)
    top = np.where(has, hull.argmax(axis=0), h)
    parts = []
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        if a < 0.0006 * w * h:
            continue
        if y <= 1 or x <= 1 or x + bw >= w - 1:  # cut by the photo edge: probably not on the table
            continue
        if y <= top[x:x + bw].min() + 6:
            continue
        parts.append([int(x), int(y), int(x + bw), int(y + bh), lbl == i])
    # merge fragments of one object (a glass shows up as rim + water line + base)
    gap = int(0.006 * max(w, h))  # small: objects placed side by side must stay separate
    merged = True
    while merged:
        merged = False
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                a, b = parts[i], parts[j]
                near = a[0] - gap < b[2] and b[0] - gap < a[2] and a[1] - gap < b[3] and b[1] - gap < a[3]
                # stacked in one column (rim above water line above base): merge across a bigger vertical gap
                x_ov = min(a[2], b[2]) - max(a[0], b[0])
                stacked = x_ov > 0.5 * min(a[2] - a[0], b[2] - b[0]) and \
                    max(a[1], b[1]) - min(a[3], b[3]) < 0.07 * h
                if near or stacked:
                    parts[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), a[4] | b[4]]
                    parts.pop(j)
                    merged = True
                    break
            if merged:
                break
    out = []
    for x0, y0, x1, y1, m in parts:
        a = int(m.sum())
        if a < 0.0012 * w * h or a > 0.2 * w * h:
            continue
        out.append({"box": [x0, y0, x1 - x0, y1 - y0], "mask": m, "area": a})
    out.sort(key=lambda o: -o["area"])
    return table, out[:MAX_OBJECTS]


class Camera:
    """Pinhole camera at height H, pitched down by `pitch` degrees, looking along +y."""

    def __init__(self, shape, pitch_deg=45.0, height=CAM_HEIGHT):
        self.h_img, self.w_img = shape[:2]
        self.f = 0.78 * max(self.h_img, self.w_img)
        self.cx, self.cy = self.w_img / 2, self.h_img / 2
        th = np.radians(np.clip(pitch_deg, 15, 89))
        self.C = np.array([0.0, 0.0, height])
        self.F = np.array([0.0, np.cos(th), -np.sin(th)])
        self.R = np.array([1.0, 0.0, 0.0])
        self.D = np.cross(self.F, self.R)

    def ray(self, u, v):
        return self.R * (u - self.cx) / self.f + self.D * (v - self.cy) / self.f + self.F

    def ground(self, u, v):
        r = self.ray(u, v)
        if r[2] >= -1e-6:
            return None
        return self.C + r * (-self.C[2] / r[2])

    def project(self, P):
        p = np.asarray(P, float) - self.C
        z = p @ self.F
        return np.array([self.cx + self.f * (p @ self.R) / z, self.cy + self.f * (p @ self.D) / z])

    def height_at(self, u, v, y_world):
        """Height of the point seen at pixel (u, v) if it stands at depth y_world."""
        r = self.ray(u, v)
        if abs(r[1]) < 1e-6:
            return 0.0
        t = (y_world - self.C[1]) / r[1]
        return float(self.C[2] + t * r[2])


def marks_image(bgr, objs) -> np.ndarray:
    img = bgr.copy()
    for i, o in enumerate(objs, 1):
        x, y, w, h = o["box"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 5)
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 220, 255), 2)
        cv2.putText(img, str(i), (x + 4, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, str(i), (x + 4, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 220, 255), 2, cv2.LINE_AA)
    return img


SHAPES = ("box", "cylinder", "flat", "round")

NAMING_PROMPT = """The photo shows a table with numbered yellow boxes around things on it.
For each number, name the object in 1-4 plain words and say its 3D shape:
box (boxy), cylinder (glass, bottle, cup), flat (paper, wrapper, card, phone lying down), round (egg-shaped, ball, case).
Reply with JSON only: {"objects": [{"id": 1, "name": "black earbud case", "shape": "round"}]}
Use "skip": true for a box that is not a real object (shadow, reflection, part of the table)."""


def analyse(jpeg_or_bgr, pitch_deg: float | None = None) -> dict:
    """Stage 1 (no AI): segment and measure. Returns everything the layout needs."""
    bgr = jpeg_or_bgr if isinstance(jpeg_or_bgr, np.ndarray) else \
        cv2.imdecode(np.frombuffer(jpeg_or_bgr, np.uint8), cv2.IMREAD_COLOR)
    bgr = _prep(bgr)
    pitch = 45.0 if pitch_deg is None else float(pitch_deg)
    cam = Camera(bgr.shape, pitch)
    table, objs = segment(bgr)
    items = []
    for o in objs:
        x, y, w, h = o["box"]
        ys, xs = np.nonzero(o["mask"])
        vb = float(np.percentile(ys, 97))  # contact line with the table
        pl, pr = cam.ground(x, vb), cam.ground(x + w, vb)
        pc = cam.ground(x + w / 2, vb)
        if pl is None or pr is None or pc is None:
            continue
        width = float(np.linalg.norm(pr - pl))
        depth = width
        centre = pc + np.array([0, depth / 2, 0])
        height = max(0.005, cam.height_at(x + w / 2, float(np.percentile(ys, 2)), centre[1]))
        colour = cv2.mean(bgr, mask=o["mask"].astype(np.uint8))[:3][::-1]
        items.append({"box": o["box"], "xy": centre[:2].tolist(), "size": [width, depth, min(height, 3 * width + 0.05)],
                      "rgb": [c / 255 for c in colour], "name": None, "shape": "box", "mask": o["mask"]})
    return {"bgr": bgr, "table": table, "items": items, "cam": cam, "pitch": pitch,
            "marks": marks_image(bgr, [{"box": i["box"]} for i in items])}


def apply_names(res: dict, ai: dict | None):
    names = {}
    if ai and isinstance(ai.get("objects"), list):
        for o in ai["objects"]:
            try:
                names[int(o["id"])] = o
            except (KeyError, ValueError, TypeError):
                pass
    keep = []
    for i, it in enumerate(res["items"], 1):
        n = names.get(i, {})
        if n.get("skip"):
            continue
        it["name"] = str(n.get("name") or f"object {i}")[:40]
        if n.get("shape") in SHAPES:
            it["shape"] = n["shape"]
        keep.append(it)
    res["items"] = keep
    return res


def build(res: dict, out_dir) -> tuple[list[dict], str, np.ndarray]:
    """Real-size objects on a table that fits them -> (props for the Layout, texture path, annotated image).
    Sets res["table_half"] (metres): the mapped area, with the shape of the texture."""
    items, cam, bgr = res["items"], res["cam"], res["bgr"]
    pts = np.array([it["xy"] for it in items]) if items else np.array([[0.0, 0.35]])
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    span_x = max(0.5, (pts[:, 0].max() - pts[:, 0].min()) * 1.6 + 0.15)
    span_y = max(0.35, (pts[:, 1].max() - pts[:, 1].min()) * 1.6 + 0.15)
    real_w = max(span_x, span_y * TABLE_ASPECT)
    real_h = real_w / TABLE_ASPECT
    tw, th = real_w, real_h
    s = 1.0  # sim metres are real metres
    res["table_half"] = (float(real_w / 2), float(real_h / 2))
    # texture: warp the photo's table plane onto the sim table
    corners = [(cx - real_w / 2, cy - real_h / 2), (cx + real_w / 2, cy - real_h / 2),
               (cx + real_w / 2, cy + real_h / 2), (cx - real_w / 2, cy + real_h / 2)]
    img_pts = np.array([cam.project([x, y, 0]) for x, y in corners], np.float32)
    dst = np.array([[0, TEX_PX[1]], [TEX_PX[0], TEX_PX[1]], [TEX_PX[0], 0], [0, 0]], np.float32)
    H = cv2.getPerspectiveTransform(img_pts, dst)
    tex = cv2.warpPerspective(bgr, H, TEX_PX, borderMode=cv2.BORDER_REPLICATE)
    tex_path = out_dir / "table.png"
    cv2.imwrite(str(tex_path), tex)
    px = tw / TEX_PX[0]  # sim metres per texture pixel
    props = []
    for i, it in enumerate(items):
        x, y = (it["xy"][0] - cx) * s, (it["xy"][1] - cy) * s
        w, d, h = (v * s for v in it["size"])
        yaw, skin = 0.0, None
        if it["shape"] == "flat" and it.get("mask") is not None:
            # flat things: exact outline and rotation from the top-down view, skin = their top-down look
            m = cv2.warpPerspective(it["mask"].astype(np.uint8) * 255, H, TEX_PX, flags=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                (u, v), (rw, rh), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
                x, y = (u / TEX_PX[0] - 0.5) * tw, (0.5 - v / TEX_PX[1]) * th
                w, d, yaw = rw * px, rh * px, -ang
                M = cv2.getRotationMatrix2D((u, v), ang, 1.0)
                rot = cv2.warpAffine(tex, M, TEX_PX, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                skin = cv2.getRectSubPix(rot, (max(4, int(rw)), max(4, int(rh))), (u, v))
            h = min(h, 0.012)
        elif it.get("mask") is not None:
            # standing things: skin = their photo, background filled with their own colour
            bx, by, bw, bh = it["box"]
            crop = bgr[by:by + bh, bx:bx + bw].copy()
            mk = it["mask"][by:by + bh, bx:bx + bw]
            crop[~mk] = np.array(it["rgb"][::-1]) * 255
            skin = crop
        if skin is not None and skin.size:
            path = out_dir / f"prop_{i}.png"
            cv2.imwrite(str(path), cv2.resize(skin, (256, 256)))  # each face stretches it to fit
            skin = str(path.resolve())
        props.append({"name": it["name"], "pos": (float(x), float(y)), "yaw": float(yaw),
                      "size": (float(max(w, 0.02) / 2), float(max(d, 0.02) / 2), float(max(h, 0.006) / 2)),
                      "rgb": it["rgb"], "shape": it["shape"], "skin": skin})
    ann = res["marks"].copy()
    for i, it in enumerate(items, 1):
        x, y, w, h = it["box"]
        cv2.putText(ann, it["name"], (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(ann, it["name"], (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    # the phone's own viewpoint in sim coordinates, so the twin can be seen exactly like the photo
    fovy = float(np.degrees(2 * np.arctan(cam.h_img / 2 / cam.f)))
    up = -cam.D
    res["view"] = {"pos": [float(-cx * s), float(-cy * s), float(cam.C[2] * s)],
                   "xyaxes": [*map(float, cam.R), *map(float, up)], "fovy": min(fovy, 75.0)}
    return props, str(tex_path.resolve()), ann


NUMBERS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _plural(n: str) -> str:
    if n.endswith(("s", "x", "ch", "sh")):
        return n + "es"
    return n[:-1] + "ies" if n.endswith("y") and n[-2:-1] not in "aeiou" else n + "s"


def listing(names: list[str]) -> str:
    """['glass', 'box', 'box'] -> 'a glass and two boxes'."""
    counts = {}
    for n in names:
        counts[n.lower()] = counts.get(n.lower(), 0) + 1
    parts = []
    for n, c in counts.items():
        if c > 1:
            parts.append(f"{NUMBERS.get(c, c)} {_plural(n)}")
        else:
            own = n.split(" ", 1)[0] in ("my", "your", "the", "our", "his", "her", "their")
            parts.append(n if own else ("an " if n[:1] in "aeiou" else "a ") + n)
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def greeting(items: list[dict]) -> str:
    if not items:
        return "I can see your table, but nothing on it. Put a few things down and snap again."
    listed = listing([it["name"] for it in items])
    return f"I've mapped your table. I see {listed}. That's an approximate map. What should I move?"
