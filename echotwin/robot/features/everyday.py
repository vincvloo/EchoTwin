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

from ...perception import marker as MK
from . import lathe
from ..scene import ATLAS_SUFFIX, DEFAULT_SURFACE, TABLE_ASPECT, Layout

CAM_HEIGHT = 0.45        # metres above the table, assumed
MAX_OBJECTS = 8
TEX_PX = (1188, 840)     # texture size, same aspect as the sim table
TILE = 192               # one face of an object's skin, in pixels
ROUND_FILL = 0.58        # a rounded thing fills at least this share of its box ...
ROUND_CORNERS = 0.33     # ... and leaves its corners mostly empty (see guess_shape)
THIN = 0.3               # a mask filling less than this share of its box is a thin, long thing (a cable): it lies flat


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
ROUND = ("round", "cylinder")     # shapes that are built from the object's outline (lathe.py), whatever the outline
TAPER = 0.2                       # an outline whose width changes this much up its height is not a box, whatever it is called

NAMING_PROMPT = """The photo shows things lying on a surface (a table, the floor, a shelf, a bed...), with numbered yellow
boxes around them. For each number, name the object in 1-4 plain words and say its 3D shape:
box (boxy), cylinder (glass, bottle, cup), flat (paper, wrapper, card, phone lying down), round (egg-shaped, ball, case).
Also say what they stand on: "table" (a table or desk), "floor", or "other" (a shelf, a bed, a chair...).
Reply with JSON only: {"surface": "table", "objects": [{"id": 1, "name": "black earbud case", "shape": "round"}]}
Use "skip": true for a box that is not a real object (shadow, reflection, part of the table or floor)."""

# How high the phone is assumed to be above what it photographs, when nothing measured it (no marker): the whole
# one-photo estimate scales with it. Above a table someone leans over it; the floor is photographed standing.
PHONE_ABOVE = {"table": CAM_HEIGHT, "floor": 1.2}
STANDING_PHONE = 1.25    # m above the floor; above a surface of height h the phone is about this minus h


def phone_height(surface: dict | None) -> float:
    s = surface or DEFAULT_SURFACE
    if s.get("kind") in PHONE_ABOVE:
        return PHONE_ABOVE[s["kind"]]
    return float(np.clip(STANDING_PHONE - float(s.get("height", 0.45)), 0.35, PHONE_ABOVE["floor"]))


def on_surface(layout: Layout, surface: dict) -> Layout:
    """The same photo, now known to show things on `surface`: everything estimated from the phone height scales with it.
    The table rescale the user made (meta["scale"]) is kept as it was."""
    k = phone_height(surface) / phone_height(layout.surface)
    lay = layout.scaled(k) if abs(k - 1) > 1e-9 else layout.copy()
    lay.meta["scale"] = layout.meta.get("scale", 1.0)
    if "scale" not in layout.meta:
        lay.meta.pop("scale", None)
    lay.surface = {"kind": surface["kind"], "height": 0.0 if surface["kind"] == "floor" else float(surface.get("height", 0.45))}
    return lay


def surface_from_ai(answer) -> dict | None:
    """The surface the AI named in its naming answer, or None (no answer, or an odd one)."""
    kind = answer.get("surface") if isinstance(answer, dict) else None
    if kind == "table":
        return dict(DEFAULT_SURFACE)
    if kind == "floor":
        return {"kind": "floor", "height": 0.0}
    if kind == "other":
        return {"kind": "other", "height": 0.45}
    return None


def erase_marker(bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Paint the printed marker (and its white border) with the colour of the table around it, so it is not an object."""
    c = corners.mean(0)
    big = (c + (corners - c) * 1.7).astype(np.int32)
    outer = (c + (corners - c) * 2.6).astype(np.int32)
    ring = np.zeros(bgr.shape[:2], np.uint8)
    cv2.fillConvexPoly(ring, outer, 255)
    cv2.fillConvexPoly(ring, big, 0)
    colour = np.median(bgr[ring > 0], axis=0) if (ring > 0).any() else (200, 200, 200)
    out = bgr.copy()
    cv2.fillConvexPoly(out, big, tuple(int(v) for v in colour))
    return out


def analyse(jpeg_or_bgr, pitch_deg: float | None = None, marker_m: float | None = None) -> dict:
    """Stage 1 (no AI): segment and measure. Returns everything the layout needs.

    With `marker_m` (the printed marker's side in metres) and the marker in the photo, the camera height and
    angle come from the marker instead of the phone's pitch and an assumed height."""
    bgr = jpeg_or_bgr if isinstance(jpeg_or_bgr, np.ndarray) else \
        cv2.imdecode(np.frombuffer(jpeg_or_bgr, np.uint8), cv2.IMREAD_COLOR)
    bgr = _prep(bgr)
    pitch = 45.0 if pitch_deg is None else float(pitch_deg)
    height, cal = CAM_HEIGHT, {"source": "estimate"}
    corners = MK.detect(bgr) if marker_m else None
    if corners is not None:
        p = MK.pose(corners, marker_m, bgr.shape, f=0.78 * max(bgr.shape[:2]))
        if p["error_px"] < 3.0:
            pitch, height = float(np.clip(p["pitch_deg"], 15, 89)), p["height"]
            cal = {"source": "marker", "height": round(height, 3), "pitch_deg": round(pitch, 1),
                   "error_px": round(p["error_px"], 2)}
            bgr = erase_marker(bgr, corners)
    cam = Camera(bgr.shape, pitch, height)
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
                      "rgb": [c / 255 for c in colour], "name": None,
                      "shape": guess_shape(o["mask"][y:y + h, x:x + w]), "mask": o["mask"]})
    return {"bgr": bgr, "table": table, "items": items, "cam": cam, "pitch": pitch, "calibration": cal,
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


def cube_atlas(rgb, near=None, top=None) -> np.ndarray:
    """The six faces of an object in one image (3 rows x 4 columns, the layout MuJoCo reads when the file ends in
    ATLAS_SUFFIX). `near` is the face the phone camera sees (its photo), `top` the face seen from above; the other faces
    are the object's plain colour, shaded. With no photo of a face, one colour is the better guess than a stretched photo."""
    base = np.array(rgb[::-1], float) * 255

    def plain(k):
        return np.full((TILE, TILE, 3), np.clip(base * k, 0, 255), np.uint8)

    tiles = {"U": plain(0.7), "D": plain(1.0), "L": plain(0.8), "R": plain(0.9), "F": plain(1.1), "B": plain(0.5)}
    if near is not None:
        tiles["D"] = cv2.resize(near, (TILE, TILE), interpolation=cv2.INTER_AREA)   # D: the face toward the camera (-y)
    if top is not None:
        tiles["F"] = cv2.resize(top, (TILE, TILE), interpolation=cv2.INTER_AREA)    # F: the top (+z), image up = +y
    atlas = np.zeros((3 * TILE, 4 * TILE, 3), np.uint8)
    for r, row in enumerate((".U..", "LFRB", ".D..")):
        for c, ch in enumerate(row):
            atlas[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = tiles.get(ch, 0)
    return atlas


def guess_shape(mask_crop: np.ndarray) -> str:
    """round or box from the outline alone (used until the AI says what the thing is): a compact blob whose corners are empty
    is rounded; a ragged mask (a wrapper, a cable) or one that fills its corners stays a box."""
    h, w = mask_crop.shape
    if h < 8 or w < 8:
        return "box"
    s = max(2, int(0.2 * min(h, w)))
    corners = float(np.mean([c.mean() for c in (mask_crop[:s, :s], mask_crop[:s, -s:], mask_crop[-s:, :s], mask_crop[-s:, -s:])]))
    return "round" if float(mask_crop.mean()) >= ROUND_FILL and corners < ROUND_CORNERS else "box"


def built_from_outline(shape: str, mask_crop: np.ndarray) -> bool:
    """Is the object built from its own outline (lathe.py) rather than as a box? Yes when called round or a cylinder, and
    also when its outline says so whatever it is called: rounded (guess_shape), or its width changes up its height
    (a mug, a bottle, a tub). A box seen from the side has straight sides and stays a box."""
    if shape in ROUND:
        return True
    if shape == "flat" or mask_crop is None or not mask_crop.any():
        return False
    if guess_shape(mask_crop) == "round":
        return True
    prof = lathe.profile(mask_crop)
    if prof is None:
        return False
    body = prof[2:-2] if len(prof) > 6 else prof               # not the foot and the top, which the outline rounds off
    return float(body.max() - body.min()) > TAPER


def is_thin(it: dict) -> bool:
    """A cable or a pen: long and thin in the photo, so it lies on the table and is not a block."""
    if it.get("mask") is None:
        return False
    x, y, w, h = it["box"]
    return w * h > 0 and float(it["mask"].sum()) / (w * h) < THIN


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
        if is_thin(it):
            it["shape"] = "flat"
        x, y = (it["xy"][0] - cx) * s, (it["xy"][1] - cy) * s
        w, d, h = (v * s for v in it["size"])
        yaw, skin, mesh = 0.0, None, {}
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
                top_view = cv2.getRectSubPix(rot, (max(4, int(rw)), max(4, int(rh))), (u, v))
                skin = cube_atlas((top_view.reshape(-1, 3).mean(0)[::-1] / 255).tolist(), top=top_view)   # edges: the top's own colour
            h = min(h, 0.012)
        elif it.get("mask") is not None:
            # standing things: skin = their photo, background filled with their own colour
            bx, by, bw, bh = it["box"]
            crop = bgr[by:by + bh, bx:bx + bw].copy()
            mk = it["mask"][by:by + bh, bx:bx + bw]
            from_outline = built_from_outline(it["shape"], mk)
            edge = 11 if from_outline else 7               # the mask edge is mostly table; a rounded thing shows its edge more
            inner = cv2.erode(mk.astype(np.uint8), np.ones((edge, edge), np.uint8),
                              borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)   # also at the photo's own edge
            crop[~(inner if inner.any() else mk)] = np.array(it["rgb"][::-1]) * 255
            skin = cube_atlas(it["rgb"], near=crop)
            if from_outline:                               # a rounded or tapered thing gets the shape of its own outline
                prof = lathe.profile(mk)
                if prof is not None:
                    v, uv, f = lathe.build(prof, max(w, 0.02), max(d, 0.02), max(h, 0.006))
                    (out_dir / f"prop_{i}.obj").write_text(lathe.to_obj(v, uv, f))
                    cv2.imwrite(str(out_dir / f"prop_{i}_tex.png"), lathe.texture(crop, it["rgb"]))
                    mesh = {"mesh": str((out_dir / f"prop_{i}.obj").resolve()), "mesh_scale": (1.0, 1.0, 1.0),
                            "mesh_texture": str((out_dir / f"prop_{i}_tex.png").resolve())}
        if skin is not None and skin.size:
            path = out_dir / f"prop_{i}{ATLAS_SUFFIX}"
            cv2.imwrite(str(path), skin)
            skin = str(path.resolve())
        props.append({"name": it["name"], "pos": (float(x), float(y)), "yaw": float(yaw),
                      "size": (float(max(w, 0.02) / 2), float(max(d, 0.02) / 2), float(max(h, 0.006) / 2)),
                      "rgb": it["rgb"], "shape": it["shape"], "skin": None if mesh else skin, **mesh})
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


SURFACE_WORD = {"table": "table", "floor": "floor", "other": "surface"}


def greeting(items: list[dict], kind: str = "table") -> str:
    word = SURFACE_WORD.get(kind, "surface")
    if not items:
        return f"I can see the {word}, but nothing on it. Put a few things down and snap again."
    listed = listing([it["name"] for it in items])
    where = "your table" if kind == "table" else f"the {word}"
    return f"I've mapped {where}. I see {listed}. That's an approximate map. What should I move?"
