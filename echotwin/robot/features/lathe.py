"""A rounded object from its own outline: revolve the silhouette's width profile around the vertical axis.

Works for any rounded thing (a glass that widens at the top, a bottle, a ball, a mug, an earbud case): the shape comes from
the object's mask, nothing here knows what the object is. No heavy dependencies (numpy only).

The mesh is centred on its bounding box and written as an OBJ with texture coordinates, which is how `scene.build_xml`
expects a mesh prop. Texture: the left half of the image is the object's photo (for the side the camera sees), the right half
one plain colour (for the side nobody photographed).
"""
import numpy as np

RINGS = 20          # heights the profile is sampled at
SEGMENTS = 28       # points around each ring (half of them face the camera)
MIN_ROWS = 8        # a mask shorter than this is not an outline worth revolving
FLOOR = 0.5         # the foot is at least this share of the widest part, so the object stands


def profile(mask: np.ndarray, rings: int = RINGS) -> np.ndarray | None:
    """Half-width of the outline at `rings` heights, bottom first, scaled so the widest is 1. None if the mask is too small."""
    rows = np.nonzero(mask.any(axis=1))[0]
    if len(rows) < MIN_ROWS:
        return None
    m = mask[rows[0]:rows[-1] + 1]
    widths = np.array([(np.nonzero(r)[0].max() - np.nonzero(r)[0].min() + 1) / 2 if r.any() else 0.0 for r in m])
    bands = np.array_split(widths[::-1], rings)                       # bottom row first
    prof = np.array([b.mean() if len(b) else 0.0 for b in bands])
    k = np.ones(3) / 3
    prof = np.convolve(np.pad(prof, 1, mode="edge"), k, mode="valid")  # smooth the pixel noise
    top = prof.max()
    if top <= 0:
        return None
    prof = prof / top
    prof[0] = max(prof[0], FLOOR)
    prof[-1] = max(prof[-1], 0.08)                                     # a closed, slightly rounded top
    return prof


def build(prof: np.ndarray, width: float, depth: float, height: float, segments: int = SEGMENTS):
    """-> (vertices (n, 3), texture coordinates (n, 2), faces (m, 3) indices), in metres, centred on the bounding box.
    Texture v: 0 at the top, 1 at the bottom of the object (the first row of the photo is the top)."""
    rings = len(prof)
    rx, ry = width / 2, depth / 2
    z = np.linspace(-height / 2, height / 2, rings)
    verts, uvs, faces = [], [], []
    for near in (True, False):
        # near: angles from pi to 2*pi (y <= 0, toward the camera); far: 0 to pi. Own vertices, so the colours do not blend.
        a = np.linspace(np.pi, 2 * np.pi, segments // 2 + 1) if near else np.linspace(0, np.pi, segments // 2 + 1)
        base = len(verts)
        for i in range(rings):
            for ang in a:
                x, y = prof[i] * rx * np.cos(ang), prof[i] * ry * np.sin(ang)
                verts.append((x, y, z[i]))
                if near:
                    uvs.append((0.5 * (0.5 + 0.5 * np.cos(ang) * prof[i]), 1 - i / (rings - 1)))   # x on the photo, its width
                else:
                    uvs.append((0.75, 0.5))
        cols = len(a)
        for i in range(rings - 1):
            for j in range(cols - 1):
                p = base + i * cols + j
                faces += [(p, p + 1, p + cols), (p + 1, p + cols + 1, p + cols)] if near else \
                         [(p, p + cols, p + 1), (p + 1, p + cols, p + cols + 1)]
        for top in (False, True):                                     # a cap on each end, a fan around its centre
            c = len(verts)
            i = rings - 1 if top else 0
            verts.append((0.0, 0.0, z[i]))
            uvs.append((0.25, 0.0 if top else 1.0) if near else (0.75, 0.5))
            for j in range(cols - 1):
                p = base + i * cols + j
                faces.append((c, p, p + 1) if top == near else (c, p + 1, p))
    return np.array(verts), np.array(uvs), np.array(faces, int)


def _outward(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Turn every face so its normal points away from the centre (the shape is convex and centred, so this is always right)."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(b - a, c - a)
    out = np.einsum("ij,ij->i", n, (a + b + c) / 3) < 0
    faces = faces.copy()
    faces[out] = faces[out][:, ::-1]
    return faces


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(b - a, c - a)                                        # bigger faces count more
    acc = np.zeros_like(verts)
    for k in range(3):
        np.add.at(acc, faces[:, k], n)
    return acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)


def to_obj(verts: np.ndarray, uvs: np.ndarray, faces: np.ndarray) -> str:
    faces = _outward(verts, faces)
    normals = vertex_normals(verts, faces)
    lines = [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts]
    lines += [f"vt {u:.4f} {v:.4f}" for u, v in uvs]
    lines += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in normals]
    lines += [f"f {a + 1}/{a + 1}/{a + 1} {b + 1}/{b + 1}/{b + 1} {c + 1}/{c + 1}/{c + 1}" for a, b, c in faces]
    return "\n".join(lines) + "\n"


def texture(photo: np.ndarray, rgb) -> np.ndarray:
    """The image the mesh's texture coordinates read: photo on the left, the object's plain colour on the right."""
    import cv2
    tile = 256
    left = cv2.resize(photo, (tile, tile), interpolation=cv2.INTER_AREA)
    right = np.full((tile, tile, 3), np.clip(np.array(rgb[::-1]) * 255 * 0.85, 0, 255), np.uint8)
    return np.hstack([left, right])
