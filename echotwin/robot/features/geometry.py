"""What an object's shape is, measured from the geometry the twin really built: a box, a cylinder, an ellipsoid or a
scanned mesh, all the same way. The robot reads this, not the shape label (flat, box, cylinder, round), which only says
how the object was drawn.

Everything is computed from points on the object's surface in its own frame (its body in the simulation), turned by its
current rotation when asked, so a thing that was knocked over or turned is measured as it lies now.
"""
from __future__ import annotations

import mujoco
import numpy as np

N_DIRECTIONS = 90             # directions tried for the narrowest width across (every 2 degrees)
ROUND_OUTLINE = 0.03          # widths across within 3 % of each other: a round outline, no narrow direction of its own
BALL_ROUNDNESS = 1.15         # the surface is never more than 15 % further from the centre than its nearest point: a ball
GRIP_BAND = 0.6               # the jaws close on the lower 60 % of a thing (where the plan puts the pads, prop_skills.waypoints)


def _fibonacci(n: int) -> np.ndarray:
    """n directions spread evenly over the sphere."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def _ellipsoid(r) -> np.ndarray:
    """Points on it, with the tips of its axes (where its widths are measured)."""
    tips = np.vstack([np.eye(3), -np.eye(3)])
    return np.vstack([_fibonacci(1500), tips]) * np.asarray(r, float)


def _box(h) -> np.ndarray:
    g = np.linspace(-1.0, 1.0, 9)
    a, b = np.meshgrid(g, g)
    a, b = a.ravel(), b.ravel()
    one = np.ones_like(a)
    faces = [np.stack(f, axis=1) for s in (-1, 1) for f in ((s * one, a, b), (a, s * one, b), (a, b, s * one))]
    return np.concatenate(faces) * np.asarray(h, float)


def _cylinder(r: float, hz: float) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    rim = np.stack([np.cos(t), np.sin(t)], axis=1) * r / np.cos(np.pi / 120)   # around the circle: widths never short
    side = [np.column_stack([rim, np.full(len(t), z)]) for z in np.linspace(-hz, hz, 7)]
    caps = [np.column_stack([rim * f, np.full(len(t), s * hz)]) for s in (-1, 1) for f in (0.0, 0.5)]
    return np.concatenate(side + caps)


def _mesh_surface(model: "mujoco.MjModel", m: int, n: int = 3000) -> np.ndarray:
    """Points spread over a mesh's surface by area, and its corners: a mesh's corners alone can all lie on its rims
    (a cylinder's), which says nothing about its sides."""
    a = int(model.mesh_vertadr[m])
    v = model.mesh_vert[a:a + int(model.mesh_vertnum[m])].astype(float)
    f = model.mesh_face[int(model.mesh_faceadr[m]):int(model.mesh_faceadr[m]) + int(model.mesh_facenum[m])]
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if area.sum() <= 0:
        return v
    rng = np.random.default_rng(0)                       # the same points every time: measurements do not wobble
    pick = rng.choice(len(f), size=n, p=area / area.sum())
    u, w = rng.random(n), rng.random(n)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    t = tri[pick]
    return np.vstack([v, t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + w[:, None] * (t[:, 2] - t[:, 0])])


def surface_points(model: "mujoco.MjModel", geom: int) -> np.ndarray:
    """Points on a geom's surface, in its body's frame (the twin's own shape: a mesh as MuJoCo placed it)."""
    t = int(model.geom_type[geom])
    s = model.geom_size[geom]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        pts = _mesh_surface(model, int(model.geom_dataid[geom]))
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        pts = _box(s[:3])
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        pts = _cylinder(float(s[0]), float(s[1]))
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        pts = _ellipsoid(s[:3])
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        pts = _ellipsoid([float(s[0])] * 3)
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        d = _fibonacci(1500) * float(s[0])
        pts = d + np.array([0.0, 0.0, float(s[1])]) * np.sign(d[:, 2:3] + 1e-12)
    else:                                                     # anything else: its bounding box
        pts = _box(model.geom_aabb[geom][3:])
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, model.geom_quat[geom])
    return pts @ R.reshape(3, 3).T + model.geom_pos[geom]


def mesh_volume(model: "mujoco.MjModel", geom: int) -> float | None:
    """The volume a mesh geom encloses (m^3), from its triangles; None for the other kinds."""
    if int(model.geom_type[geom]) != mujoco.mjtGeom.mjGEOM_MESH:
        return None
    m = int(model.geom_dataid[geom])
    v = model.mesh_vert[int(model.mesh_vertadr[m]):int(model.mesh_vertadr[m]) + int(model.mesh_vertnum[m])].astype(float)
    f = model.mesh_face[int(model.mesh_faceadr[m]):int(model.mesh_faceadr[m]) + int(model.mesh_facenum[m])]
    vol = abs(float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum()) / 6.0)
    return vol if vol > 0 else None


class Shape:
    """One object's shape: surface points in its own frame (centred on its body), and what follows from them."""

    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, float)
        r = np.linalg.norm(self.points - self.points.mean(axis=0), axis=1)
        self.roundness = float(r.max() / max(r.min(), 1e-6))   # 1.0 for a sphere, about 1.7 for a cube

    @property
    def has_up(self) -> bool:
        """Does it matter which way up it lies? Not for a ball: its surface is about as far from its centre everywhere."""
        return self.roundness > BALL_ROUNDNESS

    def placed(self, rot: np.ndarray) -> np.ndarray:
        """The points turned as the object lies now (rot: its 3x3 rotation), still centred on it."""
        return self.points @ np.asarray(rot, float).T

    def widths(self, rot: np.ndarray, band: float = GRIP_BAND) -> tuple[np.ndarray, np.ndarray]:
        """(directions in radians, width across in each), seen from above, of the part the jaws close on: the lower
        `band` of its height as it lies now."""
        low = self._low(rot, band)
        ang = np.linspace(0.0, np.pi, N_DIRECTIONS, endpoint=False)
        d = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        proj = low[:, :2] @ d.T
        return ang, proj.max(axis=0) - proj.min(axis=0)

    def grip(self, rot: np.ndarray) -> tuple[float, float | None]:
        """(how wide the jaws must open, the direction to close them along in radians, in [0, pi)). The direction is
        None for a round outline (any direction is as good). The narrowest width of an outline is found with one of its
        edges flat against a jaw, so it is exact from the outline's edges, not from a grid of directions."""
        _, w = self.widths(rot)
        hull = _hull(self._low(rot)[:, :2])
        best = (np.inf, 0.0)
        for a, b in zip(hull, np.roll(hull, -1, axis=0)):
            e = b - a
            if np.hypot(*e) < 1e-9:
                continue
            nrm = np.array([-e[1], e[0]]) / np.hypot(*e)                # across the edge: the jaws close along this
            width = float(np.ptp(hull @ nrm))
            if width < best[0] - 1e-12:
                best = (width, float(np.arctan2(nrm[1], nrm[0]) % np.pi))
        width = min(best[0], float(w.min()))
        if w.max() - width <= ROUND_OUTLINE * w.max():
            return width, None
        return width, best[1]

    def _low(self, rot: np.ndarray, band: float = GRIP_BAND) -> np.ndarray:
        p = self.placed(rot)
        z = p[:, 2]
        return p[z <= z.min() + band * (z.max() - z.min()) + 1e-9]

    def top_outline(self, rot: np.ndarray, n: int = 9) -> np.ndarray:
        """Up to n points (x, y, relative to its centre) under its top: where a hand coming down would sit on it."""
        p = self.placed(rot)
        z = p[:, 2]
        top = p[z >= z.max() - 0.15 * (z.max() - z.min()) - 1e-9][:, :2]
        if len(top) <= n:
            return top
        keep = [int(np.argmin(np.linalg.norm(top, axis=1)))]          # the middle, then the points furthest apart
        for _ in range(n - 1):
            d = np.min(np.linalg.norm(top[:, None] - top[keep][None], axis=2), axis=1)
            keep.append(int(np.argmax(d)))
        return top[keep]


def _turn(a, b, c) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _hull(p: np.ndarray) -> np.ndarray:
    """The convex outline of 2D points, counter-clockwise (monotone chain)."""
    p = np.unique(np.round(p, 9), axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        out = []
        for q in pts:
            while len(out) >= 2 and _turn(out[-2], out[-1], q) <= 0:
                out.pop()
            out.append(q)
        return out
    lower, upper = half(p), half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def of_body(model: "mujoco.MjModel", body: int) -> Shape:
    """The shape of everything a body is made of."""
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body]
    return Shape(np.concatenate([surface_points(model, g) for g in geoms]))
