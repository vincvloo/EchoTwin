"""Scale and table plane from the marker, on a synthetic scene whose true size is known."""
import numpy as np
import pytest

from echotwin.perception import marker_scale as MS

W, H = 1280, 960                  # the photo
mw, mh = 518, 392                 # what the model sees
F = 0.75 * W
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
Km = K * np.array([[mw / W], [mh / H], [1]])
Km[2, 2] = 1.0
UNITS = 3.7                       # VGGT units per metre: the unknown the marker must recover
SIDE = 0.10


def rot(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return (np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]) @ np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
            @ np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]]))


Q = rot(0.4, -0.3, 0.7)           # the table in VGGT's arbitrary world frame
T = np.array([0.3, -0.2, 0.5])


def view(yaw_deg, height=0.45, pitch_deg=55.0):
    """Camera-from-table-frame (R, t) in metres, looking at the origin; x right, y away, z up."""
    yaw, pit = np.radians(yaw_deg), np.radians(pitch_deg)
    d = np.array([np.sin(yaw) * np.cos(pit), np.cos(yaw) * np.cos(pit), -np.sin(pit)])
    right = np.cross(d, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(d, right), d])
    c = -d * (height / np.sin(pit))
    return R, -R @ c


def extr_world(R, t):
    return np.hstack([R @ Q.T, (UNITS * (t - R @ Q.T @ T))[:, None]])


def point_map(R, t):
    """World point (VGGT units) of every pixel of the model image: the ray meets the table plane z = 0."""
    ys, xs = np.mgrid[0:mh, 0:mw]
    rays = (np.linalg.inv(Km) @ np.stack([xs.ravel(), ys.ravel(), np.ones(mw * mh)])).T @ R      # in table frame
    c = -R.T @ t
    lam = -c[2] / rays[:, 2]
    p = c + rays * lam[:, None]
    return (UNITS * (p @ Q.T + T)).reshape(mh, mw, 3)


def corners_px(R, t, side=SIDE):
    s = side / 2
    p = (Km @ (R @ np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]]).T + t[:, None])).T
    return p[:, :2] / p[:, 2:]


def frames(yaws=(0, 90, 180, 270), units_error=None):
    out = []
    for i, yw in enumerate(yaws):
        R, t = view(yw)
        pm = point_map(R, t)
        if units_error and i == len(yaws) - 1:
            pm = pm * units_error                    # one photo whose depth is off
        out.append({"point_map": pm, "corners": corners_px(R, t), "extr": extr_world(R, t), "intr": Km})
    return out


def test_scale_is_recovered():
    r = MS.estimate(frames(), SIDE)
    assert r["reliable"] and r["photos_seen"] == 4
    assert r["scale"] == pytest.approx(1 / UNITS, rel=0.01)          # metres per VGGT unit


def test_noisy_points_still_give_the_scale_within_3_percent():
    rng = np.random.default_rng(0)
    fr = frames()
    for f in fr:
        f["point_map"] = f["point_map"] * (1 + rng.normal(0, 0.005, f["point_map"].shape))
        f["corners"] = f["corners"] + rng.normal(0, 0.4, (4, 2))
    r = MS.estimate(fr, SIDE)
    assert r["reliable"] and r["scale"] == pytest.approx(1 / UNITS, rel=0.03)


def test_photos_that_disagree_are_not_trusted():
    r = MS.estimate(frames(units_error=1.35), SIDE)
    assert r is not None and not r["reliable"] and r["spread"] > MS.MAX_SPREAD


def test_one_photo_is_not_enough():
    r = MS.estimate(frames(yaws=(0,)), SIDE)
    assert r is not None and not r["reliable"]


def test_plane_origin_and_axes_match_the_table():
    r = MS.estimate(frames(), SIDE)
    n = np.array(r["normal"])
    assert abs(n @ (Q @ np.array([0, 0, 1.0]))) == pytest.approx(1.0, abs=1e-3)       # the table normal
    assert np.array(r["origin"]) == pytest.approx(UNITS * (Q @ np.zeros(3) + T), abs=0.02)
    x = np.array(r["x_axis"])
    assert abs(x @ (Q @ np.array([1.0, 0, 0]))) == pytest.approx(1.0, abs=0.01)       # along the marker's top edge


def test_from_photos_finds_the_marker_in_rendered_photos():
    cv2 = pytest.importorskip("cv2")
    from echotwin.perception import marker as MK
    fr = frames()
    photos = []
    tag = cv2.copyMakeBorder(MK.make_marker(400), 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    for yw in (0, 90, 180, 270):
        R, t = view(yw)
        half = SIDE / 2 * (480 / 400)
        world = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
        p = (K @ (R @ world.T + t[:, None])).T
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [480, 0], [480, 480], [0, 480]]), (p[:, :2] / p[:, 2:]).astype(np.float32))
        photos.append(cv2.warpPerspective(tag, M, (W, H), borderValue=255))
    photos.append(np.full((H, W), 200, np.uint8))                       # a fifth photo without the marker
    maps = np.stack([f["point_map"] for f in fr] + [fr[0]["point_map"]])
    ex = np.stack([f["extr"] for f in fr] + [fr[0]["extr"]])
    r = MS.from_photos(photos, maps, ex, np.stack([Km] * 5), SIDE)
    assert r["reliable"] and r["photos_seen"] == 4 and r["photos_total"] == 5
    assert r["scale"] == pytest.approx(1 / UNITS, rel=0.02)
