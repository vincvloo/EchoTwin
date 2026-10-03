"""The 3D scan's table texture: a synthetic table with a known colour pattern, seen from four sides."""
import cv2
import numpy as np
import pytest

from echotwin.robot.features.everyday import TEX_PX
from echotwin.robot.twin_import import texture3d as TX
from echotwin.scene import schema

PW, PH = 640, 480                  # a photo
mw, mh = 518, 392                  # the model's pixels
F = 0.75 * PW
K_photo = np.array([[F, 0, PW / 2], [0, F, PH / 2], [0, 0, 1.0]])
K_model = np.array([[F * mw / PW, 0, mw / 2], [0, F * mh / PH, mh / 2], [0, 0, 1.0]])
UNITS = 3.7
Q = np.array([[0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, 1.0]])                 # the table in VGGT's arbitrary frame
T = np.array([0.3, -0.2, 0.5])


def pattern(x, y):
    """Table colour (RGB) at map position (x, y): red on the left of x = 0.05, blue on the right, a green 6 cm square."""
    col = np.where((x < 0.05)[..., None], [200, 40, 40], [40, 40, 200]).astype(np.uint8)
    green = (np.abs(x - 0.20) < 0.03) & (np.abs(y - 0.05) < 0.03)
    col[green] = [40, 200, 40]
    return col


def view(yaw_deg, height=0.45, pitch_deg=55.0):
    yaw, pit = np.radians(yaw_deg), np.radians(pitch_deg)
    d = np.array([np.sin(yaw) * np.cos(pit), np.cos(yaw) * np.cos(pit), -np.sin(pit)])
    right = np.cross(d, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(d, right), d])
    c = np.array([0.1, 0.0, 0.0]) - d * (height / np.sin(pit))
    return R, -R @ c


def render(R, t, hide=None):
    """What the camera sees of the table: every pixel's ray meets z = 0; `hide` paints a square object over the table."""
    ys, xs = np.mgrid[0:PH, 0:PW]
    rays = (np.linalg.inv(K_photo) @ np.stack([xs.ravel(), ys.ravel(), np.ones(PW * PH)])).T @ R
    c = -R.T @ t
    lam = -c[2] / np.where(rays[:, 2] < -1e-6, rays[:, 2], -1e-6)
    p = c + rays * lam[:, None]
    img = pattern(p[:, 0].reshape(PH, PW), p[:, 1].reshape(PH, PW))
    if hide:
        (x0, y0), (x1, y1) = hide
        inside = ((p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)).reshape(PH, PW)
        img[inside] = [250, 250, 250]
    return img


@pytest.fixture
def scan(tmp_path):
    extr, frames = [], {}
    for i, yaw in enumerate((0, 90, 180, 270, 45)):
        R, t = view(yaw)
        extr.append(np.hstack([R @ Q.T, (UNITS * (t - R @ Q.T @ T))[:, None]]))
        # an object stands at (0.05, 0.0): in view 0 only it hides the table there
        frames[f"frame_{i}"] = render(R, t, hide=((-0.02, -0.04), (0.08, 0.06)) if i == 0 else None)
    np.savez(tmp_path / "cloud.cams.npz", extrinsic=np.stack(extr), intrinsic=np.stack([K_model] * 5), up_rotation=np.eye(3))
    np.savez(tmp_path / "cloud.pix.npz", model_hw=np.array([mh, mw]), **frames)
    # cloud = UNITS * (Q p_map + T)  =>  p_map = Q^T (cloud / UNITS - T)
    A = Q.T / UNITS
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = A, -Q.T @ T
    doc = schema.build_scene(
        [{"class": "cup", "x": 0.05, "y": 0.0, "size_x": 0.07, "size_y": 0.07, "height": 0.1}], {}, name="t",
        extra={"calibration": {"source": "marker", "scale": 1 / UNITS, "marker_size_m": 0.10, "cloud_to_map": M.tolist()}})
    return doc, tmp_path


def texel(scene, x, y):
    """Column and row of the map point (x, y) in the texture."""
    P, (cx, cy, w, h) = TX.window_points(scene, 0.0, TEX_PX)
    return int((x - cx) / w * TEX_PX[0] + TEX_PX[0] / 2), int(TEX_PX[1] / 2 - (y - cy) / h * TEX_PX[1])


def test_texture_has_the_table_colours_in_the_right_places(scan):
    scene, folder = scan
    name = TX.build(scene, folder)
    assert name == "table.png"
    img = cv2.cvtColor(cv2.imread(str(folder / name)), cv2.COLOR_BGR2RGB)
    assert img.shape[1::-1] == TEX_PX
    c, r = texel(scene, 0.15, -0.12)
    assert img[r, c] == pytest.approx([40, 40, 200], abs=25)             # right of x = 0.05: blue
    c, r = texel(scene, -0.10, 0.10)
    assert img[r, c] == pytest.approx([200, 40, 40], abs=25)             # left: red
    c, r = texel(scene, 0.20, 0.05)
    assert img[r, c] == pytest.approx([40, 200, 40], abs=40)             # the green square, 20 cm right of the origin


def test_an_object_hiding_the_table_in_one_photo_does_not_show(scan):
    scene, folder = scan
    TX.build(scene, folder)
    img = cv2.cvtColor(cv2.imread(str(folder / "table.png")), cv2.COLOR_BGR2RGB)
    c, r = texel(scene, 0.04, 0.02)                                      # under the object in photo 0, free in the others
    assert img[r, c].min() < 120, img[r, c]                              # not the object's white


def test_no_texture_without_the_files_or_the_transform(tmp_path, scan):
    scene, folder = scan
    assert TX.build(scene, tmp_path / "nowhere") is None
    no_cal = {**scene, "calibration": {"source": "estimate", "scale": 1.0}}
    assert TX.build(no_cal, folder) is None
