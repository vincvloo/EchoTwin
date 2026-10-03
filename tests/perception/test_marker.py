"""The printed marker: page, detection, plane homography, camera pose. Needs cv2 (skipped where it is missing)."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from echotwin.perception import marker as MK                       # noqa: E402

W, H = 1280, 960
F = MK.F_FACTOR * max(W, H)                                          # the synthetic camera has the guessed focal length
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])


def camera(height=0.45, pitch_deg=55.0, yaw_deg=20.0, target=(0.0, 0.0)):
    """A camera `height` above the table, looking down at `pitch_deg` below the horizon towards `target` (table frame,
    x right, y away, z up). Returns R (world -> camera), t and the camera position."""
    yaw, pit = np.radians(yaw_deg), np.radians(pitch_deg)
    d = np.array([np.sin(yaw) * np.cos(pit), np.cos(yaw) * np.cos(pit), -np.sin(pit)])     # viewing direction
    right = np.cross(d, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(d, right)
    R = np.stack([right, down, d])
    dist = height / np.sin(pit)
    c = np.array([target[0], target[1], 0.0]) - d * dist
    return R, -R @ c, c


def photo(side=0.10, **cam):
    """A white photo with the tag lying on the table (centre at the origin). Returns the image and the true corners."""
    R, t, c = camera(**cam)
    tag = cv2.copyMakeBorder(MK.make_marker(400), 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    half = side / 2 * (480 / 400)                                     # the white quiet zone is part of the picture
    world = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    p = (K @ (R @ world.T + t[:, None])).T
    img_pts = (p[:, :2] / p[:, 2:]).astype(np.float32)
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [480, 0], [480, 480], [0, 480]]), img_pts)
    img = cv2.warpPerspective(tag, M, (W, H), borderValue=255)
    s = side / 2
    q = (K @ (R @ np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]]).T + t[:, None])).T
    return img, q[:, :2] / q[:, 2:]


def test_page_has_the_tag_at_exact_size():
    page, dpi = MK.make_page(100.0)
    assert page.shape[0] > page.shape[1] and abs(page.shape[1] / dpi * 25.4 - 210) < 1
    c = MK.detect(page)
    assert c is not None
    side_px = np.linalg.norm(c[1] - c[0])
    assert abs(side_px / dpi * 25.4 - 100.0) < 0.3                    # the printed square is 100 mm


def test_detects_corners_under_perspective():
    img, true = photo()
    c = MK.detect(img)
    assert c is not None and np.abs(c - true).max() < 1.0


def test_no_marker_gives_none():
    assert MK.detect(np.full((H, W), 200, np.uint8)) is None


def test_homography_measures_the_table_in_metres():
    img, _ = photo()
    c = MK.detect(img)
    inv = np.linalg.inv(MK.homography(c, 0.10))
    for a, b, metres in ((0, 1, 0.10), (1, 2, 0.10), (0, 2, 0.10 * 2 ** 0.5)):
        pa, pb = [inv @ np.array([*c[i], 1.0]) for i in (a, b)]
        assert np.linalg.norm(pa[:2] / pa[2] - pb[:2] / pb[2]) == pytest.approx(metres, rel=0.01)


@pytest.mark.parametrize("height,pitch", [(0.45, 55.0), (0.30, 70.0), (0.60, 40.0)])
def test_pose_recovers_camera_height_and_angle(height, pitch):
    img, _ = photo(height=height, pitch_deg=pitch)
    p = MK.pose(MK.detect(img), 0.10, img.shape)
    assert p["height"] == pytest.approx(height, rel=0.04)
    assert p["pitch_deg"] == pytest.approx(pitch, abs=4.0)
    assert p["error_px"] < 1.0


def test_marker_size_from_env(monkeypatch):
    monkeypatch.setenv("MARKER_SIZE_CM", "8")
    assert MK.size_m() == pytest.approx(0.08)
    monkeypatch.setenv("MARKER_SIZE_CM", "oops")
    assert MK.size_m() == pytest.approx(0.10)
