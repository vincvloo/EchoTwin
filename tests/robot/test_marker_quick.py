"""Quick mode with the printed marker: the camera height and angle come from the marker, not from a guess."""
import cv2
import numpy as np
import pytest

from echotwin.perception import marker as MK
from echotwin.robot.features import everyday as E

SHAPE = (960, 540, 3)
TRUE_H, TRUE_PITCH = 0.45, 55.0


def scene(marker_at=(-0.07, 0.33), obj_at=(0.07, 0.30), obj_w=0.06):
    """A plain table seen by the camera model at the true height and angle, with a 10 cm marker and a red 6 cm block."""
    cam = E.Camera(SHAPE, TRUE_PITCH, TRUE_H)
    rng = np.random.default_rng(0)
    img = np.full(SHAPE, 225, np.uint8)
    img[:] += np.linspace(0, 20, SHAPE[0], dtype=np.uint8)[:, None, None]
    img = cv2.add(img, rng.integers(0, 6, SHAPE, dtype=np.uint8))
    img[:120] = (60, 70, 90)                                                   # background behind the table
    tag = cv2.copyMakeBorder(MK.make_marker(400), 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    half = 0.05 * 480 / 400
    cx, cy = marker_at
    world = [(cx - half, cy + half), (cx + half, cy + half), (cx + half, cy - half), (cx - half, cy - half)]
    dst = np.float32([cam.project([x, y, 0]) for x, y in world])
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [480, 0], [480, 480], [0, 480]]), dst)
    patch = cv2.warpPerspective(cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR), M, (SHAPE[1], SHAPE[0]), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full(tag.shape, 255, np.uint8), M, (SHAPE[1], SHAPE[0]))
    img[mask > 0] = patch[mask > 0]
    ox, oy = obj_at
    quad = np.int32([cam.project([ox + dx, oy + dy, 0]) for dx, dy in
                     ((-obj_w / 2, -obj_w / 2), (obj_w / 2, -obj_w / 2), (obj_w / 2, obj_w / 2), (-obj_w / 2, obj_w / 2))])
    cv2.fillConvexPoly(img, quad, (30, 30, 200))
    return img


def test_marker_gives_height_and_pitch_and_true_size():
    res = E.analyse(scene(), pitch_deg=30.0, marker_m=0.10)                     # the phone's pitch reading is far off
    cal = res["calibration"]
    assert cal["source"] == "marker"
    assert cal["height"] == pytest.approx(TRUE_H, rel=0.06) and cal["pitch_deg"] == pytest.approx(TRUE_PITCH, abs=5)
    boxes = [i for i in res["items"] if i["rgb"][2] < 0.3 and i["rgb"][0] > 0.5]    # the red block (bgr -> rgb below)
    red = [i for i in res["items"] if i["rgb"][0] > 0.6 and i["rgb"][1] < 0.3]
    assert len(red) == 1, [i["rgb"] for i in res["items"]]
    assert red[0]["size"][0] == pytest.approx(0.06, abs=0.015)
    del boxes


def test_the_marker_itself_is_not_an_object():
    res = E.analyse(scene(), pitch_deg=30.0, marker_m=0.10)
    assert len(res["items"]) == 1


def test_without_a_marker_nothing_changes():
    img = scene(marker_at=(5.0, 5.0))                                           # marker far outside the picture
    res = E.analyse(img, pitch_deg=40.0, marker_m=0.10)
    assert res["calibration"]["source"] == "estimate" and res["pitch"] == 40.0
    assert E.analyse(img, 40.0)["calibration"]["source"] == "estimate"
