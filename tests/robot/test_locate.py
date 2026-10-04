"""Finding an object by looking: the blob finder and the pixel-to-table map."""
import cv2
import numpy as np
import pytest

from echotwin.robot.features import locate as L


def table(w=640, h=480, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (150, 175, 205), np.uint8)                   # a tan table (BGR)
    return cv2.add(img, rng.integers(0, 6, img.shape, dtype=np.uint8))


def test_finds_a_coloured_block_near_where_it_should_be():
    img = table()
    cv2.rectangle(img, (300, 200), (340, 250), (30, 30, 200), -1)         # red, centre (320, 225) approximately
    f = L.locate(img, (310, 215), 40)                                     # the belief is 14 px off
    assert f is not None and np.hypot(f.px[0] - 320, f.px[1] - 225) < 2.0
    assert abs(f.area - 41 * 51) < 120


def test_nothing_there_gives_none():
    assert L.locate(table(), (320, 240), 40) is None


def test_a_faint_shadow_is_not_an_object():
    img = table()
    shade = img.copy()
    cv2.ellipse(shade, (320, 240), (40, 25), 0, 0, 360, (130, 155, 185), -1)           # a slightly darker patch
    img = cv2.GaussianBlur(shade, (0, 0), 6)
    assert L.locate(img, (320, 240), 40) is None


def test_the_nearest_of_two_blocks_wins():
    img = table()
    cv2.rectangle(img, (100, 100), (140, 140), (200, 40, 40), -1)
    cv2.rectangle(img, (400, 300), (440, 340), (40, 200, 40), -1)
    f = L.locate(img, (415, 310), 40)
    assert abs(f.px[0] - 420) < 3 and abs(f.px[1] - 320) < 3


def test_a_neighbour_in_the_window_border_does_not_fool_the_table_colour():
    img = table()
    cv2.rectangle(img, (300, 200), (340, 250), (30, 30, 200), -1)
    cv2.rectangle(img, (380, 200), (400, 215), (200, 40, 40), -1)         # another block at the edge of the window
    f = L.locate(img, (320, 225), 40)
    assert np.hypot(f.px[0] - 320, f.px[1] - 225) < 2.0


def test_the_window_is_clipped_at_the_edge_of_the_frame():
    img = table()
    cv2.rectangle(img, (2, 2), (40, 40), (30, 30, 200), -1)
    f = L.locate(img, (20, 20), 40)
    assert f is not None and f.px[0] < 30


# ---------------- PlaneMap ----------------
def top():
    return L.PlaneMap.from_camera((0, 0, 1.0), (1, 0, 0, 0, 1, 0), 45.0, (640, 480))


def test_top_camera_round_trip_and_scale():
    pm = top()
    for xy, z in (((0.1, -0.05), 0.0), ((-0.3, 0.2), 0.05), ((0.35, 0.25), 0.1)):
        assert pm.to_table(pm.to_pixel(xy, z), z) == pytest.approx(xy, abs=1e-9)
    # straight down: the image centre is the origin, right is +x and up is +y
    assert pm.to_table((320, 240)) == pytest.approx((0, 0), abs=1e-9)
    assert pm.to_table((420, 240))[0] > 0 and pm.to_table((320, 140))[1] > 0
    assert 560 < pm.metres_to_pixels(1.0, (0, 0)) < 600                    # 240 / tan(22.5 deg) = 579 px per metre at 1 m


def test_height_matters_for_a_camera_but_the_table_point_is_right_at_its_own_height():
    pm = top()
    px = pm.to_pixel((0.3, 0.2), 0.10)                                    # the top of a 10 cm object
    assert np.linalg.norm(pm.to_table(px, 0.10) - (0.3, 0.2)) < 1e-9
    assert np.linalg.norm(pm.to_table(px, 0.0) - (0.3, 0.2)) > 0.015       # read at the table, it would be about 3 cm off


def test_homography_map_round_trip():
    H = np.array([[600.0, 40.0, 320.0], [-30.0, 590.0, 400.0], [0.02, 0.1, 1.0]])
    pm = L.PlaneMap.from_homography(H)
    assert pm.to_table(pm.to_pixel((0.12, 0.34))) == pytest.approx((0.12, 0.34), abs=1e-9)
