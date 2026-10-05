"""How the twin's objects look: one tile per face (photo on the face the camera sees), thin things lie flat."""
import cv2
import numpy as np

from echotwin.robot.features import everyday as E
from echotwin.robot.scene import ATLAS_SUFFIX, Layout, build_xml
from echotwin.robot.world import World


def test_the_atlas_has_the_photo_only_on_the_face_toward_the_camera():
    photo = np.zeros((40, 30, 3), np.uint8)
    photo[:] = (0, 0, 255)                                              # a red photo (BGR)
    a = E.cube_atlas((0.2, 0.4, 0.8), near=photo)
    T = E.TILE
    assert a.shape == (3 * T, 4 * T, 3)
    tile = lambda r, c: a[r * T:(r + 1) * T, c * T:(c + 1) * T]
    assert (tile(2, 1) == (0, 0, 255)).all()                            # D, the near face, is the photo
    for r, c in [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3)]:               # the other faces are one shaded colour
        t = tile(r, c)
        assert (t == t[0, 0]).all() and not (t == (0, 0, 255)).all()
    assert tile(1, 0)[0, 0][2] < tile(1, 0)[0, 0][0]                    # the colour is the object's (blue-ish in BGR)


def test_an_atlas_skin_is_read_as_six_faces_and_a_plain_skin_as_one_image():
    base = {"name": "x", "shape": "box", "pos": (0, 0), "yaw": 0, "size": (0.03, 0.03, 0.03), "rgb": (0.5, 0.5, 0.5)}
    atlas = build_xml(_layout({**base, "skin": f"/t/prop_0{ATLAS_SUFFIX}"}))
    plain = build_xml(_layout({**base, "skin": "/t/prop_0.png"}))
    assert 'gridsize="3 4" gridlayout=".U..LFRB.D.."' in atlas and "gridsize" not in plain


def _layout(prop):
    lay = Layout()
    lay.props = [prop]
    return lay


def _mask(shape, pts, thickness):
    m = np.zeros(shape, np.uint8)
    cv2.polylines(m, [np.array(pts, np.int32)], False, 1, thickness)
    return m.astype(bool)


def test_a_long_thin_thing_is_flat_and_a_blob_is_not():
    cable = _mask((400, 300), [(30, 380), (150, 200), (270, 20)], 6)    # diagonal and thin: a big box, little mask
    ys, xs = np.nonzero(cable)
    box = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())]
    assert E.is_thin({"box": box, "mask": cable})
    blob = np.zeros((400, 300), bool)
    blob[100:200, 100:200] = True
    assert not E.is_thin({"box": [100, 100, 100, 100], "mask": blob})
    assert not E.is_thin({"box": [0, 0, 5, 5]})                          # no mask: nothing to say


def test_the_twin_of_a_photo_has_atlas_skins_and_no_brick_for_a_cable(tmp_path):
    img = np.full((540, 960, 3), (190, 205, 215), np.uint8)             # a plain table
    cv2.ellipse(img, (300, 330), (60, 45), 0, 0, 360, (30, 30, 30), -1)
    cv2.line(img, (700, 520), (800, 300), (250, 250, 250), 9)           # a white cable, diagonal
    cv2.line(img, (700, 520), (800, 300), (90, 90, 90), 2)
    res = E.analyse(img, 45.0)
    for i, it in enumerate(res["items"]):
        it["name"] = f"thing {i}"
    props, tex, ann = E.build(res, tmp_path)
    assert props and all(p["skin"].endswith(ATLAS_SUFFIX) for p in props if p.get("skin"))
    assert any(p["skin"] for p in props)
    for p in props:                                                     # whatever it is, it builds into the sim
        assert p["size"][2] > 0
    lay = Layout()
    lay.texture, lay.props = tex, props
    World(lay).settle(5)
