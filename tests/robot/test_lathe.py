"""Rounded objects get the shape of their own outline (features/lathe.py), whatever the object is."""
import cv2
import numpy as np
import pytest

from echotwin.robot.features import everyday as E
from echotwin.robot.features import lathe as L
from echotwin.robot.scene import Layout
from echotwin.robot.world import World


def _mask(poly, shape=(120, 100)):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    return m.astype(bool)


CONE = [(10, 110), (90, 110), (55, 10), (45, 10)]                       # wide foot, narrow top
CUP = [(30, 110), (70, 110), (95, 10), (5, 10)]                         # a tapered glass: wide at the top
BALL = None


def _ball():
    m = np.zeros((100, 100), np.uint8)
    cv2.circle(m, (50, 50), 45, 1, -1)
    return m.astype(bool)


def test_the_profile_follows_the_outline_whatever_it_is():
    cone, cup, ball = L.profile(_mask(CONE)), L.profile(_mask(CUP)), L.profile(_ball())
    assert cone[0] == pytest.approx(1.0, abs=0.05) and cone[-1] < 0.3 and (np.diff(cone[2:]) <= 0.02).all()    # narrows going up
    assert cup[-1] == pytest.approx(1.0, abs=0.05) and cup[0] < 0.6                                         # widens going up
    assert int(np.argmax(ball)) in range(7, 13) and ball[0] >= L.FLOOR and ball[-1] < 0.5                    # widest in the middle
    assert L.profile(np.zeros((50, 50), bool)) is None and L.profile(_mask([(1, 1), (9, 1), (9, 4)], (10, 10))) is None


def test_the_mesh_is_centred_closed_and_the_size_of_the_object():
    prof = L.profile(_mask(CUP))
    v, uv, f = L.build(prof, 0.08, 0.06, 0.12)
    assert v[:, 0].min() == pytest.approx(-0.04, abs=1e-6) and v[:, 0].max() == pytest.approx(0.04, abs=1e-6)
    assert v[:, 1].min() == pytest.approx(-0.03, abs=1e-6) and v[:, 2].min() == pytest.approx(-0.06) and v[:, 2].max() == pytest.approx(0.06)
    assert f.min() >= 0 and f.max() < len(v) and len(uv) == len(v) and uv.min() >= 0 and uv.max() <= 1
    f = L._outward(v, f)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    assert (np.einsum("ij,ij->i", np.cross(b - a, c - a), (a + b + c) / 3) >= -1e-12).all()                  # every face looks outward
    n = L.vertex_normals(v, f)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)


def test_the_camera_side_uses_the_photo_and_the_far_side_one_plain_colour():
    v, uv, f = L.build(L.profile(_ball()), 0.06, 0.06, 0.06)
    near, far = v[:, 1] < -1e-9, v[:, 1] > 1e-9
    assert (uv[near][:, 0] <= 0.5).all() and (uv[far][:, 0] >= 0.5).all()
    img = L.texture(np.full((40, 30, 3), (10, 20, 200), np.uint8), (0.5, 0.5, 0.5))
    assert img.shape == (256, 512, 3) and tuple(img[10, 10]) == (10, 20, 200) and len(set(map(tuple, img[:, 300:].reshape(-1, 3)))) == 1


@pytest.mark.parametrize("poly", [CONE, CUP])
def test_a_mesh_prop_stands_on_the_table_with_its_own_size(tmp_path, poly):
    h = 0.1
    v, uv, f = L.build(L.profile(_mask(poly)), 0.07, 0.07, h)
    (tmp_path / "p.obj").write_text(L.to_obj(v, uv, f))
    cv2.imwrite(str(tmp_path / "p.png"), L.texture(np.full((20, 20, 3), 90, np.uint8), (0.4, 0.4, 0.4)))
    lay = Layout()
    lay.props = [{"name": "thing", "shape": "round", "pos": (0.1, 0.1), "yaw": 0.0, "size": (0.035, 0.035, h / 2), "rgb": (0.4, 0.4, 0.4),
                  "mesh": str(tmp_path / "p.obj"), "mesh_scale": (1.0, 1.0, 1.0), "mesh_texture": str(tmp_path / "p.png")}]
    w = World(lay)
    w.settle(80)
    assert w.obj_pos("prop_0")[2] == pytest.approx(h / 2, abs=0.01)                  # it rests on the table, upright


def _scene(shape_name, draw):
    img = np.full((540, 960, 3), (190, 205, 215), np.uint8)               # a plain table
    draw(img)
    res = E.analyse(img, 45.0)
    assert res["items"], "nothing found"
    for it in res["items"]:
        it["name"], it["shape"] = "thing", shape_name
    return res


@pytest.mark.parametrize("shape", ["round", "cylinder"])
def test_rounded_things_in_a_photo_become_meshes_and_boxes_stay_boxes(tmp_path, shape):
    res = _scene(shape, lambda img: cv2.ellipse(img, (480, 380), (70, 55), 0, 0, 360, (40, 40, 45), -1))
    props, _, _ = E.build(res, tmp_path)
    p = props[0]
    assert p["mesh"].endswith(".obj") and p["mesh_texture"].endswith("_tex.png") and p["skin"] is None
    assert (tmp_path / "prop_0.obj").exists() and (tmp_path / "prop_0_tex.png").exists()
    v = np.array([[float(x) for x in l.split()[1:]] for l in (tmp_path / "prop_0.obj").read_text().splitlines() if l.startswith("v ")])
    assert (v.max(0) - v.min(0)) / 2 == pytest.approx(np.array(p["size"]), rel=0.02)   # the mesh has the size the physics uses
    lay = Layout()
    lay.props = props
    World(lay).settle(10)
    boxy = _scene("box", lambda img: cv2.rectangle(img, (400, 330), (560, 440), (40, 40, 45), -1))
    pb, _, _ = E.build(boxy, tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path)
    assert "mesh" not in pb[0] and pb[0]["skin"].endswith("_cube.png")


def _bbox(m):
    ys, xs = np.nonzero(m)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _filled(draw, shape=(100, 100)):
    m = np.zeros(shape, np.uint8)
    draw(m)
    return _bbox(m.astype(bool))                                                              # the crop of its own box, as analyse gives it


def test_the_shape_is_guessed_from_the_outline_when_the_ai_has_not_said():
    ellipse = _filled(lambda m: cv2.ellipse(m, (50, 50), (45, 38), 0, 0, 360, 1, -1))
    rect = _filled(lambda m: cv2.rectangle(m, (5, 10), (94, 89), 1, -1))
    ragged = _filled(lambda m: [cv2.rectangle(m, (5, 5), (30, 30), 1, -1), cv2.rectangle(m, (60, 60), (94, 94), 1, -1)])   # a wrapper, a cable
    assert E.guess_shape(ellipse) == "round"
    assert E.guess_shape(rect) == "box" and E.guess_shape(ragged) == "box"
    assert E.guess_shape(np.ones((4, 4), bool)) == "box"                                    # too small to tell


def test_an_unnamed_round_thing_gets_its_shape_without_the_ai(tmp_path):
    img = np.full((540, 960, 3), (190, 205, 215), np.uint8)
    cv2.ellipse(img, (300, 380), (70, 55), 0, 0, 360, (40, 40, 45), -1)                    # a round thing
    cv2.rectangle(img, (600, 340), (760, 450), (40, 40, 45), -1)                           # a boxy thing
    res = E.analyse(img, 45.0)
    assert sorted(it["shape"] for it in res["items"]) == ["box", "round"]
    E.apply_names(res, None)                                                                  # no AI: names are "object 1", ...
    props, _, _ = E.build(res, tmp_path)
    assert sorted("mesh" in p for p in props) == [False, True]
