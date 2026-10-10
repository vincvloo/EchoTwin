"""The robot reads an object's shape from the geometry the twin built, not from its shape label."""
import numpy as np
import pytest

from echotwin.robot.features import geometry as GE
from echotwin.robot.scene import DENSITY, Layout
from echotwin.robot.world import World

trimesh = pytest.importorskip("trimesh")


def _prop(name, shape, half, pos=(0.0, 0.0), yaw=0.0, **extra):
    return {"name": name, "shape": shape, "pos": pos, "yaw": yaw, "size": half, "rgb": (0.5, 0.5, 0.5), **extra}


def _world(*props):
    lay = Layout()
    lay.props = list(props)
    w = World(lay)
    w.settle(10)
    return w


def _obj(tmp_path, mesh, name):
    path = tmp_path / f"{name}.obj"
    mesh.export(path)
    return str(path)


def test_a_turned_box_is_gripped_across_its_narrow_side_whatever_its_label():
    w = _world(_prop("bar", "box", (0.05, 0.02, 0.02), yaw=30.0))
    assert w.grasp_width("prop_0") == pytest.approx(0.04, abs=1e-6)
    assert np.degrees(w.grasp_yaw("prop_0")) % 180 == pytest.approx(120.0, abs=0.5)


def test_round_outlines_keep_the_old_directions():
    w = _world(_prop("glass", "cylinder", (0.03, 0.03, 0.05)), _prop("ball", "round", (0.025, 0.025, 0.025), pos=(0.15, 0.0)))
    assert w.grasp_width("prop_0") == pytest.approx(0.06, abs=1e-6) and w.grasp_yaw("prop_0") == 0.0
    assert w.grasp_width("prop_1") == pytest.approx(0.05, abs=2e-4) and w.grasp_yaw("prop_1") == 0.0     # settled: a hair turned


def test_a_scanned_ball_has_no_up_and_a_scanned_bottle_does(tmp_path):
    ball = _obj(tmp_path, trimesh.creation.icosphere(subdivisions=3, radius=0.03), "ball")
    bottle = _obj(tmp_path, trimesh.creation.cylinder(radius=0.03, height=0.16, sections=32), "bottle")
    w = _world(_prop("orange", "box", (0.03, 0.03, 0.03), mesh=ball, mesh_scale=(1, 1, 1)),        # labelled a box
               _prop("bottle", "box", (0.03, 0.03, 0.08), pos=(0.15, 0.0), mesh=bottle, mesh_scale=(1, 1, 1)))
    assert not w.has_up("prop_0") and w.has_up("prop_1")
    assert w.grasp_width("prop_1") == pytest.approx(0.06, abs=0.002)


def test_a_mesh_weighs_what_its_volume_does(tmp_path):
    ball = _obj(tmp_path, trimesh.creation.icosphere(subdivisions=3, radius=0.03), "ball")
    w = _world(_prop("orange", "box", (0.03, 0.03, 0.03), mesh=ball, mesh_scale=(1, 1, 1)))
    sphere = 4 / 3 * np.pi * 0.03 ** 3
    assert w.mass("prop_0") == pytest.approx(np.clip(sphere * DENSITY, 0.02, 0.4), rel=0.03)   # not its 6 cm box's


def test_the_grip_width_is_taken_low_where_the_jaws_close():
    """A thing that is wide only at the top (a cone standing on its point would be gripped low, where it is narrow)."""
    cone = np.array([[0, 0, -0.05]] + [[0.04 * np.cos(a), 0.04 * np.sin(a), z]
                                       for z in (0.0, 0.05) for a in np.linspace(0, 2 * np.pi, 40, endpoint=False)])
    s = GE.Shape(cone)
    w, _ = s.grip(np.eye(3))
    assert w < 0.08 and s.has_up


def test_an_irregular_outline_is_gripped_across_its_narrowest_width():
    pts = np.array([[0.06, 0.0], [0.0, 0.02], [-0.05, 0.01], [-0.05, -0.01], [0.0, -0.02]])        # a long flat leaf shape
    s = GE.Shape(np.column_stack([np.vstack([pts, pts]), np.repeat([-0.005, 0.005], len(pts))]))
    w, across = s.grip(np.eye(3))
    assert w == pytest.approx(0.04, abs=1e-3) and abs(np.sin(across)) > 0.95                  # across the leaf, not along
