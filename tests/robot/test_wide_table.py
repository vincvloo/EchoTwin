"""The table is never smaller than a normal table: a photo shows a patch of it, the rest is plain surface."""
import numpy as np
import pytest

from echotwin.robot.scene import DEFAULT_TABLE_HALF, SOLID_TABLE_MIN, Layout, build_xml
from echotwin.robot.world import World

CUBE = {"name": "cube", "shape": "box", "pos": (0.0, 0.0), "yaw": 0.0, "size": (0.02, 0.02, 0.02), "rgb": (0.8, 0.3, 0.3)}


def _geom_size(model, name):
    import mujoco
    return model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)]


def test_a_small_patch_sits_on_a_full_size_table():
    lay = Layout(table_half=(0.12, 0.09))
    w = World(lay)
    hx, hy, _ = _geom_size(w.model, "table")
    assert hx == pytest.approx(SOLID_TABLE_MIN[0] + 0.02) and hy == pytest.approx(SOLID_TABLE_MIN[1] + 0.02)
    assert _geom_size(w.model, "tabletop")[:2] == pytest.approx([0.12, 0.09])         # the photo covers its patch only
    assert "tablesurround" in build_xml(lay)
    assert w.layout.table_half == (0.12, 0.09)                                          # goals and the arm still use the patch


def test_the_default_table_and_a_bigger_one_are_unchanged():
    assert DEFAULT_TABLE_HALF == SOLID_TABLE_MIN
    assert "tablesurround" not in build_xml(Layout())
    big = Layout(table_half=(0.6, 0.45))
    w = World(big)
    assert _geom_size(w.model, "table")[:2] == pytest.approx([0.62, 0.47]) and "tablesurround" not in build_xml(big)
    wide_only = Layout(table_half=(0.7, 0.1))                                           # each side on its own
    assert _geom_size(World(wide_only).model, "table")[:2] == pytest.approx([0.72, SOLID_TABLE_MIN[1] + 0.02])


def test_something_outside_the_patch_rests_on_the_table_and_does_not_fall():
    lay = Layout(table_half=(0.12, 0.09))
    lay.props = [{**CUBE, "pos": (0.3, 0.2)}]                                           # well outside the patch
    w = World(lay)
    w.settle(60)
    z = w.obj_pos("prop_0")[2]
    assert z == pytest.approx(0.02, abs=0.005)


def test_the_surround_takes_the_colour_of_the_texture_edge(tmp_path):
    import cv2
    img = np.zeros((100, 140, 3), np.uint8)
    img[:] = (40, 80, 200)                                                              # BGR: an orange-red table
    cv2.imwrite(str(tmp_path / "t.png"), img)
    xml = build_xml(Layout(texture=str(tmp_path / "t.png"), table_half=(0.12, 0.09)))
    assert 'rgba="0.784 0.314 0.157 1"' in xml
    wood = build_xml(Layout(table_half=(0.12, 0.09)))                                  # no texture: wood
    assert 'name="tablesurround"' in wood and 'size="0.4000 0.3000 0.0100" rgba="0.62 0.50 0.38 1"' in wood
