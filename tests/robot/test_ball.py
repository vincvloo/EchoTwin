"""Balls: a ball has no up, so rolling on the spot is not tipping over."""
import numpy as np
import pytest

from echotwin.robot import skillcheck as S
from echotwin.robot.features import prop_skills as PS
from echotwin.robot.scene import Layout
from echotwin.robot.world import World


def _world(shape, at, size=None):
    lay = Layout()
    lay.props = [S._prop("mover", shape, size or S.SHAPES[shape], at, 1.0)]
    w = World(lay)
    w.settle(20)
    return w


def test_a_ball_has_no_up_an_egg_and_a_box_do():
    assert not _world("round", (0.0, 0.0)).has_up("prop_0")
    assert _world("round", (0.0, 0.0), size=(0.05, 0.05, 0.08)).has_up("prop_0")      # an egg standing on end
    assert _world("box", (0.0, 0.0), size=(0.05, 0.05, 0.05)).has_up("prop_0")       # a cube is not round


def test_a_ball_that_rolled_on_the_spot_has_not_tipped_over():
    w = _world("round", (0.0, 0.0))
    task = PS.make_task(w, {"prop": 0, "goal": ("dir", (-1, 0), 0.0)}, "leave it")
    a = w.obj_qadr["prop_0"]
    w.data.qpos[a + 3:a + 7] = [np.cos(np.radians(30)), np.sin(np.radians(30)), 0.0, 0.0]   # rolled 60 degrees
    assert w.tilt("prop_0") == pytest.approx(60.0, abs=0.5)
    res = PS.outcome(w, task, {"prop_0": w.obj_pos("prop_0")[:2]})
    assert res["ok"] and res["tilt"] == 0.0
