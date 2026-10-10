"""A thing nearly as tall as the arm reaches: the hand comes in from the side instead of sweeping its top."""
import numpy as np
import pytest

from echotwin.robot import arm as A
from echotwin.robot import skillcheck as S
from echotwin.robot.features import prop_skills as PS
from echotwin.robot.scene import Layout
from echotwin.robot.world import World

so100 = pytest.mark.skipif(A.load("so_arm100").missing_files() != [], reason="so_arm100 is not downloaded")


def _world(size, at, arm=None):
    lay = Layout()
    lay.props = [S._prop("mover", "cylinder", size, at, 1.0)]
    w = World(lay, arm)
    w.settle(20)
    return w


def test_the_jaws_hang_below_the_tool_point():
    assert 0.0 <= World(Layout()).jaw_drop() < 0.01


@so100
def test_a_tall_glass_is_approached_from_the_side_along_the_jaws():
    w = _world((0.034, 0.034, 0.08), (0.055, -0.027), A.load("so_arm100"))
    assert w.jaw_drop() > 0.01
    task = PS.make_task(w, {"prop": 0, "goal": ("dir", (-1, 0), 0.12)}, "left")
    wps = PS.waypoints(w, task, dict(PS.DEFAULTS))
    moves = [a for k, a in wps if k == "move"]
    side, low, at = moves[0], moves[1], moves[2]                # above the side spot, down beside it, slide in
    assert side[:2] == low[:2] and low[2] == at[2] < 0.05
    slide = np.asarray(at[:2]) - np.asarray(side[:2])
    y = w.grasp_yaw("prop_0")
    assert abs(slide @ np.array([np.cos(y), np.sin(y)])) < 1e-6   # along the jaws' plane, not into a jaw
    assert len(wps) == 9


def test_a_low_thing_is_approached_from_above_as_before():
    w = _world((0.05, 0.05, 0.04), (0.0, 0.0))
    task = PS.make_task(w, {"prop": 0, "goal": ("dir", (-1, 0), 0.12)}, "left")
    assert len(PS.waypoints(w, task, dict(PS.DEFAULTS))) == 8
