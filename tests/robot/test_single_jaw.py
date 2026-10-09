"""A single moving jaw holds things off-centre (the SO-ARM100): the carry steers the thing, not the tool point. And a step
above the arm's ceiling is reached at the ceiling instead of waiting out the tick limit."""
import numpy as np
import pytest

from echotwin.robot import arm as A
from echotwin.robot import skillcheck as S
from echotwin.robot.features import prop_skills as PS
from echotwin.robot.scene import Layout
from echotwin.robot.world import World

so100 = pytest.mark.skipif(A.load("so_arm100").missing_files() != [], reason="so_arm100 is not downloaded")


def _move(arm, shape="box"):
    lay = Layout()
    lay.props = [S._prop("mover", shape, S.SHAPES[shape], (0.0, 0.0), 1.0), S._prop("other", "box", S.OTHER, (0.15, 0.08), 1.0)]
    w = World(lay, arm)
    w.settle(20)
    task = PS.make_task(w, {"prop": 0, "goal": ("dir", (-1, 0), 0.12)}, "put it to the left")
    wps = PS.waypoints(w, task, dict(PS.DEFAULTS))
    r = {"wps": wps, "i": 0, "speed": PS.DEFAULTS["speed"], "yaw": w.grasp_yaw("prop_0")}
    for ticks in range(1500):
        a = PS.waypoint_action(w, r)
        if a is None:
            break
        w.step(a)
    w.settle(30)
    return w, task, ticks


def test_a_parallel_gripper_holds_centred_so_nothing_is_corrected():
    w = World(Layout())
    assert (PS._held_off(w, None, w.hand) == 0).all()


@so100
@pytest.mark.parametrize("shape", ["box", "round"])
def test_the_so_arm100_puts_the_thing_on_the_spot_and_does_not_wait_at_its_ceiling(shape):
    w, task, ticks = _move("so_arm100", shape)
    err = float(np.linalg.norm(w.obj_pos("prop_0")[:2] - np.array(task["goal"])))
    assert err < 0.01                                       # was 1.3 cm: the thing swung around the tool point
    assert ticks < 250                                      # was about 360: every high step waited out 80 ticks


def test_the_builtin_arm_still_moves_the_box_cleanly():
    w, task, ticks = _move("builtin")
    assert float(np.linalg.norm(w.obj_pos("prop_0")[:2] - np.array(task["goal"]))) < 0.005 and ticks < 150
