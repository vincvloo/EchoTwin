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
    assert err < 0.02                                       # it swung around the tool point (1.3 cm, plus the release);
                                                            # the 8 mm fixed-jaw gap costs ~1 cm at release, for safer grips
    assert ticks < 250                                      # was about 360: every high step waited out 80 ticks


def test_the_builtin_arm_still_moves_the_box_cleanly():
    w, task, ticks = _move("builtin")
    assert float(np.linalg.norm(w.obj_pos("prop_0")[:2] - np.array(task["goal"]))) < 0.005 and ticks < 150


@so100
def test_the_fixed_jaw_stands_on_the_side_the_arm_can_really_hold_it():
    """Closing along yaw or yaw + pi is the same line, but a single jaw has a side: at (0.2, -0.16) the wrist cannot turn
    to the side the plain yaw asks for, so the grasp turns round and the fixed jaw goes on the other side of the box."""
    lay = Layout()
    lay.props = [S._prop("mover", "box", S.SHAPES["box"], (0.2, -0.16), 1.0)]
    w = World(lay, "so_arm100")
    yaw = w.grasp_yaw("prop_0")
    q, ep, _ = w.ik.solve(w.obj_pos("prop_0") + np.array([0, 0, 0.03]), yaw, w.ik.q_down, iters=150,
                          tilt=w.workspace.tilt_for(w._rel(w.obj_pos("prop_0"), w.hand.base), 0.05))
    jaws = w.ik.pose(q)[1] @ w.ik.c_axis
    assert jaws @ np.array([np.cos(yaw), np.sin(yaw), 0.0]) > 0.9              # the arm holds its jaws the way the plan says
    side = np.sign(w.grasp_offset("prop_0")[1])
    assert side == -1                                                            # here: the fixed jaw on the -y side


def test_a_parallel_gripper_keeps_the_plain_grasp_direction():
    lay = Layout()
    lay.props = [S._prop("mover", "box", S.SHAPES["box"], (0.2, -0.16), 1.0)]
    w = World(lay)
    assert w.grasp_yaw("prop_0") == pytest.approx(np.pi / 2) and (w.grasp_offset("prop_0") == 0).all()
