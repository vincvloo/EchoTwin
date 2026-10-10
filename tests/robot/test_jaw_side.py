"""A single moving jaw: which side the fixed jaw takes, and when the arm cannot get its jaws around a thing."""
import numpy as np
import pytest

from echotwin.robot import arm as A
from echotwin.robot import skillcheck as S
from echotwin.robot.features import prop_skills as PS
from echotwin.robot.scene import Layout
from echotwin.robot.world import World

pytestmark = pytest.mark.skipif(A.load("so_arm100").missing_files() != [], reason="so_arm100 is not downloaded")


def _world(shape, at):
    lay = Layout()
    lay.props = [S._prop("mover", shape, S.SHAPES[shape], at, 1.0)]
    w = World(lay, A.load("so_arm100"))
    w.settle(20)
    return w


def _lined_up(w, yaw):
    """How well the arm lines its jaws up with `yaw` where the tool stands beside prop_0, low down (cos of the twist)."""
    at = w.obj_pos("prop_0") + w.grasp_offset("prop_0", yaw)
    at[2] = 0.02
    q, _, _ = w.ik.solve(at, yaw, w.ik.q_down, iters=150)
    return float(w.ik.pose(q)[1] @ w.ik.c_axis @ np.array([np.cos(yaw), np.sin(yaw), 0.0]))


def test_where_the_wrist_falls_short_one_way_round_it_goes_the_other_way():
    w = _world("flat", (0.12, -0.08))                       # right of the base: turning the jaws to +90 degrees falls short
    assert _lined_up(w, np.pi / 2) < np.cos(np.radians(30))
    yaw = w.grasp_yaw("prop_0")
    assert np.isclose(np.cos(yaw), 0.0, atol=1e-6) and _lined_up(w, yaw) > np.cos(np.radians(10))   # still across it


def test_it_keeps_the_way_asked_when_that_works():
    w = _world("box", (-0.15, -0.12))
    assert [round(float(np.degrees(y))) for _, y, ok in w._jaw_fits("prop_0", np.pi / 2) if ok] == [90, 270]
    assert w.grasp_yaw("prop_0") == pytest.approx(np.pi / 2)


def test_it_declines_a_flat_thing_right_in_front_of_its_base():
    w = _world("flat", (-0.04, -0.11))                      # the base side is too close; the far side, the open jaw hits the arm
    assert w.reachable(w.obj_pos("prop_0")[:2])
    assert w.refusal("prop_0") == "I can't get my jaws around it from here"
    assert w.refusal("prop_0") == "I can't get my jaws around it from here"     # cached, same answer


def test_the_open_jaw_hitting_the_arm_is_seen():
    w = _world("flat", (-0.04, -0.11))
    assert not w._hits_itself(w.ik.q_down)                  # pointing down in the middle of its reach: clear
    folded = w.ik.solve(np.array([-0.04, -0.072, 0.01]), np.pi / 2, w.ik.q_down, iters=150)[0]
    assert w._hits_itself(folded)


def test_while_held_the_offset_follows_the_jaws_not_a_new_choice():
    w = _world("box", (0.0, -0.05))
    a = w.grasp_offset("prop_0", 0.3)
    assert np.hypot(*a[:2]) == pytest.approx(w.grasp_width("prop_0") / 2 + 0.008)
    assert np.arctan2(a[1], a[0]) == pytest.approx(0.3)
    hs = w.hand.clone()
    hs.held, hs.yaw = "prop_0", 0.3
    off = PS._held_off(w, None, hs)
    seen = w.obj_pos("prop_0")[:2] - w.hand_pos()[:2]
    assert off == pytest.approx(seen + a[:2])


def test_a_parallel_gripper_is_not_affected():
    lay = Layout()
    lay.props = [S._prop("mover", "flat", S.SHAPES["flat"], (-0.04, -0.11), 1.0)]
    w = World(lay)
    assert w.grasp_offset("prop_0", 1.0).tolist() == [0.0, 0.0, 0.0] and "jaws around" not in w.refusal("prop_0")


def test_leaning_out_its_hand_does_not_hang_over_the_thing():
    """Far out the arm leans; standing on the far side of a thing would put its hand over the thing and press it down."""
    w = _world("box", (0.0, 0.10))                          # 33 cm in front of the base: it leans there
    off = w.grasp_offset("prop_0")
    assert w.workspace.tilt_for(w._rel(w.obj_pos("prop_0")[:2] + off[:2], w.hand.base), 0.01) > 0
    assert off[1] < 0                                        # the tool on the base side, the thing further out
    fits = w._jaw_fits("prop_0", w._jaw_free_yaw("prop_0"))
    assert [ok for _, y, ok in fits if np.sin(y) > 0] == [False]


def test_a_round_outline_is_gripped_from_another_direction_when_one_fails():
    w = _world("round", (0.25, 0.0))
    assert w.refusal("prop_0") == ""
    assert any(ok for _, _, ok in w._jaw_fits("prop_0", w.grasp_yaw("prop_0")))


def test_the_arm_against_the_table_counts_its_jaws_resting_on_it_does_not():
    w = _world("box", (0.0, -0.05))
    at = w.obj_pos("prop_0") + w.grasp_offset("prop_0")
    at[2] = 0.004
    q = w.ik.solve(at, w.grasp_yaw("prop_0"), w.ik.q_down, iters=150)[0]
    assert not w._collides(q)                               # gripping low: the jaws sit on the table
    sunk = w.ik.solve(at - np.array([0.0, 0.0, 0.12]), w.grasp_yaw("prop_0"), w.ik.q_down, iters=200)[0]
    assert w._collides(sunk)                                # 12 cm into the table: past the jaws, the wrist is in it
