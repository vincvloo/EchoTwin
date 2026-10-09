"""Stacking: the carried object must clear the top it is set on, under the arm's ceiling along the way."""
import numpy as np
import pytest

from echotwin.robot import arm as A
from echotwin.robot import skillcheck as S
from echotwin.robot.features import prop_skills as PS
from echotwin.robot.scene import Layout
from echotwin.robot.world import STACK_CLEAR, World


def _world(shape, at, ref_at, size=None, ref_size=S.OTHER, arm=None):
    lay = Layout()
    lay.props = [S._prop("mover", shape, size or S.SHAPES[shape], at, 1.0), S._prop("other", "box", ref_size, ref_at, 1.0)]
    w = World(lay, arm)
    w.settle(20)
    return w


def _stack(w):
    return PS.make_task(w, {"prop": 0, "goal": ("near", 1, "on top of")}, "put it on top")


def test_the_palm_is_how_deep_the_tool_goes_over_an_objects_top():
    w = _world("box", (0.0, 0.0), (0.12, 0.06))
    assert w.palm("prop_0") == pytest.approx(0.054, abs=0.003)          # the built-in hand sits 5.4 cm above the pad tips
    assert w.stack_hang("prop_0") == pytest.approx(0.004)                # a 4 cm box fits under it: held near its bottom
    tall = _world("cylinder", (0.0, 0.0), (0.12, 0.06))
    assert tall.stack_hang("prop_0") == pytest.approx(0.10 - tall.palm("prop_0"), abs=1e-6)   # a 10 cm one sticks up into the hand


def test_the_path_ceiling_is_never_above_the_ceiling_at_either_end():
    w = _world("box", (0.0, 0.0), (0.12, 0.06))
    a, b = np.array([0.0, 0.0]), np.array([0.12, 0.06])
    c = w.path_ceiling(a, b)
    assert c <= w.ceiling(a) + 1e-9 and c <= w.ceiling(b) + 1e-9 and c <= w.carry_height() + 1e-9


def test_a_stack_is_carried_under_the_ceiling_and_over_the_other_top():
    w = _world("round", (0.0, 0.0), (0.12, 0.06))
    task = _stack(w)
    wps = PS.waypoints(w, task, dict(PS.DEFAULTS))
    moves = [a for k, a in wps if k == "move"]
    grasp_z, carry = moves[1][2], moves[2][2]
    assert carry <= w.path_ceiling(moves[0][:2], moves[3][:2], "prop_0") + 1e-9
    top = w.obj_pos("prop_1")[2] + w.half("prop_1")
    assert carry - w.stack_hang("prop_0", grasp_z - (w.obj_pos("prop_0")[2] - w.half("prop_0"))) - top >= STACK_CLEAR - 1e-9


def test_other_moves_plan_as_before():
    w = _world("box", (0.0, 0.0), (0.12, 0.06))
    task = PS.make_task(w, {"prop": 0, "goal": ("near", 1, "next to")}, "put it next to")
    sk = dict(PS.DEFAULTS)
    carry = min(w.tallest() + w.half("prop_0") + sk["lift"], w.carry_height())
    assert [a[2] for k, a in PS.waypoints(w, task, sk) if k == "move"][0] == pytest.approx(carry)


def test_it_says_so_when_it_cannot_lift_high_enough():
    w = _world("cylinder", (0.0, 0.0), (0.12, 0.06), ref_size=(0.05, 0.05, 0.12))     # onto a 24 cm tall box
    goal = _stack(w)["goal"]
    assert w.refusal("prop_0", goal) == ""                                          # moving it there is fine
    assert "high enough over the other" in w.refusal("prop_0", goal, on="prop_1")


def test_the_built_in_arm_stacks_a_ball_on_a_box():
    w = _world("round", (0.0, 0.0), (0.12, 0.06))
    task = _stack(w)
    r = {"wps": PS.waypoints(w, task, dict(PS.DEFAULTS)), "i": 0, "speed": PS.DEFAULTS["speed"], "yaw": w.grasp_yaw("prop_0")}
    for _ in range(1500):
        a = PS.waypoint_action(w, r)
        if a is None:
            break
        w.step(a)
    w.settle(40)
    assert PS.goal_met(w, task)


@pytest.mark.skipif(A.load("so_arm100").missing_files() != [], reason="so_arm100 is not downloaded")
def test_the_so_arm100s_palm_is_measured_with_its_jaws_open():
    w = _world("box", (0.0, 0.0), (0.12, 0.06), arm=A.load("so_arm100"))
    assert 0.03 < w.palm("prop_0") < 0.09                                 # not the closed moving jaw, nor a visual-only mesh


def test_the_path_ceiling_is_looked_at_every_centimetre():
    w = _world("box", (0.0, 0.0), (0.12, 0.06))
    a, b = np.array([-0.2, -0.1]), np.array([0.25, -0.15])
    every_cm = min(w.ceiling(a + (b - a) * f) for f in np.linspace(0.0, 1.0, 200))
    assert w.path_ceiling(a, b) == pytest.approx(every_cm)


@pytest.mark.skipif(A.load("so_arm100").missing_files() != [], reason="so_arm100 is not downloaded")
def test_a_dip_in_the_ceiling_on_the_way_is_not_missed():
    """Far out to the right the SO-ARM100's ceiling dips on the way to the box: carrying over it, the ball hit the box."""
    w = _world("round", (-0.134, -0.022), (0.275, -0.15), arm=A.load("so_arm100"))
    assert "high enough" in w.refusal("prop_0", _stack(w)["goal"], on="prop_1")


def test_skillcheck_reports_a_layout_the_arm_declines_as_refused(monkeypatch):
    monkeypatch.setattr(World, "refusal", lambda self, name, goal_xy=None, on=None: "it is out of my reach")
    r = S.trial("box", "next to", np.random.default_rng(0))
    assert not r["ok"] and r["why"] == "refused: it is out of my reach" and S.reason_key(r["why"]) == "refused"
