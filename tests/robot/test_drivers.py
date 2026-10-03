"""The real back-end on the mock arm: a perturbed second world behind the same contract."""
import numpy as np
import pytest

from echotwin.robot import backend as B
from echotwin.robot import drivers as D
from echotwin.robot import real as R
from echotwin.robot.features import prop_skills as PS
from echotwin.robot.scene import Layout
from echotwin.robot.world import World


def _layout(*props):
    lay = Layout()
    lay.props = [{"name": n, "shape": s, "pos": p, "yaw": 0.0, "size": z, "rgb": (0.8, 0.3, 0.3)} for n, s, p, z in props]
    return lay


CUBE = ("cube", "box", (0.0, -0.05), (0.025, 0.02, 0.025))


def _real(layout, **kw):
    twin = World(layout)
    drv = D.MockDriver(twin.layout, twin.arm, **kw)
    return R.RealBackend(twin, drv, observer=R.TruthObserver(drv.plant)), drv


def _move(r, name="prop_0", goal=(0.13, 0.0)):
    task = {"object": name, "name": name, "goal": np.array(goal), "h": r.twin.half(name)}
    run = {"wps": PS.waypoints(r, task, dict(PS.DEFAULTS)), "i": 0, "speed": PS.DEFAULTS["speed"], "yaw": r.grasp_yaw(name)}
    for _ in range(1500):
        a = PS.waypoint_action(r, run)
        if a is None:
            break
        r.step(a)
    for _ in range(40):
        r.step(np.array([0, 0, 0, 0.0, run["yaw"]]))
    return r.obj_pos(name)


def test_real_backend_is_a_complete_backend():
    r, _ = _real(_layout(CUBE))
    assert B.missing(r) == [] and all(hasattr(r, a) for a in ("hand", "layout", "twin", "name")) and r.name == "real"
    assert r.twin.name == "sim" and r.layout is r.twin.layout


def test_tool_position_comes_from_the_measured_joints():
    r, _ = _real(_layout(CUBE), encoder_offset=0.0, encoder_noise=0.0)
    for _ in range(40):
        r.step((0.05, 0.0, 0.0, 0, 0.0))
    assert np.linalg.norm(r.hand_pos() - r.twin.ik.pose(r.q_meas)[0]) < 1e-9
    assert np.linalg.norm(r.hand_pos() - r.twin.hand.target) < 0.02          # the arm has caught up with the target


@pytest.mark.parametrize("shape,half", [("box", (0.025, 0.02, 0.025)), ("cylinder", (0.03, 0.03, 0.04))])
def test_the_default_skill_moves_an_object_on_the_mock_arm(shape, half):
    r, _ = _real(_layout(("thing", shape, (0.0, -0.05), half), ("mark", "box", (0.2, 0.0), (0.02, 0.02, 0.02))))
    r.settle(20)
    end = _move(r)
    assert np.linalg.norm(end[:2] - np.array([0.13, 0.0])) < 0.05, end        # judged on the mock's own world


def test_the_gripper_settles_and_the_held_object_is_reported():
    r, _ = _real(_layout(CUBE))
    r.settle(20)
    tool = r.obj_pos("prop_0")
    for _ in range(120):
        d = np.array([tool[0], tool[1], 0.012]) - r.hand_pos()
        r.step((*np.clip(d * 5, -0.25, 0.25), 0, 0.0))
    for _ in range(25):
        r.step((0, 0, 0, 1, 0.0))
    assert r.grip_settled() and r.twin.hand.held == "prop_0" and r.closure < R.CLOSED_ALL_THE_WAY


def test_jaws_that_closed_on_nothing_clear_the_held_flag():
    r, _ = _real(_layout(CUBE))
    r.twin.hand.held = "prop_0"                       # the twin believes it holds the cube ...
    r.driver.closed = True
    r.closure = 0.99                                  # ... the jaws went all the way
    r.twin.hand.grip = True
    # one more tick with the jaws closed on nothing
    for _ in range(30):
        r.step((0, 0, 0, 1, 0.0))
    assert r.twin.hand.held is None


def test_stop_freezes_the_arm_and_resume_moves_again():
    r, _ = _real(_layout(CUBE))
    for _ in range(20):
        r.step((0.1, 0.0, 0.0, 0, 0.0))
    r.stop()
    p = r.hand_pos().copy()
    for _ in range(20):
        r.step((0.2, 0.0, 0.0, 0, 0.0))
    assert np.allclose(r.hand_pos(), p, atol=1e-6)
    r.resume()
    for _ in range(30):
        r.step((0.2, 0.0, 0.0, 0, 0.0))
    assert np.linalg.norm(r.hand_pos() - p) > 0.01


def test_imagination_runs_on_the_twin_not_on_the_arm():
    r, _ = _real(_layout(("thing", "box", (0.0, -0.05), (0.025, 0.02, 0.025)), ("mark", "box", (0.2, 0.0), (0.02, 0.02, 0.02))))
    before = r.hand_pos().copy()
    task = {"object": "prop_0", "name": "thing", "goal": np.array([0.13, 0.0]), "h": 0.025}
    res = PS.imagine(r, task, PS.waypoints(r, task, dict(PS.DEFAULTS)), 0.2)
    assert res["ok"] and np.allclose(r.hand_pos(), before)                  # the real arm did not move


def test_the_unarmed_backend_does_not_move_until_armed():
    twin = World(_layout(CUBE))
    r = R.RealBackend(twin, D.MockDriver(twin.layout, twin.arm), armed=False)
    p = r.hand_pos().copy()
    for _ in range(10):
        r.step((0.2, 0.0, 0.0, 0, 0.0))
    assert not r.arm_ready() and np.allclose(r.hand_pos(), p)
    r.enable()
    for _ in range(20):
        r.step((0.2, 0.0, 0.0, 0, 0.0))
    assert np.linalg.norm(r.hand_pos() - p) > 0.01


def test_a_new_layout_rebuilds_the_mock_world():
    r, drv = _real(_layout(CUBE))
    r.build(_layout(("a", "box", (0.1, 0.0), (0.02, 0.02, 0.02)), ("b", "box", (-0.1, 0.0), (0.02, 0.02, 0.02))))
    assert len(drv.plant.things()) == 2 and len(r.things()) == 2
