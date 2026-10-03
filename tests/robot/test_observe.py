"""Looking at the table through a camera: where the robot thinks an object is, and moving that belief to where it is seen."""
import numpy as np
import pytest

from echotwin.robot import drivers as D
from echotwin.robot import real as R
from echotwin.robot.scene import Layout
from echotwin.robot.world import World

PROPS = [
    ("cube", "box", (-0.12, 0.00), (0.025, 0.02, 0.025), (0.8, 0.2, 0.2)),
    ("glass", "cylinder", (0.12, 0.05), (0.035, 0.035, 0.05), (0.2, 0.7, 0.3)),
    ("ball", "round", (-0.02, 0.12), (0.025, 0.025, 0.025), (0.2, 0.3, 0.9)),
    ("bar", "flat", (0.05, -0.03), (0.05, 0.03, 0.01), (0.9, 0.8, 0.1)),
]


def _layout(props=PROPS):
    lay = Layout()
    lay.props = [{"name": n, "shape": s, "pos": p, "yaw": 0.0, "size": z, "rgb": c} for n, s, p, z, c in props]
    return lay


def _mock(props=PROPS):
    twin = World(_layout(props))
    drv = D.MockDriver(twin.layout, twin.arm)
    return R.RealBackend(twin, drv, camera=R.mock_camera(drv)), drv


def _park(r):
    """Move the arm to where it looks at the table from."""
    target = r.twin.observe_pose()
    for _ in range(160):
        d = target - r.hand_pos()
        r.step((*np.clip(d * 5, -0.25, 0.25), 0, 0.0))
    return r


def test_the_camera_sees_every_object_within_1_5_cm():
    r, drv = _mock()
    _park(r)
    for i, (n, *_rest) in enumerate(PROPS):
        name = f"prop_{i}"
        seen = r.observe(name)
        truth = drv.plant.obj_pos(name)[:2]
        assert seen is not None, n
        assert np.linalg.norm(seen - truth) < 0.015, (n, seen, truth)


def test_a_moved_object_is_found_and_the_belief_follows():
    r, drv = _mock()
    _park(r)
    drv.plant.nudge("prop_1", (-0.05, 0.04))                         # somebody moves the glass; the twin does not know
    assert np.linalg.norm(r.twin.obj_pos("prop_1")[:2] - drv.plant.obj_pos("prop_1")[:2]) > 0.06
    seen = r.observe("prop_1")
    assert np.linalg.norm(seen - drv.plant.obj_pos("prop_1")[:2]) < 0.015
    assert np.linalg.norm(r.twin.obj_pos("prop_1")[:2] - seen) < 1e-6          # the twin now believes what the camera saw


def test_an_object_that_is_gone_is_not_seen():
    r, drv = _mock()
    _park(r)
    drv.plant.set_obj_pose("prop_0", (0.3, 0.25))                                # off to a corner, far from where it was expected
    assert r.observe("prop_0") is None


def test_set_obj_pose_moves_one_object_and_keeps_its_height_and_the_others():
    w = World(_layout())
    w.settle(20)
    z, other = w.obj_pos("prop_0")[2], w.obj_pos("prop_1").copy()
    w.set_obj_pose("prop_0", (0.2, 0.1))
    assert w.obj_pos("prop_0")[:2] == pytest.approx((0.2, 0.1)) and w.obj_pos("prop_0")[2] == pytest.approx(z)
    assert np.allclose(w.obj_pos("prop_1"), other)


def test_the_simulation_is_its_own_truth():
    w = World(_layout())
    assert np.allclose(w.observe("prop_2"), w.obj_pos("prop_2")[:2])
    assert w.observe_pose()[2] < 0.1 and abs(w.observe_pose()[0] - w.base[0]) < 1e-9


def test_the_arm_parked_to_look_does_not_hide_the_reachable_table():
    r, drv = _mock()
    _park(r)
    seen = [r.observe(f"prop_{i}") for i in range(len(PROPS))]
    assert all(s is not None for s in seen)


def test_without_a_camera_the_real_backend_can_only_believe():
    twin = World(_layout())
    r = R.RealBackend(twin, D.MockDriver(twin.layout, twin.arm))
    assert np.allclose(r.observe("prop_0"), twin.obj_pos("prop_0")[:2])
