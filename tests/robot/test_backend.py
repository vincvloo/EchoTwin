"""The robot contract and the sim back-end."""
import numpy as np

from echotwin.robot import backend as B
from echotwin.robot.scene import Layout
from echotwin.robot.world import World


def _world():
    lay = Layout()
    lay.props = [{"name": "cube", "shape": "box", "pos": (0.0, -0.05), "yaw": 0.0, "size": (0.025, 0.02, 0.025), "rgb": (0.8, 0.3, 0.3)}]
    return World(lay)


def test_world_is_a_complete_backend():
    w = _world()
    assert B.missing(w) == [] and isinstance(w, B.Backend)
    assert w.name == "sim" and w.twin is w and w.arm_ready()


def test_command_and_advance_are_what_step_does():
    a, b = _world(), _world()
    for _ in range(10):
        a.step((0.1, 0.0, -0.05, 0, 0.0))
        b.command((0.1, 0.0, -0.05, 0, 0.0))
        b.advance(b.data, b.hand)
    assert np.allclose(a.hand_pos(), b.hand_pos())


def test_tilt_and_carry_height():
    w = _world()
    w.settle(20)
    assert w.tilt("prop_0") < 2.0
    assert 0.1 < w.carry_height() < 0.25
