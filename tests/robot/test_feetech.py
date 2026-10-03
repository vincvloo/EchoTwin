"""The servo driver, against a fake bus. This checks the calibration maths and the command flow, not any hardware."""
import copy
import json

import numpy as np
import pytest

from echotwin.robot import arm as A
from echotwin.robot import backend as B
from echotwin.robot import drivers as D
from echotwin.robot import feetech as F


class FakeBus:
    """Servos that jump to their target at once, and remember what they were told."""

    def __init__(self):
        self.pos, self.torque_on, self.writes = {}, {}, []

    def write_pos(self, i, ticks, speed, acc):
        self.pos[i] = ticks
        self.writes.append((i, ticks, speed, acc))

    def read_pos(self, i):
        return self.pos.get(i, 2048)

    def torque(self, i, on):
        self.torque_on[i] = on

    def close(self):
        self.closed = True


def _arm(calibrated=True, **real):
    d = json.loads((A.ARMS_DIR / "so_arm100.json").read_text())
    d["real"] = {**d["real"], "calibrated": calibrated, **real}
    return A.parse(d)


def test_ticks_and_radians_round_trip_and_respect_directions():
    drv = F.FeetechDriver(_arm(direction=[1, -1, 1, -1, 1], zero_ticks=[2048, 2000, 2048, 2100, 2048]), "COMX", FakeBus())
    q = np.array([0.3, -0.5, 0.8, 0.4, -0.2])
    t = drv.to_ticks(q)
    assert t[0] == round(2048 + 0.3 * 651.9) and t[1] == round(2000 + 0.5 * 651.9)       # direction -1 flips the sign
    assert np.allclose(drv.to_rad(t), q, atol=1 / 651.9)


def test_ticks_never_leave_the_servo_range_and_joints_stay_inside_the_sims_limits():
    drv = F.FeetechDriver(_arm(), "COMX", FakeBus())
    assert drv.to_ticks([100.0] * 5).max() <= 4095 and drv.to_ticks([-100.0] * 5).min() >= 0
    assert np.all(drv.to_rad(drv.to_ticks([100.0] * 5)) <= drv.hi + 1.0 / 651.9)


def test_command_writes_every_servo_and_the_gripper():
    bus = FakeBus()
    drv = F.FeetechDriver(_arm(), "COMX", bus)
    drv.connect(np.zeros(5))
    drv.command(np.zeros(5), True)
    assert sorted(i for i, *_ in bus.writes) == [1, 2, 3, 4, 5, 6]
    assert bus.pos[6] == 1500 and all(bus.torque_on.values())
    q, closure = drv.read()
    assert closure == pytest.approx(1.0) and np.allclose(q, 0.0, atol=2e-3)
    drv.command(np.zeros(5), False)
    assert drv.read()[1] == pytest.approx(0.0)


def test_stop_switches_the_torque_off():
    bus = FakeBus()
    drv = F.FeetechDriver(_arm(), "COMX", bus)
    drv.connect(np.zeros(5))
    drv.stop()
    assert not any(bus.torque_on.values())
    drv.start()
    assert all(bus.torque_on.values())


def test_an_uncalibrated_or_unknown_arm_refuses_to_move():
    with pytest.raises(D.DriverError, match="not calibrated"):
        F.FeetechDriver(_arm(calibrated=False), "COMX", FakeBus())
    with pytest.raises(D.DriverError, match="no 'real' block"):
        F.FeetechDriver(A.load("builtin"), "COMX", FakeBus())


def test_the_shipped_descriptor_is_a_placeholder_that_cannot_move():
    assert A.load("so_arm100").real["calibrated"] is False


def test_the_real_backend_drives_the_fake_bus_through_the_same_contract():
    from echotwin.robot.scene import Layout
    from echotwin.robot.world import World
    lay = Layout()
    lay.props = [{"name": "cube", "shape": "box", "pos": (0.0, 0.0), "yaw": 0.0, "size": (0.025, 0.02, 0.025), "rgb": (0.8, 0.3, 0.3)}]
    arm = _arm()
    twin = World(lay, arm)
    # the fake bus starts at the servo reading of the twin's rest pose
    bus = FakeBus()
    drv = F.FeetechDriver(arm, "COMX", bus)
    for i, t in zip(drv.ids, drv.to_ticks(twin.hand.q)):
        bus.pos[i] = int(t)
    from echotwin.robot.real import RealBackend
    rb = RealBackend(twin, _NoWait(drv))
    before = rb.hand_pos().copy()
    for _ in range(30):
        rb.step((0.1, 0.0, 0.0, 0, 0.0))
    assert np.linalg.norm(rb.hand_pos() - before) > 0.005 and rb.name == "real"
    rb.stop()
    assert not any(bus.torque_on.values())


class _NoWait:
    """The driver, without sleeping, and with the fake bus following its commands exactly."""

    def __init__(self, drv):
        self.d = drv
        self.plant = None

    def __getattr__(self, n):
        return getattr(self.d, n)

    def advance(self, dt):
        pass

    def on_layout(self, layout):
        pass
