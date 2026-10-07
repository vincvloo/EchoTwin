"""Driving a mobile base by hand (phone joystick or dashboard pad): World.drive_by, Sim.set_drive, the 'drive' message."""
import asyncio
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from echotwin.robot import hub, ws
from echotwin.robot.scene import Layout
from echotwin.robot.world import CTRL_DT, World


def _floor(*props):
    lay = Layout(table_half=(1.0, 0.8), surface={"kind": "floor", "height": 0.0})
    lay.props = [dict(p) for p in props]
    return lay


def _cube(x, y):
    return {"name": "cube", "shape": "box", "pos": (x, y), "yaw": 0.0, "size": (0.025, 0.025, 0.025), "rgb": (0.8, 0.3, 0.3)}


@pytest.fixture
def w():
    return World(_floor(_cube(0.6, 0.5)), None, "mobile_arm")


def test_forward_sideways_and_turning_are_in_the_bases_own_frame(w):
    start = w.hand.base.copy()
    assert w.drive_by(1, 0, 0)
    assert w.hand.base[:2] - start[:2] == pytest.approx([0.0, w.robot.speed * CTRL_DT])          # forward is +y at first
    w.hand.base[2] = np.pi / 2                                                                   # turned left a quarter
    before = w.hand.base.copy()
    assert w.drive_by(1, 0, 0)
    assert w.hand.base[:2] - before[:2] == pytest.approx([-w.robot.speed * CTRL_DT, 0.0])        # now forward is -x
    before = w.hand.base.copy()
    assert w.drive_by(0, 1, 0)
    assert w.hand.base[:2] - before[:2] == pytest.approx([0.0, w.robot.speed * CTRL_DT])         # and sideways is +y
    before = w.hand.base.copy()
    assert w.drive_by(0, 0, -1)
    assert w.hand.base[2] - before[2] == pytest.approx(-w.robot.turn * CTRL_DT)


def test_the_arm_rides_along_and_holds_still(w):
    w.settle(10)
    tool = w.hand_pos().copy()
    for _ in range(20):
        w.drive_by(1, 0.3, 0)
        w.step((0, 0, 0, 0))
    w.settle(15)
    moved = w.hand_pos() - tool
    assert moved[:2] == pytest.approx(w.hand.base[:2] - w.base_xy(w.layout), abs=0.01)           # the tool came along
    assert abs(moved[2]) < 0.01                                                                  # at the same height


def test_it_stops_before_driving_into_a_thing_but_may_turn(w):
    w.hand.base[:] = [0.6, 0.5 - 0.225, 0.0]                                                     # touching the cube already
    w.ik.set_base(w.hand.base)
    assert not w.drive_by(1, 0, 0)                                                               # forward is into the cube
    assert w.hand.base[:2] == pytest.approx([0.6, 0.275])
    assert w.drive_by(-1, 0, 0)                                                                  # backing away is fine
    assert not w.drive_by(1, 0, 1) and w.hand.base[2] > 0                                        # blocked, but it turned


def test_it_stays_near_the_mapped_area(w):
    w.hand.base[:] = [0.0, -1.295, 0.0]                                                          # half a metre in front of the map
    assert not w.drive_by(-1, 0, 0) and w.hand.base[1] == pytest.approx(-1.295)


def test_a_fixed_arm_does_not_drive():
    w = World(Layout())
    base = w.hand.base.copy()
    assert not w.drive_by(1, 1, 1) and (w.hand.base == base).all()


@pytest.fixture(scope="module")
def sim():
    from echotwin.robot import dataset as D
    from echotwin.robot.sim import Sim
    orig = D.EPISODES
    D.Dataset.__init__.__defaults__ = (Path(tempfile.mkdtemp()),)
    s = Sim(lambda m: None)
    s.haptics = []
    s.haptic = s.haptics.append
    s.apply_scan(_floor(_cube(0.6, 0.5)), {"greeting": "floor", "mode": "file"})
    yield s
    D.Dataset.__init__.__defaults__ = (orig,)


def test_the_robot_drives_while_the_phone_sends_and_stops_when_it_stops(sim):
    assert sim.world.robot.mobile and sim.authority == "human"
    y0 = sim.world.hand.base[1]
    sim.set_drive(1, 0, 0)
    for _ in range(10):
        sim._tick()
    y1 = sim.world.hand.base[1]
    assert y1 - y0 == pytest.approx(10 * sim.world.robot.speed * CTRL_DT)
    sim.base_t = time.time() - 1.0                                                               # the phone went quiet
    for _ in range(5):
        sim._tick()
    assert sim.world.hand.base[1] == y1


def test_blocked_buzzes_the_phone_once_a_second(sim):
    w = sim.world
    w.hand.base[:] = [0.6, w.obj_pos("prop_0")[1] - 0.225, 0.0]
    sim.haptics.clear()
    sim.base_blocked_t = 0.0
    sim.set_drive(1, 0, 0)
    for _ in range(6):
        sim._tick()
    assert len(sim.haptics) == 1
    sim.set_drive(0, 0, 0)


def test_the_drive_message_reaches_the_robot(monkeypatch):
    got = []
    monkeypatch.setattr(hub.sim, "set_drive", lambda *a: got.append(a))
    monkeypatch.setattr(ws.sim, "set_drive", lambda *a: got.append(a))
    asyncio.run(ws.on_message({"t": "drive", "forward": 0.5, "sideways": -1, "turn": 0}, "phone"))
    assert got == [(0.5, -1, 0)]
