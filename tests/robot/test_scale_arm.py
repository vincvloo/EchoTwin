"""Rescaling the twin (when the photo's sizes are wrong, or to make Pip look bigger or smaller) and switching the arm."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from echotwin.robot import arm as A
from echotwin.robot import hub, server
from echotwin.robot.scene import Layout

PROPS = [{"name": "glass", "shape": "cylinder", "pos": (-0.1, 0.05), "yaw": 0.0, "size": (0.03, 0.03, 0.05), "rgb": (0.5, 0.6, 0.7)},
         {"name": "case", "shape": "box", "pos": (0.1, 0.1), "yaw": 0.0, "size": (0.04, 0.03, 0.02), "rgb": (0.1, 0.1, 0.1), "mesh_scale": (1.0, 1.0, 1.0)}]


def _layout(mesh=True):
    lay = Layout()
    lay.props = [dict(p) for p in PROPS]
    lay.obstacles = [{"name": "wall", "shape": "box", "pos": (0.5, 0.0), "size": (0.1, 0.1, 0.1), "rgb": (0.5, 0.5, 0.5), "yaw": 0.0}]
    if mesh:                                                      # a file that does not exist: only for the layout maths
        lay.scene = [{"file": "m.obj", "pos": (0.0, 0.0, -0.75), "euler": (0, 0, 0), "scale": 2.0}]
    lay.view = {"pos": [0.0, -0.4, 0.45], "xyaxes": [1, 0, 0, 0, 0.7, 0.7], "fovy": 60.0}
    return lay


def test_scaled_scales_everything_except_the_view_angle_and_leaves_the_original():
    lay = _layout()
    s = lay.scaled(0.5)
    assert s.table_half == tuple(v * 0.5 for v in lay.table_half)
    assert s.props[0]["size"] == (0.015, 0.015, 0.025) and s.props[0]["pos"] == (-0.05, 0.025)
    assert s.props[1]["mesh_scale"] == (0.5, 0.5, 0.5)
    assert s.obstacles[0]["size"] == (0.05, 0.05, 0.05) and s.obstacles[0]["pos"] == (0.25, 0.0)
    assert s.scene[0]["scale"] == 1.0 and s.scene[0]["pos"] == (0.0, 0.0, -0.375)
    assert s.view["pos"] == [0.0, -0.2, 0.225] and s.view["xyaxes"] == lay.view["xyaxes"] and s.view["fovy"] == 60.0
    assert lay.props[0]["size"] == (0.03, 0.03, 0.05) and "scale" not in lay.meta        # the original is untouched
    again = s.scaled(4.0)
    assert again.meta["scale"] == pytest.approx(2.0)                                     # the product so far
    assert again.props[0]["size"] == pytest.approx((0.06, 0.06, 0.1))


@pytest.fixture(scope="module")
def sim():
    from echotwin.robot import dataset as D
    from echotwin.robot.sim import Sim
    orig = D.EPISODES
    D.Dataset.__init__.__defaults__ = (Path(tempfile.mkdtemp()),)
    s = Sim(lambda m: None)
    s.say = lambda text: said.append(text)
    s.voice.emit = lambda m: None
    yield s
    D.Dataset.__init__.__defaults__ = (orig,)


said: list[str] = []


def _fresh(sim):
    sim._rebuild(_layout(mesh=False))
    sim.base_layout = sim.world.layout.copy()
    said.clear()


def test_the_robot_rescales_the_table_and_the_arm_stays(sim):
    _fresh(sim)
    before = sim.world.arm.name, sim.world.base_xy(sim.world.layout)
    w0 = 2 * sim.world.layout.props[0]["size"][0]
    sim.set_scale(0.5)
    assert 2 * sim.world.layout.props[0]["size"][0] == pytest.approx(w0 / 2)
    assert sim.world.arm.name == before[0] and "0.50" in said[-1]
    assert sim.world.base_xy(sim.world.layout)[1] == pytest.approx(-sim.world.layout.table_half[1] + 0.07)   # still at the front edge
    sim.set_scale(2.0)                                                                                     # and back
    assert 2 * sim.world.layout.props[0]["size"][0] == pytest.approx(w0)
    assert sim.base_layout.meta["scale"] == pytest.approx(1.0)


def test_a_scale_outside_the_limits_is_refused_and_nothing_changes(sim):
    _fresh(sim)
    size = sim.world.layout.props[0]["size"]
    sim.set_scale(0.01)
    sim.set_scale(100)
    assert sim.world.layout.props[0]["size"] == size and "stay between" in said[-1]


def test_the_endpoint_turns_a_real_width_into_a_factor(sim, monkeypatch):
    _fresh(sim)
    monkeypatch.setattr(hub.sim, "world", sim.world)
    calls = []
    monkeypatch.setattr(hub.sim, "submit", lambda fn, *a: calls.append((fn.__name__, a)))
    monkeypatch.setattr(hub.sim, "base_layout", sim.base_layout)
    c = TestClient(server.app)
    r = c.post("/api/twin/scale", json={"object": 0, "width_cm": 3}).json()            # the glass is 6 cm in the twin
    assert r["ok"] and r["factor"] == 0.5 and calls == [("set_scale", (0.5,))]
    assert c.post("/api/twin/scale", json={"factor": 3.0}).json()["factor"] == 3.0
    for bad in ({}, {"factor": 0}, {"factor": 50}, {"object": 9, "width_cm": 3}, {"object": 0, "width_cm": "x"}):
        r = c.post("/api/twin/scale", json=bad).json()
        assert r["ok"] is False and r["error"], bad


def test_the_arm_list_says_which_arms_are_ready():
    arms = {a["name"]: a for a in A.describe_arms()}
    assert arms["builtin"]["ready"] and arms["builtin"]["about"]
    c = TestClient(server.app)
    r = c.get("/api/arms").json()
    assert {a["name"] for a in r["arms"]} >= {"builtin"} and r["current"] and r["can_switch"] is True


def test_switching_to_an_unknown_or_missing_arm_is_refused(sim, monkeypatch):
    c = TestClient(server.app)
    assert c.post("/api/arm", json={"name": "nope"}).json() == {"ok": False, "error": "No such arm."}
    _fresh(sim)
    keep = sim.world.arm
    sim.set_arm("nope")
    assert sim.world.arm is keep and "keep the one I have" in said[-1]


def test_the_robot_changes_its_arm_and_keeps_the_table(sim):
    ready = [a["name"] for a in A.describe_arms() if a["ready"] and a["name"] != "builtin"]
    if not ready:
        pytest.skip("no second arm is downloaded")
    _fresh(sim)
    names = [p["name"] for p in sim.world.layout.props]
    sim.set_arm(ready[0])
    assert sim.world.arm.name == A.load(ready[0]).name and [p["name"] for p in sim.world.layout.props] == names
    assert ready[0] in said[-1] or sim.world.arm.name in said[-1]
    sim.set_arm("builtin")
    assert sim.world.arm.name == "builtin"
