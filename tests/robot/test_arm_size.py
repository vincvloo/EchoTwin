"""The arm at another size: what it would take to pick up the objects on the table (arm.sized, arm.resize, World.size_needed)."""
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest
from fastapi.testclient import TestClient

from echotwin.robot import arm as A
from echotwin.robot import hub, server
from echotwin.robot.features import prop_skills as PS
from echotwin.robot.scene import Layout
from echotwin.robot.world import BASE_INSET, World

BOX15 = {"name": "box", "shape": "box", "pos": (0.0, 0.05), "yaw": 0.0, "size": (0.075, 0.075, 0.05), "rgb": (0.6, 0.5, 0.4)}
CARD = {"name": "card", "shape": "flat", "pos": (0.12, 0.0), "yaw": 0.0, "size": (0.03, 0.04, 0.003), "rgb": (0.9, 0.9, 0.9)}
MARK = {"name": "mark", "shape": "box", "pos": (0.25, 0.1), "yaw": 0.0, "size": (0.02, 0.02, 0.02), "rgb": (0.2, 0.3, 0.8)}


def _layout(*props):
    lay = Layout()
    lay.props = [dict(p) for p in props]
    return lay


def test_sized_scales_the_tool_the_jaws_and_the_opening_and_comes_back():
    arm = A.load("builtin")
    big = arm.sized(2.0)
    assert big.scale == 2.0 and big.max_opening == pytest.approx(0.16)
    assert big.tool["pos"] == pytest.approx([2 * v for v in arm.tool["pos"]])
    assert big.gripper["closed"] == pytest.approx([2 * v for v in arm.gripper["closed"]])     # sliding pads: a length
    back = big.sized(1.0)
    assert back.tool["pos"] == pytest.approx(arm.tool["pos"]) and back.max_opening == pytest.approx(0.08)
    assert arm.scale == 1.0                                                                  # the original is untouched


def test_a_turning_jaw_keeps_its_angles():
    arm = A.load("so_arm100")
    if arm.missing_files():
        pytest.skip("so_arm100 is not downloaded")
    assert arm.sized(2.0).gripper["closed"] == arm.gripper["closed"]


def _model(k):
    spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
    arm = A.load("builtin").sized(k)
    A.compose(spec, arm)
    return spec.compile(), arm


def _actuator(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, A.PREFIX + name)


def test_resize_scales_lengths_masses_and_gains_like_a_bigger_arm():
    m1, a1 = _model(1.0)
    m2, a2 = _model(2.0)
    arm_mass = lambda m: sum(m.body_mass[b] for b in range(m.nbody) if m.body(b).name.startswith(A.PREFIX))
    assert arm_mass(m2) == pytest.approx(8 * arm_mass(m1), rel=1e-6)                         # k**3
    pad = lambda m: m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, A.PREFIX + "pad_l")]
    assert pad(m2) == pytest.approx(2 * pad(m1))
    pan1, pan2 = _actuator(m1, "pan"), _actuator(m2, "pan")
    assert m2.actuator_gainprm[pan2, 0] == pytest.approx(32 * m1.actuator_gainprm[pan1, 0])  # torque gain k**5
    fl1, fl2 = _actuator(m1, "fl"), _actuator(m2, "fl")
    assert m2.actuator_gainprm[fl2, 0] == pytest.approx(8 * m1.actuator_gainprm[fl1, 0])     # sliding jaw gain k**3
    assert m2.actuator_forcerange[fl2] == pytest.approx(2 * m1.actuator_forcerange[fl1])    # the squeeze only grows with k
    assert m2.actuator_ctrlrange[fl2] == pytest.approx(2 * m1.actuator_ctrlrange[fl1])      # the jaws travel twice as far


def test_a_bigger_arm_reaches_further_and_stands_further_in():
    w1, w2 = World(Layout(), A.load("builtin")), World(Layout(), A.load("builtin").sized(2.0))
    assert w2.workspace.r_max == pytest.approx(2 * w1.workspace.r_max, abs=0.05)
    assert w2.base[1] - w1.base[1] == pytest.approx(BASE_INSET)


def test_what_it_takes_for_each_object():
    w = World(_layout(BOX15, CARD, MARK))
    box, card, mark = (w.size_needed(f"prop_{i}") for i in range(3))
    assert box["needed"] == pytest.approx((0.15 + 0.004) / 0.08, abs=0.01) and box["width_cm"] == 15.0
    assert card["needed"] is None and "thick" in card["why"]                                # no size pinches a card
    assert mark["needed"] <= 1.0 + 1e-9 or mark["reach"] > 1.0
    big = World(_layout(BOX15, CARD, MARK), A.load("builtin").sized(2.5))                     # the answer does not depend on
    assert big.size_needed("prop_0")["needed"] == pytest.approx(box["needed"], abs=0.02)     # the size the arm has now
    assert big.size_needed("prop_2")["reach"] == pytest.approx(mark["reach"], abs=0.06)


def test_far_things_need_reach_and_close_things_cap_the_size():
    far = World(_layout({**MARK, "pos": (0.3, 0.27)}))
    o = far.size_needed("prop_0")
    assert o["needed"] == o["reach"] > 1.0
    near = World(_layout({**MARK, "pos": (0.0, -0.17)}))                                     # just in front of the base
    assert near.size_needed("prop_0")["most"] < 3.0


def _imagine(w):
    task = {"object": "prop_0", "name": "box", "goal": np.array([0.2, 0.15]), "h": 0.05}
    why = w.refusal("prop_0")
    if why:
        return False, why
    res = PS.imagine(w, task, PS.waypoints(w, task, dict(PS.DEFAULTS)), PS.DEFAULTS["speed"])
    return bool(res["ok"]), res["text"]


def test_a_15_cm_box_needs_a_bigger_arm_and_the_bigger_arm_moves_it():
    lay = _layout(BOX15, MARK)
    lay.table_half = (0.6, 0.45)
    ok, why = _imagine(World(lay))
    assert not ok and "gripper opens 8 cm" in why
    w = World(lay, A.load("builtin").sized(2.0))
    w.settle(20)
    ok, text = _imagine(w)
    assert ok, text


@pytest.fixture(scope="module")
def sim():
    from echotwin.robot import dataset as D
    from echotwin.robot.sim import Sim
    orig = D.EPISODES
    D.Dataset.__init__.__defaults__ = (Path(tempfile.mkdtemp()),)
    s = Sim(lambda m: None)
    s.said = []
    s.say = s.said.append
    s._rebuild(_layout(BOX15, MARK))
    s.base_layout = s.world.layout.copy()
    yield s
    D.Dataset.__init__.__defaults__ = (orig,)


def test_the_robot_resizes_its_arm_keeps_the_table_and_says_what_it_can_do(sim):
    names = [p["name"] for p in sim.world.layout.props]
    sim.set_arm(size=2.0)
    assert sim.world.arm.scale == 2.0 and sim.world.arm.name == "builtin" and [p["name"] for p in sim.world.layout.props] == names
    assert "2.00 times its size" in sim.said[-1] and "jaws open 16 cm" in sim.said[-1]
    assert not sim.world.refusal("prop_0")                                                    # the 15 cm box is in its grip now
    sim.set_arm(size=9)
    assert sim.world.arm.scale == 2.0 and "between" in sim.said[-1]                          # outside the tested range
    sim.set_arm(size=1.0)
    assert sim.world.arm.scale == 1.0 and "gripper opens 8 cm" in sim.world.refusal("prop_0")


def test_the_endpoints(sim, monkeypatch):
    monkeypatch.setattr(hub.sim, "world", sim.world)
    calls = []
    monkeypatch.setattr(hub.sim, "submit", lambda fn, *a: calls.append((fn.__name__, a)))
    c = TestClient(server.app)
    r = c.get("/api/arm/needed").json()
    assert r["ok"] and r["size"] == 1.0 and r["opening_cm"] == 8.0 and r["range"] == [0.75, 4.0]
    assert [o["name"] for o in r["objects"]] == ["box", "mark"] and r["all"] == pytest.approx(1.93, abs=0.01)
    assert c.post("/api/arm", json={"size": 2.5}).json()["ok"] and calls[-1] == ("set_arm", (None, 2.5))
    assert c.post("/api/arm", json={"name": "builtin", "size": 1}).json()["ok"] and calls[-1] == ("set_arm", ("builtin", 1.0))
    for bad in ({}, {"size": 10}, {"size": "big"}, {"name": "nope"}):
        r = c.post("/api/arm", json=bad).json()
        assert r["ok"] is False and r["error"], bad
