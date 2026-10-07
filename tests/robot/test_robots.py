"""Robots as files: a fixed table-top arm, or the same arm on a base that drives up to things (robots.py, World, skills)."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from echotwin.robot import hub, server
from echotwin.robot import robots as RB
from echotwin.robot.features import prop_skills as PS
from echotwin.robot.scene import Layout
from echotwin.robot.world import World

FLOOR = {"kind": "floor", "height": 0.0}


def _floor(*props, obstacles=()):
    lay = Layout(table_half=(1.0, 0.8), surface=dict(FLOOR))
    lay.props = [dict(p) for p in props]
    lay.obstacles = [dict(o) for o in obstacles]
    return lay


def _cube(x, y, name="cube", half=0.025):
    return {"name": name, "shape": "box", "pos": (x, y), "yaw": 0.0, "size": (half, half, half), "rgb": (0.8, 0.3, 0.3)}


def test_the_robots_are_files():
    assert {"table_arm", "mobile_arm"} <= set(RB.list_robots())
    t, m = RB.load("table_arm"), RB.load("mobile_arm")
    assert not t.mobile and m.mobile and m.footprint == (0.26, 0.26) and m.mount_height == 0.12 and m.speed == 0.2
    assert RB.load(None).name == "table_arm"
    assert RB.for_surface({"kind": "table"}) == "table_arm" and RB.for_surface(FLOOR) == "mobile_arm"
    assert RB.for_surface({"kind": "other", "height": 0.45}) == "mobile_arm" and RB.for_surface(None) == "table_arm"


def test_a_robot_of_your_own(tmp_path):
    f = tmp_path / "big.json"
    f.write_text(json.dumps({"name": "big", "base": {"kind": "mobile", "footprint_cm": [40, 30], "mount_height_cm": 20}, "arm_size": 1.5}))
    r = RB.load(str(f))
    assert r.footprint == (0.4, 0.3) and r.mount_height == 0.2 and r.arm_size == 1.5 and r.arm == "builtin"
    for bad in ({"name": "x"}, {"name": "x", "base": {"kind": "legs"}}, {"name": "x", "base": {"kind": "mobile", "footprint_cm": [0, 10]}},
                {"name": "x", "base": {"kind": "mobile", "speed_cm_s": "fast"}}):
        f.write_text(json.dumps(bad))
        with pytest.raises(RB.RobotError):
            RB.load(str(f))
    with pytest.raises(RB.RobotError):
        RB.load("nope")


def test_the_table_arm_is_the_arm_as_before():
    w = World(Layout())
    assert not w.mobile and w.base_act == [] and w.ik.base_qadr == []
    assert w.base.tolist() == [0.0, -0.3 + 0.07] and w.hand.base[2] == 0.0


def test_the_base_frame_turns_points_both_ways():
    pose = np.array([0.3, -0.2, 0.7])
    for p in ([0.5, 0.1], [-0.4, 0.6]):
        assert World._abs(World._rel(p, pose), pose) == pytest.approx(p)
    assert World._rel([0.3, 0.0], np.array([0.3, -0.2, 0.0])) == pytest.approx([0.0, 0.2])               # ahead is +y
    assert World._rel([0.3 - 0.2, -0.2], np.array([0.3, -0.2, np.pi / 2])) == pytest.approx([0.0, 0.2])  # turned left: ahead is -x


def test_a_mobile_robot_stands_on_the_floor_and_finds_a_free_spot_facing_the_thing():
    w = World(_floor(_cube(0.45, 0.4), _cube(0.45, 0.1, "other")), None, "mobile_arm")
    assert w.mobile and len(w.base_act) == 3
    assert w.data.qpos[w.ik.base_qadr[1]] == pytest.approx(w.hand.base[1])
    assert not w.reachable(w.obj_pos("prop_0")[:2])                                   # too far from where it starts
    at = w.standoff(w.obj_pos("prop_0")[:2], ignore="prop_0")
    assert at is not None and w.reachable_from(at, w.obj_pos("prop_0")[:2])
    ahead = World._rel(w.obj_pos("prop_0")[:2], at)
    assert abs(ahead[0]) < 1e-6 and ahead[1] > 0                                       # it faces the cube
    body = 0.5 * np.hypot(*w.robot.footprint)
    assert np.hypot(*(at[:2] - w.obj_pos("prop_1")[:2])) > body + 0.025                 # and does not stand on the other one
    assert w.refusal("prop_0", np.array([-0.5, 0.3])) == ""


def test_with_no_room_to_stand_it_says_so():
    walls = [{"name": f"w{k}", "shape": "box", "pos": (0.3 * np.cos(a), 0.3 * np.sin(a)), "yaw": 0.0, "size": (0.08, 0.08, 0.2),
              "rgb": (0.5, 0.5, 0.5)} for k, a in enumerate(np.linspace(0, 2 * np.pi, 12, endpoint=False))]
    w = World(_floor(_cube(0.0, 0.0), obstacles=walls), None, "mobile_arm")
    assert "no free spot" in w.refusal("prop_0")


def test_it_drives_to_the_thing_picks_it_up_drives_to_the_place_and_puts_it_down():
    w = World(_floor(_cube(0.45, 0.4), _cube(-0.5, 0.3, "mark", 0.02)), None, "mobile_arm")
    w.settle(20)
    task = {"object": "prop_0", "name": "cube", "goal": np.array([-0.38, 0.3]), "h": 0.025, "plan": {"goal": ("dir", (-1, 0), 0.2)}}
    wps = PS.waypoints(w, task, dict(PS.DEFAULTS))
    assert [k for k, _ in wps] == ["drive", "move", "move", "grip", "move", "drive", "move", "move", "grip", "move"]
    res = PS.imagine(w, task, wps, PS.DEFAULTS["speed"])
    assert res["ok"], res["text"]
    assert w.hand.drive is None and w.hand.base[1] < -0.5                               # imagining did not move the real robot


def test_no_drive_when_the_arm_reaches_already():
    w = World(_floor(_cube(0.0, -0.75)), None, "mobile_arm")          # just in front of where it starts
    task = {"object": "prop_0", "name": "cube", "goal": np.array([0.08, -0.75]), "h": 0.025, "plan": {"goal": ("dir", (1, 0), 0.08)}}
    assert "drive" not in [k for k, _ in PS.waypoints(w, task, dict(PS.DEFAULTS))]


def test_size_needed_on_wheels_is_about_the_jaws_only():
    w = World(_floor({**_cube(0.8, 0.7), "size": (0.075, 0.075, 0.05)}), None, "mobile_arm")
    o = w.size_needed("prop_0")
    assert o["reach"] == 0.0 and o["needed"] == pytest.approx((0.15 + 0.004) / 0.08, abs=0.01)


@pytest.fixture(scope="module")
def sim():
    from echotwin.robot import dataset as D
    from echotwin.robot.sim import Sim
    orig = D.EPISODES
    D.Dataset.__init__.__defaults__ = (Path(tempfile.mkdtemp()),)
    s = Sim(lambda m: None)
    s.said = []
    s.say = s.said.append
    yield s
    D.Dataset.__init__.__defaults__ = (orig,)


def test_the_surface_picks_the_robot_unless_you_chose_one(sim):
    sim.robot_choice = None
    sim.apply_scan(_floor(_cube(0.3, 0.3)), {"greeting": "floor", "mode": "file"})
    assert sim.world.robot.name == "mobile_arm"
    sim.apply_scan(Layout(), {"greeting": "table", "mode": "file"})
    assert sim.world.robot.name == "table_arm"
    sim.set_robot("mobile_arm")
    assert sim.world.robot.name == "mobile_arm" and sim.robot_choice == "mobile_arm" and "drives up" in sim.said[-1]
    sim.apply_scan(Layout(), {"greeting": "table again", "mode": "file"})
    assert sim.world.robot.name == "mobile_arm"                                        # your choice stays
    sim.set_robot("auto")
    assert sim.world.robot.name == "table_arm" and sim.robot_choice is None
    sim.set_robot("nope")
    assert sim.world.robot.name == "table_arm" and "keep the one" in sim.said[-1]


def test_the_endpoints(monkeypatch):
    calls = []
    monkeypatch.setattr(hub.sim, "submit", lambda fn, *a: calls.append((fn.__name__, a)))
    c = TestClient(server.app)
    r = c.get("/api/robots").json()
    assert {x["name"] for x in r["robots"]} >= {"table_arm", "mobile_arm"} and r["can_switch"]
    assert c.post("/api/robot", json={"name": "mobile_arm"}).json()["ok"] and calls[-1] == ("set_robot", ("mobile_arm",))
    assert c.post("/api/robot", json={}).json()["ok"] and calls[-1] == ("set_robot", ("auto",))
    assert c.post("/api/robot", json={"name": "legs"}).json() == {"ok": False, "error": "No such robot."}
