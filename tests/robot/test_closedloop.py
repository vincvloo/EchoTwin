"""The closed loop: look before gripping, check the grasp and the goal, retry, then ask."""
import numpy as np
import pytest

from echotwin.robot import drivers as D
from echotwin.robot import real as R
from echotwin.robot.features import prop_skills as PS
from echotwin.robot.scene import Layout
from echotwin.robot.world import World

GOAL = np.array([0.13, 0.0])


def _layout():
    lay = Layout()
    lay.props = [{"name": "cube", "shape": "box", "pos": (0.0, -0.05), "yaw": 0.0, "size": (0.025, 0.02, 0.025), "rgb": (0.8, 0.2, 0.2)},
                 {"name": "mark", "shape": "box", "pos": (0.2, 0.0), "yaw": 0.0, "size": (0.02, 0.02, 0.02), "rgb": (0.2, 0.3, 0.9)}]
    return lay


def _mock():
    twin = World(_layout())
    drv = D.MockDriver(twin.layout, twin.arm)
    return R.RealBackend(twin, drv, camera=R.mock_camera(drv)), drv.plant


def _task(b):
    return {"object": "prop_0", "name": "cube", "goal": GOAL, "h": b.twin.half("prop_0")}


def _run(b, loop=True, hook=None, max_ticks=3000):
    task, skill = _task(b), dict(PS.DEFAULTS)
    if loop:
        wps, lp = PS.loop_start(b, task, skill)
    else:
        wps, lp = PS.waypoints(b, task, skill), None
    r = {"wps": wps, "i": 0, "speed": skill["speed"], "yaw": b.grasp_yaw("prop_0"), "loop": lp}
    for t in range(max_ticks):
        if hook:
            hook(t, r)
        a = PS.waypoint_action(b, r)
        if a is None:
            break
        b.step(a)
    for _ in range(30):
        b.step(np.array([0, 0, 0, 0.0, r["yaw"]]))
    return r


def _err(plant):
    return float(np.linalg.norm(plant.obj_pos("prop_0")[:2] - GOAL))


def test_closed_loop_off_means_the_old_eight_steps_and_auto_is_for_the_real_arm():
    b, _ = _mock()
    assert len(PS.waypoints(b, _task(b), dict(PS.DEFAULTS))) == 8
    assert PS.closed_loop_on(b, {}) and not PS.closed_loop_on(World(_layout()), {})
    assert PS.closed_loop_on(World(_layout()), {"CLOSED_LOOP": "on"}) and not PS.closed_loop_on(b, {"CLOSED_LOOP": "off"})


def test_an_object_moved_after_planning_is_found_before_gripping():
    b, plant = _mock()
    plant.nudge("prop_0", (0.05, 0.03))                      # the twin still believes the old place
    r = _run(b, loop=True)
    assert _err(plant) < 0.04, plant.obj_pos("prop_0")
    assert any("isn't where I thought" in e for e in r["loop"]["log"]) and r["loop"]["attempts"] == 0


def test_without_the_loop_the_same_move_misses():
    b, plant = _mock()
    plant.nudge("prop_0", (0.05, 0.03))
    _run(b, loop=False)
    assert _err(plant) > 0.05


def test_a_missed_grasp_is_noticed_and_retried():
    b, plant = _mock()
    state = {"done": False}

    def knock(t, r):                                         # just as the jaws are about to close, the cube is pushed away
        if not state["done"] and r["i"] < len(r["wps"]) and r["wps"][r["i"]][0] == "grip" and r["wps"][r["i"]][1] == 1.0:
            plant.nudge("prop_0", (0.05, 0.0))
            state["done"] = True
    r = _run(b, hook=knock)
    log = " ".join(r["loop"]["log"])
    assert state["done"] and "I missed the cube" in log and r["loop"]["attempts"] >= 1
    assert _err(plant) < 0.04


def test_after_three_failures_it_gives_up_instead_of_looping_forever(monkeypatch):
    w = World(_layout())
    w._held = lambda d: None                                 # the pads never hold anything
    r = _run(w, loop=True)
    assert r["loop"]["attempts"] == PS.MAX_ATTEMPTS and "I tried 3 times" in " ".join(r["loop"]["log"])
    assert r["i"] >= len(r["wps"])


def test_an_object_that_is_not_there_is_reported_not_guessed():
    b, plant = _mock()
    plant.set_obj_pose("prop_0", (0.33, 0.25))
    r = _run(b)
    assert r["loop"].get("lost") and "I can't see the cube" in " ".join(r["loop"]["log"])


def test_the_simulation_with_the_loop_on_still_moves_things_and_logs_the_attempts(monkeypatch, tmp_path):
    import echotwin.robot.dataset as DS
    from echotwin.robot import runlog
    from echotwin.robot.features import move_things as MT
    from echotwin.robot.sim import Sim
    monkeypatch.setenv("CLOSED_LOOP", "on")
    (tmp_path / "episodes").mkdir()
    monkeypatch.setattr(DS.Dataset.__init__, "__defaults__", (tmp_path / "episodes",))
    s = Sim(lambda m: None)
    s._rebuild(_layout())
    s.base_layout = s.world.layout.copy()
    s.world.settle(20)
    out = []
    s.emit = out.append
    s.voice.emit = s.emit
    for ep in range(1):                                          # one demo from a good run, so it plans by itself
        pass
    task, goal, o = s._make_prop_task({"prop": 0, "goal": ("near", 1, "next to")}, "put the cube next to the mark")
    skill = dict(PS.DEFAULTS)
    wps, lp = PS.loop_start(s.world, task, skill)
    s.replay = {"wps": wps, "i": 0, "speed": skill["speed"], "task": task, "yaw": s.world.grasp_yaw("prop_0"), "loop": lp,
                "before": {k: s.world.obj_pos(k)[:2].copy() for k in s.world.things()}}
    s.mode, s.authority = "move", "robot"
    for _ in range(1500):
        if s.mode != "move":
            break
        s._tick()
    said = [m["text"] for m in out if m.get("t") == "log"]
    assert said[-1] == "Done.", said[-3:]
    row = runlog.read(s.dataset.root.parent / "runs.jsonl")[-1]
    assert row["ok"] and row["attempts"] == 0


def test_in_the_app_three_failures_end_in_asking_to_be_shown(monkeypatch, tmp_path):
    import echotwin.robot.dataset as DS
    from echotwin.robot.sim import Sim
    monkeypatch.setenv("CLOSED_LOOP", "on")
    (tmp_path / "episodes").mkdir()
    monkeypatch.setattr(DS.Dataset.__init__, "__defaults__", (tmp_path / "episodes",))
    s = Sim(lambda m: None)
    s._rebuild(_layout())
    s.base_layout = s.world.layout.copy()
    s.world.settle(20)
    s.world._held = lambda d: None                                  # the pads never hold anything
    out = []
    s.emit = out.append
    s.voice.emit = s.emit
    task, goal, o = s._make_prop_task({"prop": 0, "goal": ("near", 1, "next to")}, "put the cube next to the mark")
    skill = dict(PS.DEFAULTS)
    wps, lp = PS.loop_start(s.world, task, skill)
    s.replay = {"wps": wps, "i": 0, "speed": skill["speed"], "task": task, "yaw": s.world.grasp_yaw("prop_0"), "loop": lp,
                "before": {k: s.world.obj_pos(k)[:2].copy() for k in s.world.things()}}
    s.mode, s.authority = "move", "robot"
    for _ in range(4000):
        if s.mode != "move":
            break
        s._tick()
    said = [m["text"] for m in out if m.get("t") == "log"]
    assert s.mode == "teach", said[-3:]
    assert any("I missed the cube. Trying again" in t for t in said) and any("didn't go well" in t for t in said)
