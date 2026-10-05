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


# ---------------- lining the gripper up by looking at it ----------------
GLASS = {"name": "glass", "shape": "cylinder", "pos": (0.05, 0.0), "yaw": 0.0, "size": (0.035, 0.035, 0.05), "rgb": (0.8, 0.8, 0.8)}


def _off_arm(encoder_offset=0.02, seed=3):
    """A mock arm whose encoders are 1 degree or so off, a glass to pick and a mark to put it by."""
    lay = Layout()
    lay.props = [dict(GLASS), {"name": "mark", "shape": "box", "pos": (-0.15, 0.1), "yaw": 0.0, "size": (0.02, 0.02, 0.02), "rgb": (0.2, 0.3, 0.9)}]
    twin = World(lay)
    drv = D.MockDriver(twin.layout, twin.arm, encoder_offset=encoder_offset, seed=seed)
    return R.RealBackend(twin, drv, camera=R.mock_camera(drv)), drv.plant


def _until_grip(b, plant, align, hook=None):
    task = {"object": "prop_0", "name": "glass", "goal": np.array([-0.15, 0.0]), "h": 0.05}
    wps, lp = PS.loop_start(b, task, dict(PS.DEFAULTS), align=align)
    r = {"wps": wps, "i": 0, "speed": 0.2, "yaw": b.grasp_yaw("prop_0"), "loop": lp}
    for _ in range(2500):
        step = r["wps"][r["i"]] if r["i"] < len(r["wps"]) else None
        if step is not None and step[0] == "grip" and step[1] == 1.0:
            break
        a = PS.waypoint_action(b, r)
        if a is None:
            break
        b.step(a)
    return r, float(np.linalg.norm(plant.hand_pos()[:2] - plant.obj_pos("prop_0")[:2]))


def test_the_gripper_is_lined_up_with_the_object_when_the_encoders_are_off():
    b, plant = _off_arm()
    r, err_on = _until_grip(b, plant, align=True)
    b2, plant2 = _off_arm()
    _, err_off = _until_grip(b2, plant2, align=False)
    assert err_off > 0.008, err_off                               # without looking, the gripper is about a centimetre away
    assert err_on < 0.0055 and err_on < err_off / 1.5, (err_on, err_off)
    assert any("off from where my joints say" in e for e in r["loop"]["log"]) and any("Lined up" in e for e in r["loop"]["log"])


def test_a_gripper_that_cannot_be_seen_is_not_corrected_and_the_robot_says_so():
    b, plant = _off_arm()
    b.see_tool = lambda: None
    r, _ = _until_grip(b, plant, align=True)
    assert "I can't see the gripper" in " ".join(r["loop"]["log"]) and not r["loop"]["bias"].any()


def test_a_reading_that_cannot_be_trusted_is_not_used():
    b, plant = _off_arm()
    b.see_tool = lambda: b.hand_pos()[:2] + np.array([0.05, 0.0])   # always 5 cm away from where it should be
    r, _ = _until_grip(b, plant, align=True)
    log = " ".join(r["loop"]["log"])
    assert "Going on without" in log and not r["loop"]["bias"].any()


def test_only_a_tight_fit_is_worth_looking_at():
    lay = Layout()
    lay.props = [dict(GLASS), {"name": "cube", "shape": "box", "pos": (-0.15, 0.1), "yaw": 0.0, "size": (0.02, 0.02, 0.02), "rgb": (0.2, 0.3, 0.9)}]
    w = World(lay)
    assert PS._tight_fit(w, "prop_0") and not PS._tight_fit(w, "prop_1")        # a 7 cm glass in 8 cm jaws, a 4 cm cube


def test_the_simulation_does_not_look_at_itself_and_the_switch_works():
    w = World(_layout())
    assert not PS.align_on(w, {}) and PS.align_on(w, {"ALIGN": "on"}) and not PS.align_on(_off_arm()[0], {"ALIGN": "off"})
    assert not PS.align_on(_off_arm()[0], {}) and PS.align_on(_off_arm()[0], {"ALIGN": "on"})              # off unless asked for
    wps, lp = PS.loop_start(w, {"object": "prop_0", "name": "cube", "goal": GOAL, "h": 0.025}, dict(PS.DEFAULTS))
    assert lp["align"] is False
