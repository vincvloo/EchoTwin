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


# ---------------- choosing a back-end ----------------
def test_factory_default_is_the_simulation():
    w, note = B.make(env={})
    assert w.name == "sim" and "Simulation" in note


def test_factory_real_without_a_port_is_the_mock_arm():
    r, note = B.make(env={"BACKEND": "real"})
    assert r.name == "real" and "mock arm" in note and r.arm_ready()
    assert r.view is r.driver.plant and r.view is not r.twin


def test_factory_falls_back_to_the_simulation_with_a_reason():
    w, note = B.make(env={"BACKEND": "real", "REAL_PORT": "COM_DOES_NOT_EXIST"})
    assert w.name == "sim" and "Could not use the real arm on COM_DOES_NOT_EXIST" in note
    w2, note2 = B.make(env={"BACKEND": "robot"})
    assert w2.name == "sim" and "not sim or real" in note2


def _sim(monkeypatch, backend, tmp_path):
    from pathlib import Path

    from echotwin.robot import dataset as D
    from echotwin.robot.sim import Sim
    monkeypatch.setenv("BACKEND", backend)
    (tmp_path / "episodes").mkdir(exist_ok=True)
    monkeypatch.setattr(D.Dataset.__init__, "__defaults__", (Path(tmp_path) / "episodes",))
    return Sim(lambda m: None)


def test_sim_runs_on_the_chosen_backend_and_says_so(monkeypatch, tmp_path):
    s = _sim(monkeypatch, "real", tmp_path)
    assert s.world.name == "real" and "mock arm" in s.backend_note
    out = []
    s.emit = out.append
    s._emit_state()
    st = [m for m in out if m.get("t") == "state"][-1]
    assert st["backend"]["name"] == "real" and st["backend"]["ready"] is True
    s.halt("button")
    assert s.world.stopped and s.world.driver.stopped
    s.resume()
    assert not s.world.stopped


def test_the_real_arm_waits_to_be_armed(monkeypatch, tmp_path):
    from echotwin.robot import drivers as D
    from echotwin.robot import real as R
    from echotwin.robot.features import move_things as MT
    from echotwin.robot.router import Intent
    s = _sim(monkeypatch, "sim", tmp_path)
    lay = Layout()
    lay.props = [{"name": "cube", "shape": "box", "pos": (0.0, -0.05), "yaw": 0.0, "size": (0.025, 0.02, 0.025), "rgb": (0.8, 0.3, 0.3)},
                 {"name": "mark", "shape": "box", "pos": (0.2, 0.0), "yaw": 0.0, "size": (0.02, 0.02, 0.02), "rgb": (0.2, 0.3, 0.8)}]
    twin = World(lay)
    s.world = R.RealBackend(twin, D.MockDriver(twin.layout, twin.arm), armed=False)
    s.base_layout = s.world.layout.copy()
    said = []
    s.emit = lambda m: said.append(m["text"]) if m.get("t") == "log" else None
    s.voice.emit = s.emit
    text = "put the cube next to the mark"
    s.handle_prop_task(MT.parse(text, s.world.layout.props), text)
    assert any("waiting" in t for t in said) and s.mode != "move"
    s.handle_intent(Intent("control", "arm_robot"))
    assert s.world.arm_ready()


# ---------------- the run log ----------------
def test_run_log_records_backend_class_and_outcome(tmp_path):
    from echotwin.robot import runlog
    p = tmp_path / "runs.jsonl"
    task = {"name": "cup", "instruction": "put the cup next to the box", "m": {"width": 0.07, "height": 0.10, "length": 0.07}}
    runlog.record("sim", task, {"ok": True, "err": 0.01}, p)
    runlog.record("real", task, {"ok": False, "text": "the cup tips over", "err": 0.08}, p, seconds=12.3)
    runlog.record("real", task, {"ok": True, "err": 0.02}, p)
    rows = runlog.read(p)
    assert [r["backend"] for r in rows] == ["sim", "real", "real"] and rows[1]["why"] == "the cup tips over"
    s = runlog.summary(rows)
    assert s["sim"]["tall"] == (1, 1) and s["real"]["tall"] == (2, 1) and s["real"]["all"] == (2, 1)
    assert "| real | 1/2 |" in runlog.table(rows) or "real" in runlog.table(rows)
    assert runlog.read(tmp_path / "none.jsonl") == []


def test_a_finished_move_is_logged(monkeypatch, tmp_path):
    from echotwin.robot import runlog
    s = _sim(monkeypatch, "sim", tmp_path)
    lay = Layout()
    lay.props = [{"name": "cube", "shape": "box", "pos": (0.0, -0.05), "yaw": 0.0, "size": (0.025, 0.02, 0.025), "rgb": (0.8, 0.3, 0.3)},
                 {"name": "mark", "shape": "box", "pos": (0.2, 0.0), "yaw": 0.0, "size": (0.02, 0.02, 0.02), "rgb": (0.2, 0.3, 0.8)}]
    s._rebuild(lay)
    s.base_layout = s.world.layout.copy()
    task, goal, o = s._make_prop_task({"prop": 0, "goal": ("near", 1, "next to")}, "put the cube next to the mark")
    s.mode, s.authority = "move", "robot"
    s.replay = {"task": task, "before": {k: s.world.obj_pos(k)[:2].copy() for k in s.world.things()}}
    s._replay_done()
    rows = runlog.read(s.dataset.root.parent / "runs.jsonl")
    assert len(rows) == 1 and rows[0]["backend"] == "sim" and rows[0]["object"] == "cube"
