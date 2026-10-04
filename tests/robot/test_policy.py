"""The chunk policy: numpy inference, temporal ensembling, the executor, and (with torch) training."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from echotwin.robot import policy as P
from echotwin.robot import policy_obs as O
from echotwin.robot.scene import Layout
from echotwin.robot.world import World

K = 4


def tiny(chunk_fn=None, obs_dim=O.OBS_DIM):
    """A one-layer 'network' whose output is a fixed chunk (zero weights, bias = the chunk)."""
    chunk = np.zeros((K, O.ACT_DIM), np.float32) if chunk_fn is None else chunk_fn
    return P.ChunkPolicy({"K": np.array(K), "obs_mean": np.zeros(obs_dim, np.float32), "obs_std": np.ones(obs_dim, np.float32),
                          "act_mean": np.zeros(O.ACT_DIM, np.float32), "act_std": np.ones(O.ACT_DIM, np.float32),
                          "W0": np.zeros((obs_dim, K * O.ACT_DIM), np.float32), "b0": chunk.reshape(-1).astype(np.float32)})


def test_predict_gives_a_chunk_in_real_units():
    pol = tiny(np.arange(K * 5, dtype=np.float32).reshape(K, 5))
    out = pol.predict(np.zeros(O.OBS_DIM))
    assert out.shape == (K, 5) and out[2, 1] == 11


def test_the_hidden_layers_are_relu_and_the_normalisation_is_undone():
    rng = np.random.default_rng(0)
    W0, W1 = rng.normal(size=(O.OBS_DIM, 8)).astype(np.float32), rng.normal(size=(8, K * 5)).astype(np.float32)
    p = {"K": np.array(K), "obs_mean": np.ones(O.OBS_DIM, np.float32), "obs_std": 2 * np.ones(O.OBS_DIM, np.float32),
         "act_mean": 3 * np.ones(5, np.float32), "act_std": 0.5 * np.ones(5, np.float32),
         "W0": W0, "b0": np.zeros(8, np.float32), "W1": W1, "b1": np.zeros(K * 5, np.float32)}
    x = rng.normal(size=O.OBS_DIM).astype(np.float32)
    want = (np.maximum((x - 1) / 2 @ W0, 0) @ W1).reshape(K, 5) * 0.5 + 3
    assert np.allclose(P.ChunkPolicy(p).predict(x), want, atol=1e-5)


def test_chunks_are_ensembled_with_the_oldest_counting_most():
    first, second = np.zeros((K, 5), np.float32), np.zeros((K, 5), np.float32)
    first[:, 0], second[:, 0] = 0.1, 0.3
    calls = iter([first, second])
    pol = tiny()
    pol.predict = lambda obs: next(calls)
    a0 = pol.act(np.zeros(O.OBS_DIM))                     # only the first chunk
    a1 = pol.act(np.zeros(O.OBS_DIM))                     # the first chunk (age 1, weight 1) and the second (age 0, weight exp(-0.05))
    w = np.exp(-P.ENSEMBLE_M)
    assert a0[0] == pytest.approx(0.1)
    assert a1[0] == pytest.approx((0.1 + 0.3 * w) / (1 + w), abs=1e-6) and 0.1 < a1[0] < 0.2


def test_actions_are_made_valid():
    a = P.finish(np.array([3.0, 4.0, 0.0, 0.4, 0.2]))
    assert np.linalg.norm(a[:3]) == pytest.approx(0.25) and a[3] == 0.0 and a[4] == pytest.approx(0.2)
    assert P.finish(np.array([0, 0, 0, 0.9, 0]))[3] == 1.0


def _world():
    lay = Layout()
    lay.props = [{"name": "cube", "shape": "box", "pos": (0.0, -0.05), "yaw": 0.0, "size": (0.025, 0.02, 0.025), "rgb": (0.8, 0.2, 0.2)},
                 {"name": "mark", "shape": "box", "pos": (0.2, 0.0), "yaw": 0.0, "size": (0.02, 0.02, 0.02), "rgb": (0.2, 0.3, 0.9)}]
    return World(lay)


def test_the_executor_runs_a_policy_and_stops_at_the_time_limit():
    w = _world()
    w.settle(20)
    task = {"object": "prop_0", "name": "cube", "goal": [0.13, 0.0], "h": 0.025, "m": {"width": 0.04, "height": 0.05, "length": 0.05}}
    ex = P.LearnedExecutor(tiny(), task, max_ticks=15)
    n = 0
    while (a := ex.action(w)) is not None:
        assert a.shape == (5,)
        w.step(a)
        n += 1
    assert n == 15


def test_the_executor_stops_when_the_goal_is_met_with_the_jaws_open():
    w = _world()
    w.settle(20)
    w.set_obj_pose("prop_0", (0.1, 0.0))                  # already next to the mark: the goal is met where it stands
    w.settle(10)
    here = w.obj_pos("prop_0")
    task = {"object": "prop_0", "name": "cube", "goal": [float(here[0]), float(here[1])], "h": 0.025, "plan": {"goal": ("near", 1, "next to")},
            "ref": "prop_1", "m": {"width": 0.04, "height": 0.05, "length": 0.05}}
    ex = P.LearnedExecutor(tiny(), task, max_ticks=400)
    n = 0
    while ex.action(w) is not None:
        w.step(np.zeros(5))
        n += 1
    assert 20 < n < 60


def _torch_python():
    py = os.environ.get("PERCEPTION_PY")
    return py if py and Path(py).exists() else None


@pytest.mark.skipif(_torch_python() is None, reason="needs PERCEPTION_PY (a Python with torch)")
def test_training_reduces_the_loss_and_the_numpy_policy_matches_torch(tmp_path):
    # a toy task: the action is a fixed function of the observation, so the net can learn it
    rng = np.random.default_rng(0)
    n, eps = 576, 12
    obs = np.repeat(rng.normal(size=(eps * 4, O.OBS_DIM)).astype(np.float32), n // (eps * 4), axis=0)      # steady inside a stretch
    act = np.stack([obs[:, 0] * 0.1, obs[:, 1] * 0.1, np.zeros(n), (obs[:, 2] > 0).astype(float), np.zeros(n)], 1).astype(np.float32)
    ep = np.repeat(np.arange(eps), n // eps).astype(np.int32)       # 48 steps each, four stretches per episode
    d = tmp_path / "demos"
    d.mkdir()
    np.savez(d / "shard_0000.npz", obs=obs, act=act, episode=ep, source=np.array(["sim"]))
    out = tmp_path / "pol.npz"
    root = Path(__file__).resolve().parents[2]
    r = subprocess.run([_torch_python(), "-m", "echotwin.robot.train_policy", str(d), "--out", str(out), "--epochs", "100", "--chunk", "4",
                        "--hidden", "64"], cwd=root, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-800:]
    import json
    m = json.loads(out.with_suffix(".json").read_text())
    assert m["train_l1"] < 0.3 and m["episodes"] == eps
    pol = P.ChunkPolicy.load(out)
    pred = pol.predict(obs[3])
    assert abs(pred[0, 0] - act[3, 0]) < 0.15


# ---------------- plugging it in ----------------
def test_waypoint_action_hands_over_to_a_policy():
    from echotwin.robot.features import prop_skills as PS
    w = _world()
    w.settle(20)
    task = {"object": "prop_0", "name": "cube", "goal": [0.13, 0.0], "h": 0.025, "m": {"width": 0.04, "height": 0.05, "length": 0.05}}
    chunk = np.zeros((K, 5), np.float32)
    chunk[:, 0] = 0.1
    r = {"wps": [], "i": 0, "policy": P.LearnedExecutor(tiny(chunk), task, max_ticks=5)}
    a = PS.waypoint_action(w, r)
    assert a is not None and a[0] == pytest.approx(0.1)
    for _ in range(4):
        PS.waypoint_action(w, r)
    assert PS.waypoint_action(w, r) is None


def test_skillcheck_runs_a_trial_with_a_policy():
    from echotwin.robot import skillcheck as S
    out = S.trial("box", "next to", np.random.default_rng(3), policy=tiny())
    assert out["ok"] in (True, False) and out["seconds"] > 0


def test_pol_from_env(tmp_path):
    pol = tiny()
    np.savez(tmp_path / "p.npz", **pol.p)
    got, note = P.from_env({"POLICY": str(tmp_path / "p.npz")})
    assert got is not None and "p.npz" in note
    assert P.from_env({}) == (None, "")
    none, why = P.from_env({"POLICY": str(tmp_path / "missing.npz")})
    assert none is None and "could not be loaded" in why


def test_the_app_moves_with_the_policy_when_one_is_set(monkeypatch, tmp_path):
    import echotwin.robot.dataset as DS
    from echotwin.robot.features import move_things as MT
    from echotwin.robot.features import prop_skills as PS
    from echotwin.robot.sim import Sim
    np.savez(tmp_path / "p.npz", **tiny().p)
    monkeypatch.setenv("POLICY", str(tmp_path / "p.npz"))
    (tmp_path / "episodes").mkdir()
    monkeypatch.setattr(DS.Dataset.__init__, "__defaults__", (tmp_path / "episodes",))
    s = Sim(lambda m: None)
    assert s.policy is not None
    s._rebuild(_world().layout)
    s.base_layout = s.world.layout.copy()
    s.world.settle(20)
    # the policy path replaces the scripted steps when the app starts a move: the replay carries the executor
    s.skills.demos = [{"grip": 1.0, "lift": 0.05, "drop": 0.004, "speed": 0.2, "dist": 0.15,
                       "m": {"width": 0.04, "height": 0.05, "length": 0.05, "mass": 0.1}}]
    text = "put the cube next to the mark"
    s.handle_prop_task(MT.parse(text, s.world.layout.props), text)
    assert s.replay is not None and isinstance(s.replay.get("policy"), P.LearnedExecutor) and s.mode == "move"
