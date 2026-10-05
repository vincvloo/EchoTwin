"""Demonstrations for a learned policy: the observation, the bulk collector, the converter."""
import json

import numpy as np
import pytest

from echotwin.robot import demos as DM
from echotwin.robot import policy_obs as O
from echotwin.robot.scene import Layout
from echotwin.robot.world import World


def _world():
    lay = Layout()
    lay.props = [{"name": "cube", "shape": "box", "pos": (0.0, -0.05), "yaw": 0.0, "size": (0.025, 0.02, 0.025), "rgb": (0.8, 0.2, 0.2)},
                 {"name": "mark", "shape": "box", "pos": (0.2, 0.0), "yaw": 0.0, "size": (0.02, 0.02, 0.02), "rgb": (0.2, 0.3, 0.9)}]
    return World(lay)


def test_observation_has_the_documented_layout():
    w = _world()
    w.settle(20)
    task = {"object": "prop_0", "goal": [0.13, 0.0], "stack": True, "ref": "prop_1", "m": {"width": 0.04, "height": 0.05, "length": 0.05}}
    o = O.from_world(w, task, 40)
    assert o.shape == (O.OBS_DIM,) and len(O.OBS_NAMES) == O.OBS_DIM
    assert np.allclose(o[0:3], w.hand_pos(), atol=1e-6) and np.allclose(o[4:7], w.obj_pos("prop_0"), atol=1e-6)
    assert o[7:9] == pytest.approx([0.13, 0.0]) and o[9] == 0 and o[10:13] == pytest.approx([0.04, 0.05, 0.05])
    assert o[13] == 1 and o[16] == pytest.approx(0.1) and o[17] == pytest.approx(w.obj_pos("prop_1")[2] + w.half("prop_1"))
    assert o[14] ** 2 + o[15] ** 2 == pytest.approx(1.0)


def test_a_stored_episode_gives_the_same_observation_as_the_world_did():
    w = _world()
    w.settle(20)
    from echotwin.robot.dataset import state_vector
    task = {"object": "prop_0", "goal": [0.13, 0.0], "m": {"width": 0.04, "height": 0.05, "length": 0.05}, "kind": "prop"}
    frames = [{"timestamp": 0.0, "state": state_vector(w, "prop_0", [0.13, 0.0]).tolist() if hasattr(state_vector(w, "prop_0", [0.13, 0.0]), "tolist") else list(state_vector(w, "prop_0", [0.13, 0.0])),
               "action": [0.1, 0.0, 0.0, 0.0, 0.3]}] * 6
    obs, acts, legacy = O.from_episode({"task": task, "frames": frames})
    live = O.from_world(w, task, 0)
    assert not legacy and acts.shape == (6, 5)
    assert np.allclose(obs[0][:13], live[:13], atol=1e-5) and obs[0][14] == pytest.approx(np.sin(0.3))


def test_old_four_number_episodes_are_filled_in_and_flagged():
    task = {"object": "prop_0", "goal": [0.1, 0.0], "shape": "box", "h": 0.02, "kind": "prop"}      # no measurements, no yaw
    frames = [{"timestamp": 0.05 * i, "state": [0, 0, 0.1, 0, 0.1, 0, 0.02, 0.1, 0, 0], "action": [0.1, 0, 0, 0]} for i in range(8)]
    obs, acts, legacy = O.from_episode({"task": task, "frames": frames})
    assert legacy and acts.shape == (8, 5) and np.all(acts[:, 4] == 0) and obs.shape == (8, O.OBS_DIM)


def test_the_collector_keeps_only_moves_that_worked_and_records_the_clean_action(tmp_path):
    info = DM.collect(3, tmp_path, noise=0.5, seed=1, shard=2, log=lambda *_: None)
    assert info["episodes"] == 3 and info["shards"] == 2
    z = np.load(tmp_path / "shard_0000.npz")
    assert z["obs"].shape[1] == O.OBS_DIM and z["act"].shape[1] == O.ACT_DIM and len(z["obs"]) == len(z["act"]) == len(z["episode"])
    assert set(np.unique(z["act"][:, 3])) <= {0.0, 1.0}                  # the recorded grip is the expert's 0 or 1, not a noisy value
    assert np.abs(z["act"][:, :3]).max() <= 0.2501                        # and the recorded velocities are the expert's (capped by its speed)


def test_stored_json_demos_are_converted_with_their_source(tmp_path):
    eps = tmp_path / "eps"
    eps.mkdir()
    frames = [{"timestamp": 0.05 * i, "state": [0, 0, 0.1, 0, 0.1, 0, 0.02, 0.1, 0, 0], "action": [0.1, 0, 0, 0]} for i in range(8)]
    for i, src in enumerate(("human", "practice")):
        (eps / f"ep_{i:06d}.json").write_text(json.dumps({"episode_index": i, "source": src, "success": True,
                                                         "task": {"kind": "prop", "object": "prop_0", "h": 0.02, "shape": "box"}, "frames": frames}))
    info = DM.convert(eps, tmp_path / "out")
    assert info["episodes"] == 2 and info["sources"] == {"human": 1, "practice": 1} and info["legacy"] == 2
    z = np.load(tmp_path / "out" / "shard_0000.npz")
    assert len(z["obs"]) == 16 and list(z["source"]) == ["human", "practice"]


def test_featurize_adds_the_offsets_the_network_needs():
    obs = np.zeros(O.OBS_DIM, np.float32)
    obs[0:3] = [0.1, 0.2, 0.3]            # tool
    obs[4:7] = [0.4, 0.5, 0.6]            # object
    obs[7:9] = [0.7, 0.8]                 # goal
    obs[17] = 0.9                         # top of the stack
    f = O.featurize(obs)
    assert f.shape == (O.FEAT_DIM,) and np.allclose(f[:18], obs)
    assert np.allclose(f[18:21], [0.3, 0.3, 0.3]) and np.allclose(f[21:23], [0.6, 0.6]) and np.allclose(f[23:25], [0.3, 0.3])
    assert f[25] == pytest.approx(0.6)
    assert O.featurize(np.stack([obs, obs])).shape == (2, O.FEAT_DIM)
