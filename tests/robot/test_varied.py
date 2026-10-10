"""The skill check's varied objects, and its per-cell draws."""
from pathlib import Path

import numpy as np

from echotwin.robot import skillcheck as S
from echotwin.robot import varied
from echotwin.robot.scene import Layout
from echotwin.robot.world import World


def test_varied_draws_every_kind_and_builds_in_the_twin(tmp_path, monkeypatch):
    monkeypatch.setattr(varied, "MESH_DIR", tmp_path)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(60):
        p = varied.draw(rng)
        seen.add(p["kind"])
        assert 0.0 <= p["yaw"] <= 180.0 and min(p["size"]) > 0
        if p.get("mesh"):
            assert Path(p["mesh"]).exists()
    assert seen == set(varied.KINDS)
    lay = Layout()
    lay.props = [dict(varied.draw(np.random.default_rng(k)), pos=(0.05 * k - 0.1, 0.0)) for k in range(5)]
    w = World(lay)                                           # every kind compiles into the twin
    assert all(w.grasp_width(n) > 0 for n in w.things())


def test_a_cell_draws_the_same_layouts_whatever_runs_before_it():
    a = S.cell("box", "next to", 2, seed=3)
    S.cell("flat", "to the left", 1, seed=3)
    b = S.cell("box", "next to", 2, seed=3)
    assert a["success"] == b["success"] and a["failures"] == b["failures"] and a["kinds"] == {"box": [round(a["success"] * 2), 2]}
