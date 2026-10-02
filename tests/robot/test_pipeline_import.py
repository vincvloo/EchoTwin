"""Phone photos -> twin through the 3D pipeline: the way in, and every way out to quick mode."""
import asyncio
import sys

import pytest

from echotwin.perception import pipeline as PL
from echotwin.robot.twin_import import TwinContext
from echotwin.robot.twin_import import pipeline as P
from echotwin.scene import schema

PHOTOS = [b"jpg"] * 5
CFG = {"gpu_py": sys.executable, "maps_py": sys.executable, "mode": "auto", "frames": 12, "cam_height": 0.45,
       "min_photos": 3}


class Rec:
    """A TwinContext that remembers what the importer did."""

    def __init__(self, tmp_path, skip=None):
        self.events, self.said, self.applied, self.quick = [], [], [], []
        self.ctx = TwinContext(apply=lambda lay, s: self.applied.append((lay, s)), rename=lambda *a: None,
                               say=self.said.append, progress=lambda st, d: self.events.append((st, d)),
                               new_dir=lambda: ("sid", tmp_path), skip=skip)


@pytest.fixture(autouse=True)
def fake_quick(monkeypatch):
    async def quick(frames, pitches, ctx):
        ctx.said_quick = True
        ctx.say("QUICK")
    monkeypatch.setattr(P.photos, "import_photos", quick)
    monkeypatch.setattr(P, "_has_key", lambda: False)


def _scene(tmp_path, objs):
    schema.save(schema.build_scene(objs, name="t"), tmp_path / "objects_scene.json")


TABLE = [{"class": "dining table", "x": 0, "y": 0, "size_x": 1.2, "size_y": 0.7, "height": 0.74},
         {"class": "cup", "x": -0.2, "y": 0.1, "size_x": 0.08, "size_y": 0.08, "height": 0.1, "base_z": 0.74},
         {"class": "book", "x": 0.2, "y": -0.1, "size_x": 0.2, "size_y": 0.15, "height": 0.03, "base_z": 0.74}]


def _runner(tmp_path, results=None, scene=TABLE):
    """A fake for run_step: records the commands, gives each step a scripted result."""
    calls = []

    async def run(argv, cwd, skip, tick, env=None):
        i = len(calls)
        calls.append(argv)
        state, out = (results or {}).get(i, ("done", ""))
        if i == 0 and state == "done":
            out = "wrote cloud\ncamera height ... --scale 0.512  (then fix with --ref)\n"
        if i == 2 and state == "done" and scene is not None:
            _scene(tmp_path, scene)
        return state, out
    return run, calls


def _go(tmp_path, cfg=CFG, photos=PHOTOS, **kw):
    rec = Rec(tmp_path, **kw.pop("rec", {}))
    run, calls = _runner(tmp_path, **kw)
    asyncio.run(P.import_photos_auto(photos, [], rec.ctx, cfg=cfg, run=run))
    return rec, calls


def test_full_run_builds_the_twin_and_passes_the_scale_on(tmp_path):
    (tmp_path / "input").mkdir()
    rec, calls = _go(tmp_path)
    assert len(calls) == 3                                           # no key: the review step is left out
    assert "--scale" in calls[2] and calls[2][calls[2].index("--scale") + 1] == "0.5120"
    lay, summary = rec.applied[0]
    assert sorted(p["name"] for p in lay.props) == ["book", "cup"] and summary["mode"] == "scan3d"
    assert "QUICK" not in rec.said and (tmp_path / "twin.json").exists() and (tmp_path / "twin.jpg").exists()
    assert (tmp_path / "input" / "004.jpg").exists()                 # the photos were saved for the pipeline
    pipes = [d for st, d in rec.events if st == "pipeline"]
    assert pipes[0]["state"] == "running" and pipes[-1]["state"] == "done" and pipes[-1]["pct"] == 100
    assert [d["pct"] for d in pipes] == sorted(d["pct"] for d in pipes)       # the bar never goes back


def test_review_step_runs_only_with_a_key_and_may_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "_has_key", lambda: True)
    rec, calls = _go(tmp_path, results={3: ("failed", "no network")})
    assert len(calls) == 4 and rec.applied and "QUICK" not in rec.said       # review failed: carry on


@pytest.mark.parametrize("cfg_change,photos,why", [
    ({"gpu_py": ""}, PHOTOS, "PERCEPTION_PY"),
    ({"gpu_py": "/nope/python"}, PHOTOS, "PERCEPTION_PY"),
    ({}, PHOTOS[:2], "at least 3"),
    ({"mode": "quick"}, PHOTOS, "quick"),
])
def test_cannot_run_3d_goes_straight_to_quick_mode(tmp_path, cfg_change, photos, why):
    assert why in PL.usable({**CFG, **cfg_change}, len(photos))
    rec, calls = _go(tmp_path, cfg={**CFG, **cfg_change}, photos=photos)
    assert calls == [] and rec.said[-1] == "QUICK" and not rec.applied


def test_asking_for_3d_explains_why_it_cannot(tmp_path):
    rec, _ = _go(tmp_path, cfg={**CFG, "gpu_py": "", "mode": "3d"})
    assert any("can't build a 3D model" in s for s in rec.said) and rec.said[-1] == "QUICK"


def test_skip_uses_quick_mode(tmp_path):
    rec, calls = _go(tmp_path, results={1: ("skipped", "")})
    assert len(calls) == 2 and rec.said[-2:] == ["Okay, quick mode.", "QUICK"] and not rec.applied
    assert [d["state"] for st, d in rec.events if st == "pipeline"][-1] == "skipped"


def test_a_failed_step_uses_quick_mode(tmp_path):
    rec, calls = _go(tmp_path, results={0: ("failed", "CUDA out of memory")})
    assert len(calls) == 1 and "failed at 'Building a 3D model'" in rec.said[-2] and rec.said[-1] == "QUICK"


def test_nothing_small_enough_to_move_uses_quick_mode(tmp_path):
    chairs = [{"class": "chair", "x": 0, "y": 0, "size_x": 0.5, "size_y": 0.5, "height": 0.5}]
    rec, _ = _go(tmp_path, scene=chairs)
    assert "nothing small enough" in rec.said[-2] and rec.said[-1] == "QUICK" and not rec.applied


def test_no_scene_file_uses_quick_mode(tmp_path):
    rec, _ = _go(tmp_path, scene=None)
    assert "no objects" in rec.said[-2] and rec.said[-1] == "QUICK"


# ---------- the real subprocess runner ----------
def test_run_step_reports_output_and_failure(tmp_path):
    out = asyncio.run(P.run_step([sys.executable, "-c", "print('hello')"], tmp_path, None, lambda s: None))
    assert out == ("done", "hello\n")
    state, _ = asyncio.run(P.run_step([sys.executable, "-c", "import sys; sys.exit(3)"], tmp_path, None, lambda s: None))
    assert state == "failed"


def test_run_step_stops_a_slow_command_when_skipped(tmp_path):
    async def go():
        skip = asyncio.Event()
        asyncio.get_running_loop().call_later(0.5, skip.set)
        return await P.run_step([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path, skip, lambda s: None)
    t0 = asyncio.run(go())
    assert t0[0] == "skipped"
