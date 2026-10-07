"""Things on a table, on the floor or on something else: the guess, the twin, the sizes, the correction."""
import asyncio
import itertools
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from echotwin.robot import answers, hub, server
from echotwin.robot.features import everyday as E
from echotwin.robot.scene import DEFAULT_SURFACE, Layout, build_xml
from echotwin.robot.twin_import import layout_file as LF
from echotwin.robot.twin_import import photos as P
from echotwin.robot.twin_import.contract import TwinContext
from echotwin.robot.world import World
from echotwin.scene.to_twin import scene_to_twin, surface_of

CUBE = {"name": "cube", "shape": "box", "pos": (0.05, 0.05), "yaw": 0.0, "size": (0.02, 0.02, 0.02), "rgb": (0.6, 0.4, 0.3)}
FLOOR, BENCH = {"kind": "floor", "height": 0.0}, {"kind": "other", "height": 0.45}


def _lay(surface=None, *props):
    lay = Layout(surface=dict(surface or DEFAULT_SURFACE))
    lay.props = [dict(p) for p in props]
    return lay


def test_a_table_is_drawn_as_before_and_a_floor_has_no_table():
    table, floor, bench = build_xml(_lay()), build_xml(_lay(FLOOR)), build_xml(_lay(BENCH))
    assert 'name="table"' in table and 'pos="0 0 -0.7500"' in table and 'size="0.0250 0.0250 0.3500"' in table
    assert 'name="table"' not in floor and 'pos="0 0 -0.0000"' in floor and "0.0250 0.0250" not in floor
    assert 'name="table"' in bench and 'pos="0 0 -0.4500"' in bench and 'size="0.0250 0.0250 0.2000"' in bench


@pytest.mark.parametrize("surface", [DEFAULT_SURFACE, FLOOR, BENCH])
def test_things_rest_on_any_surface(surface):
    w = World(_lay(surface, CUBE))
    w.settle(60)
    assert w.obj_pos("prop_0")[2] == pytest.approx(0.02, abs=0.003)
    assert w.layout.surface == surface


def test_the_surface_is_saved_and_loaded_with_the_twin(tmp_path):
    lay = _lay(BENCH, CUBE)
    doc = LF.layout_to_doc(lay, None, tmp_path)
    assert doc["surface"] == {"kind": "other", "height_cm": 45.0}
    assert LF.doc_to_layout(doc, tmp_path).surface == BENCH
    assert LF.doc_to_layout({**doc, "surface": {"kind": "floor", "height_cm": 70}}, tmp_path).surface == FLOOR
    for odd in (None, "floor", {"kind": "roof"}, {"kind": "other", "height_cm": "high"}):
        assert LF.doc_to_layout({**doc, "surface": odd}, tmp_path).surface == DEFAULT_SURFACE


def test_a_3d_scan_says_what_the_things_stand_on():
    table = {"class": "dining table", "label": "dining table", "height": 0.74}
    shelf = {"class": "shelf", "label": "shelf", "height": 1.1}
    assert surface_of([{"base_z": 0.74}], table) == {"kind": "table", "height_cm": 74.0}
    assert surface_of([{"base_z": 1.1}], shelf) == {"kind": "other", "height_cm": 110.0}
    assert surface_of([{"base_z": 0.0}, {"base_z": 0.02}, {"base_z": 0.5}], None) == {"kind": "floor", "height_cm": 0.0}
    assert surface_of([{"base_z": 0.42}, {"base_z": 0.45}], None) == {"kind": "other", "height_cm": 45.0}


def test_the_phone_height_follows_the_surface():
    assert E.phone_height(DEFAULT_SURFACE) == E.CAM_HEIGHT == 0.45
    assert E.phone_height(FLOOR) == 1.2
    assert E.phone_height(BENCH) == pytest.approx(0.8)
    assert E.phone_height({"kind": "other", "height": 1.8}) == 0.35 and E.phone_height(None) == 0.45


def test_moving_the_same_photo_to_the_floor_scales_everything_by_the_phone_heights():
    lay = _lay(None, CUBE)
    lay.meta["scale"] = 0.9                                                       # a rescale the user made stays
    f = E.on_surface(lay, FLOOR)
    k = 1.2 / 0.45
    assert f.surface == FLOOR and f.props[0]["size"] == pytest.approx(tuple(v * k for v in CUBE["size"]))
    assert f.props[0]["pos"] == pytest.approx(tuple(v * k for v in CUBE["pos"])) and f.meta["scale"] == 0.9
    assert E.on_surface(f, DEFAULT_SURFACE).props[0]["size"] == pytest.approx(CUBE["size"])     # and back
    assert E.surface_from_ai({"surface": "floor"}) == FLOOR and E.surface_from_ai({"surface": "moon"}) is None
    assert E.surface_from_ai(None) is None and E.surface_from_ai({"surface": "table"}) == DEFAULT_SURFACE


def _photo() -> bytes:
    img = np.full((540, 960, 3), (190, 205, 215), np.uint8)
    cv2.ellipse(img, (300, 380), (70, 55), 0, 0, 360, (40, 40, 45), -1)
    cv2.rectangle(img, (600, 340), (760, 450), (40, 40, 45), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _scan(tmp_path, answer):
    applied, done = [], []
    ids = itertools.count()

    def new_dir(prefix=""):
        d = tmp_path / f"scan{next(ids)}"
        d.mkdir(parents=True)
        return d.name, d

    async def ask(jpeg, prompt):
        return answer
    ctx = TwinContext(apply=lambda lay, s: applied.append(lay), rename=lambda props, line: applied.append(props),
                      say=lambda t: None, progress=lambda st, d: done.append(dict(d["summary"])) if st == "done" else None,
                      new_dir=new_dir, ask_ai_json=ask)
    asyncio.run(P.import_photos([_photo()], [None], ctx))
    return applied, done


def test_when_the_ai_says_floor_the_twin_is_rebuilt_for_the_floor(tmp_path):
    names = [{"id": 1, "name": "case", "shape": "round"}, {"id": 2, "name": "box", "shape": "box"}]
    on_table, _ = _scan(tmp_path / "a", {"surface": "table", "objects": names})
    on_floor, done = _scan(tmp_path / "b", {"surface": "floor", "objects": names})
    assert isinstance(on_table[-1], list)                                         # a table: names only, as before
    lay = on_floor[-1]
    assert isinstance(lay, Layout) and lay.surface == FLOOR and done[-1]["surface"] == FLOOR
    assert "the floor" in done[-1]["greeting"] and done[-1]["naming"] is False
    big = max(lay.props, key=lambda p: p["size"][0])["size"][0]
    small = max(on_table[0].props, key=lambda p: p["size"][0])["size"][0]
    assert big == pytest.approx(small * 1.2 / 0.45, rel=0.02)


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


def _load(sim, source="estimate"):
    sim._rebuild(_lay(None, CUBE))
    sim.base_layout = sim.world.layout.copy()
    sim.scan = {"mode": "everyday", "calibration": {"source": source}}


def test_the_correction_resizes_a_one_photo_estimate_and_says_why(sim):
    _load(sim)
    sim.set_surface("floor")
    assert sim.world.layout.surface == FLOOR and sim.scan["surface"] == FLOOR
    assert sim.world.layout.props[0]["size"][0] == pytest.approx(0.02 * 1.2 / 0.45)
    assert "on the floor" in sim.said[-1] and "2.7 times" in sim.said[-1]
    assert answers.about_table(sim.world, "count", {}) == "I see 1 thing on the floor."
    assert answers.about_table(sim.world, "see", {}).startswith("On the floor I see")
    sim.set_surface("other", 0.45)
    assert sim.world.layout.surface == BENCH and "45 cm high" in sim.said[-1]
    sim.set_surface("table")
    assert sim.world.layout.props[0]["size"][0] == pytest.approx(0.02) and answers.about_table(sim.world, "see", {}).startswith("On your table")


def test_measured_sizes_are_not_changed_and_odd_answers_are_refused(sim):
    _load(sim, "marker")
    sim.set_surface("floor")
    assert sim.world.layout.surface == FLOOR and sim.world.layout.props[0]["size"][0] == pytest.approx(0.02)
    sim.set_surface("roof")
    assert sim.world.layout.surface == FLOOR and "table, on the floor" in sim.said[-1]
    sim.set_surface("other", 5.0)
    assert sim.world.layout.surface == FLOOR and "between 5 cm and 2 m" in sim.said[-1]


def test_the_endpoint(monkeypatch):
    calls = []
    monkeypatch.setattr(hub.sim, "submit", lambda fn, *a: calls.append((fn.__name__, a)))
    c = TestClient(server.app)
    assert c.post("/api/twin/surface", json={"kind": "floor"}).json()["ok"] and calls[-1] == ("set_surface", ("floor", None))
    assert c.post("/api/twin/surface", json={"kind": "other", "height_cm": 45}).json()["ok"] and calls[-1] == ("set_surface", ("other", 0.45))
    for bad in ({}, {"kind": "roof"}, {"kind": "other", "height_cm": 900}, {"kind": "other", "height_cm": "x"}):
        r = c.post("/api/twin/surface", json=bad).json()
        assert r["ok"] is False and r["error"], bad
