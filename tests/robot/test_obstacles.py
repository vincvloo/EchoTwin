"""Fixed obstacles in the twin: files, physics, and keeping goals clear of them."""
import json
import tempfile
from pathlib import Path

import numpy as np

from echotwin.robot import twin_import as TI
from echotwin.robot.features import move_things as MT
from echotwin.robot.world import World
from echotwin.scene import schema, to_twin

DOC = {"format": "phone-puppeteer-twin", "sim_scale": 2.0,
       "objects": [{"name": "mug", "shape": "cylinder", "size_cm": [8, 8, 10], "pos_cm": [-20, 0]},
                   {"name": "book", "shape": "flat", "size_cm": [15, 10, 2], "pos_cm": [-5, 15]}],
       "obstacles": [{"name": "sofa", "shape": "box", "size_cm": [20, 10, 30], "pos_cm": [20, 0], "color": "#34a853"}]}


def _layout():
    return TI.doc_to_layout(json.loads(json.dumps(DOC)), Path(tempfile.mkdtemp()))


def test_table_aspect_matches_the_twin_builder():
    from echotwin.robot.scene import TABLE_ASPECT
    assert abs(TABLE_ASPECT - to_twin.TABLE_ASPECT) < 1e-9


def test_obstacle_is_solid_and_survives_a_file_round_trip():
    lay = _layout()
    assert lay.obstacles[0]["name"] == "sofa"
    w = World(lay)
    w.settle(10)
    assert "obstacle_0" in [w.model.geom(i).name for i in range(w.model.ngeom)]
    doc = TI.layout_to_doc(lay, w, Path(tempfile.mkdtemp()))
    assert doc["obstacles"][0]["size_cm"] == [20.0, 10.0, 30.0] and doc["obstacles"][0]["pos_cm"] == [20.0, 0.0]
    assert w.tallest() >= 0.3 * 0.99              # the sofa (30 cm) counts as the tallest thing


def test_old_files_without_obstacles_still_load():
    doc = {k: v for k, v in DOC.items() if k != "obstacles"}
    lay = TI.doc_to_layout(doc, Path(tempfile.mkdtemp()))
    assert lay.obstacles == []


def test_goals_stay_clear_of_obstacles():
    w = World(_layout())
    w.settle(10)
    sofa = np.array(w.layout.obstacles[0]["pos"])
    for goal in (("dir", (1, 0), 0.5), ("center",)):
        g = MT.goal_xy(w, {"prop": 0, "goal": goal})
        reach = w.radius("prop_0") + max(w.layout.obstacles[0]["size"][:2])
        assert np.linalg.norm(g - sofa) >= reach - 1e-6, (goal, g)


def test_lounge_scene_loads_into_the_simulator():
    from tests.scene.test_to_twin import LOUNGE, MAP
    twin = to_twin.scene_to_twin(schema.build_scene(LOUNGE, MAP))
    w = World(TI.doc_to_layout(twin, Path(tempfile.mkdtemp())))
    w.settle(20)
    assert len(w.things()) == 2 and w.layout.obstacles
    for n in w.things():
        p = w.obj_pos(n)
        assert abs(p[0]) < w.layout.table_half[0] and abs(p[1]) < w.layout.table_half[1] and p[2] > 0     # on the table
