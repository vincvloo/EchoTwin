"""Scoring of the detector benchmark (no models, no GPU)."""
import json

from echotwin.perception import bench_detect as B
from echotwin.scene import catalog

SCENE = {
    "items": [
        {"id": "sofa", "names": ["couch", "sofa"], "min_photos": 2},
        {"id": "plant", "names": ["potted plant", "plant"], "min_photos": 2},
        {"id": "speaker", "names": ["speaker"], "min_photos": 1},
    ],
    "ignore": ["person", "wall"],
}


def test_recall_needs_enough_photos_and_any_accepted_name():
    dets = {"a": [("couch", .9), ("plant", .8)], "b": [("sofa", .7), ("potted plant", .6)], "c": [("couch", .9)]}
    m = B.score_scene(SCENE, dets)
    assert m["found"] == ["sofa", "plant"] and m["missing"] == ["speaker"] and abs(m["recall"] - 2 / 3) < 1e-9
    assert m["photos_seen"] == {"sofa": 3, "plant": 2, "speaker": 0}
    one_photo = B.score_scene(SCENE, {"a": [("couch", .9)], "b": []})
    assert "sofa" in one_photo["missing"]                       # seen in only one photo: not enough


def test_low_confidence_detections_are_ignored():
    m = B.score_scene(SCENE, {"a": [("speaker", .2)], "b": []}, conf=0.25)
    assert m["missing"] == ["sofa", "plant", "speaker"] and m["detections"] == 0
    assert B.score_scene(SCENE, {"a": [("speaker", .3)]}, conf=0.25)["found"] == ["speaker"]


def test_false_rate_and_false_names_skip_the_ignore_list():
    dets = {"a": [("couch", .9), ("person", .9), ("banana", .9), ("banana", .8)],
            "b": [("couch", .9), ("banana", .9), ("kite", .9)]}
    m = B.score_scene(SCENE, dets)
    assert m["detections"] == 7 and abs(m["false_rate"] - 4 / 7) < 1e-9       # banana x3 + kite; person is ignored
    assert m["false_names"] == ["banana"]                                     # kite appears in only one photo
    assert abs(m["per_photo"] - 3.5) < 1e-9


def test_names_are_case_insensitive_and_summary_is_a_table():
    m = B.score_scene(SCENE, {"a": [("Couch", .9)], "b": [("COUCH", .9)]})
    assert "sofa" in m["found"]
    table = B.summarize({"m1": {"scenes": {"lounge": m}, "ms": 12.3, "gpu_mb": 456.0}})
    assert table.splitlines()[0].startswith("| Model | lounge: recall, false |") and "| m1 | 33%, 0% | 12 | 456 |" in table


def test_the_hand_written_ground_truth_is_well_formed():
    cfg = json.loads(B.DATA.read_text(encoding="utf-8"))
    assert set(cfg["scenes"]) == {"lounge_photos", "table_photos"} and len(catalog.PROMPTS) >= 40
    assert len(set(catalog.PROMPTS)) == len(catalog.PROMPTS)                              # no duplicate prompts
    for name, scene in cfg["scenes"].items():
        ids = [i["id"] for i in scene["items"]]
        assert len(ids) == len(set(ids)) and all(i["names"] and i.get("min_photos", 2) >= 1 for i in scene["items"])
        assert len(list((B.REPO / scene["folder"]).glob("*.jpg"))) >= 10                  # the example photos exist


def test_one_wrong_name_cannot_satisfy_several_items():
    scene = {"items": [{"id": "chocolate", "names": ["candy", "remote"], "min_photos": 2},
                       {"id": "case", "names": ["case", "remote"], "min_photos": 2},
                       {"id": "charger", "names": ["charger", "remote"], "min_photos": 2}], "ignore": []}
    dets = {"a": [("remote", .9)], "b": [("remote", .9)]}
    m = B.score_scene(scene, dets)
    assert len(m["found"]) == 1 and len(m["missing"]) == 2                   # "remote" counts once, not three times
    dets["a"].append(("charger", .9))
    dets["b"].append(("charger", .9))
    m = B.score_scene(scene, dets)
    assert "charger" in m["found"] and len(m["found"]) == 2                  # the charger keeps its own name; remote goes to another
