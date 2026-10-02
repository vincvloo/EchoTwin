"""Review of detections: reading the model's answer, voting, applying it, and failing quietly."""
import json

from echotwin.scene import review as R
from echotwin.scene import schema


def _scene():
    objs = [
        {"class": "chair", "x": 0.0, "y": 0.0, "size_x": 2.1, "size_y": 0.99, "height": 0.71},     # really a sofa
        {"class": "potted plant", "x": 2.0, "y": 0.0, "size_x": 0.24, "size_y": 0.36, "height": 0.25},
        {"class": "potted plant", "x": 3.0, "y": 0.0, "size_x": 0.3, "size_y": 0.3, "height": 0.2},   # a reflection
        {"class": "cup", "x": 4.0, "y": 0.0, "size_x": 0.08, "size_y": 0.08, "height": 0.1},
    ]
    return schema.build_scene(objs, name="t")


def _answer(*entries):
    return {"objects": [dict(id=i, **e) for i, e in enumerate(entries, 1)]}


def test_parse_answer_keeps_valid_fields_only():
    got = R.parse_answer(_answer(
        {"name": "  big   sofa ", "shape": "box", "movable": False, "traits": ["soft", "sticky"], "skip": False},
        {"name": "x", "shape": "banana", "movable": "yes", "traits": "nope"},
    ))
    assert got[1] == {"name": "big sofa", "shape": "box", "movable": False, "traits": ["soft"], "skip": False}
    assert got[2]["shape"] is None and got[2]["movable"] is None and got[2]["traits"] == []
    assert R.parse_answer(None) == {} and R.parse_answer({"objects": [{"no": "id"}]}) == {}


def test_vote_over_photos():
    a = {"name": "Sofa", "shape": "box", "movable": False, "traits": ["soft"], "skip": False}
    b = {"name": "sofa", "shape": "box", "movable": False, "traits": [], "skip": False}
    c = {"name": "armchair", "shape": "box", "movable": None, "traits": ["soft"], "skip": False}
    v = R.vote([a, b, c])
    assert v["name"] == "sofa" and v["movable"] is False and v["traits"] == ["soft"] and not v["skip"]
    # one photo alone changes nothing; two that disagree give no winner
    assert R.vote([a])["name"] is None and R.vote([a])["shape"] is None and R.vote([a])["movable"] is None
    assert R.vote([a, c])["name"] is None and R.vote([a, c])["traits"] == ["soft"]    # names disagree; trait agreed
    skip = {"name": None, "shape": None, "movable": None, "traits": [], "skip": True}
    assert R.vote([skip, skip, a])["skip"] and not R.vote([skip, a])["skip"]      # needs a majority
    assert not R.vote([skip])["skip"]                                             # one photo never removes


def test_apply_review_renames_removes_and_cannot_make_big_things_movable():
    sc = _scene()
    verdicts = {
        "o1": R.vote([{"name": "sofa", "shape": "box", "movable": True, "traits": ["soft"], "skip": False}] * 2),
        "o3": R.vote([{"name": None, "shape": None, "movable": None, "traits": [], "skip": True}] * 2),
        "o4": R.vote([{"name": "mug", "shape": "cylinder", "movable": True, "traits": ["fragile", "hollow"],
                       "skip": False}] * 2),
    }
    rep = R.apply_review(sc, verdicts, model="m")
    by = {o["id"]: o for o in sc["objects"]}
    assert "o3" not in by and rep["removed"] == [{"id": "o3", "class": "potted plant"}]
    assert by["o1"]["label"] == "sofa" and by["o1"]["surface"] and not by["o1"]["movable"]   # 2 m wide: stays furniture
    assert by["o1"]["source"] == "nvidia" and by["o1"]["traits"] == ["soft"] and by["o1"]["class"] == "chair"
    assert by["o4"]["label"] == "mug" and by["o4"]["movable"] and by["o4"]["traits"] == ["fragile", "hollow"]
    assert by["o2"]["source"] == "yolo"                                                     # not reviewed: untouched
    assert sc["review"]["model"] == "m" and ["chair", "sofa"] in sc["review"]["renamed"]


def test_reviewer_can_say_a_small_thing_is_not_movable():
    sc = _scene()
    R.apply_review(sc, {"o2": R.vote([{"name": "potted plant", "shape": "cylinder", "movable": False,
                                       "traits": [], "skip": False}] * 2)})
    assert not {o["id"]: o for o in sc["objects"]}["o2"]["movable"]


def test_support_is_recomputed_after_removing_a_surface():
    sc = schema.build_scene([
        {"class": "dining table", "x": 0, "y": 0, "size_x": 1.2, "size_y": 0.7, "height": 0.74},
        {"class": "cup", "x": 0.1, "y": 0.0, "size_x": 0.08, "size_y": 0.08, "height": 0.1, "base_z": 0.74}])
    assert sc["objects"][1]["on"] == "o1"
    R.apply_review(sc, {"o1": R.vote([{"name": None, "shape": None, "movable": None, "traits": [], "skip": True}] * 2)})
    assert sc["objects"][0]["on"] is None and len(sc["objects"]) == 1


def test_no_key_means_no_call():
    called = []
    out = R.ask_json(b"jpg", "q", cfg={"key": "", "models": ["m"], "provider": "nvidia"},
                     post=lambda *a, **k: called.append(1))
    assert out is None and not called


def test_ask_json_parses_the_reply_and_falls_back_to_the_next_model():
    calls = []

    def post(url, headers, payload):
        calls.append(payload["model"])
        if payload["model"] == "down":
            raise OSError("model is down")
        text = '<think>hmm</think>sure: {"objects": [{"id": 1, "name": "sofa"}]}'
        return {"choices": [{"message": {"content": text}}]}

    cfg = {"key": "k", "models": ["down", "up"], "provider": "nvidia"}
    out = R.ask_json(b"jpg", "q", cfg=cfg, post=post, sleep=lambda s: None)
    assert out["objects"][0]["name"] == "sofa" and calls == ["down", "up"]
    bad = R.ask_json(b"jpg", "q", cfg=cfg, post=lambda *a: {"choices": [{"message": {"content": "no json"}}]},
                     sleep=lambda s: None)
    assert bad is None


def test_prompt_lists_the_detectors_guesses():
    p = R.build_prompt([(1, "chair"), (4, "potted plant")])
    assert "1: chair" in p and "4: potted plant" in p and json.loads(p[p.index('{"objects"'):p.index("\nname:")])
