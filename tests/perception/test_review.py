"""Photo side of the review: which photos to send, how objects are marked, the whole loop with a fake model."""
import numpy as np

from echotwin.perception import objects as OB
from echotwin.perception import review as PR
from echotwin.scene import schema


def test_object_views_pick_best_photos_and_normalise_boxes():
    f = np.array([0] * 50 + [1] * 30 + [2] * 5 + [3] * 2)
    row = np.concatenate([np.linspace(100, 200, 50), np.linspace(10, 50, 30), np.full(7, 5)])
    col = np.concatenate([np.linspace(300, 400, 50), np.linspace(20, 60, 30), np.full(7, 5)])
    views = OB.object_views(f, row, col, (400, 500), max_views=2)
    assert [v["frame"] for v in views] == [0, 1]
    x0, y0, x1, y1 = views[0]["box"]
    assert 0.59 < x0 < 0.63 and 0.79 < x1 < 0.83 and 0.25 < y0 < 0.3 and 0.49 < y1 < 0.52


def _scene_with_views():
    objs = [
        {"class": "chair", "x": 0, "y": 0, "size_x": .5, "size_y": .5, "height": .5,
         "views": [{"frame": 0, "box": [.1, .1, .3, .4]}, {"frame": 1, "box": [.5, .5, .7, .8]}]},
        {"class": "couch", "x": 2, "y": 0, "size_x": 1.5, "size_y": .8, "height": .8,
         "views": [{"frame": 0, "box": [.4, .1, .9, .5]}, {"frame": 1, "box": [.1, .5, .6, .9]}]},
        {"class": "cup", "x": 4, "y": 0, "size_x": .08, "size_y": .08, "height": .1,
         "views": [{"frame": 2, "box": [.2, .2, .3, .3]}]},
    ]
    return schema.build_scene(objs, name="t")


def test_select_frames_gives_every_object_a_second_opinion_where_possible():
    sc = _scene_with_views()
    chosen = PR.select_frames(sc, max_frames=2)
    assert set(chosen) == {0, 1}                                # frames 0 and 1 give the chair two views, 0 the couch
    assert {n for marks in chosen.values() for n, _ in marks} == {1, 2}
    assert set(PR.select_frames(sc, max_frames=4)) == {0, 1, 2}       # budget left: the cup is covered too


def test_mark_frame_draws_boxes_on_a_copy():
    img = np.full((600, 800, 3), 120, np.uint8)
    im = PR.mark_frame(img, [(1, [.1, .1, .5, .5])])
    assert max(im.size) == PR.SEND_SIDE
    assert (img == 120).all()                                    # the input is untouched
    arr = np.asarray(im)
    assert (arr != 120).any() and (arr[:, :, 0] > 200).any()     # yellow pixels


def test_review_loop_with_a_fake_model(tmp_path):
    sc = _scene_with_views()
    frames = {i: np.full((480, 640, 3), 90 + i, np.uint8) for i in range(3)}
    seen = []

    def fake(jpg, prompt, cfg):
        seen.append(prompt)
        ids = [int(l.split(":")[0]) for l in prompt.splitlines() if l[:1].isdigit() and ":" in l[:4]]
        return {"objects": [{"id": n, "name": "sofa" if n == 2 else "chair", "shape": "box", "movable": False,
                             "traits": ["soft"] if n == 2 else [], "skip": n == 3} for n in ids]}

    rep = PR.review_scene(sc, frames, tmp_path / "rev", max_frames=3,
                          cfg={"key": "k", "models": ["m"], "provider": "nvidia"}, ask=fake)
    by = {o["id"]: o for o in sc["objects"]}
    assert len(seen) == 3 and rep["reviewed"] == 3
    assert "o3" in by and by["o2"]["label"] == "sofa" and by["o2"]["traits"] == ["soft"]      # one skip vote: kept
    assert sorted(p.name for p in (tmp_path / "rev").glob("*.jpg")) == ["photo_00.jpg", "photo_01.jpg", "photo_02.jpg"]


def test_review_without_a_key_changes_nothing(tmp_path):
    sc = _scene_with_views()
    before = [o["label"] for o in sc["objects"]]
    out = PR.review_scene(sc, {0: np.zeros((10, 10, 3), np.uint8)}, None, cfg={"key": "", "models": [], "provider": "nvidia"})
    assert out is None and [o["label"] for o in sc["objects"]] == before
