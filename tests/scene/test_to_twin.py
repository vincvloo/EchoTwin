"""scene.json -> twin: surface or cluster choice, real scale, movable vs obstacle."""
import json

import pytest

from echotwin.scene import schema, to_twin

# the 8 objects found in the DARE lounge photos (out/lounge_objects_objects.json), a whole room
LOUNGE = [
    {"class": "chair", "x": -1.006, "y": 1.351, "size_x": 2.1, "size_y": 0.99, "height": 0.71},
    {"class": "couch", "x": -0.108, "y": 1.616, "size_x": 1.5, "size_y": 0.78, "height": 0.78},
    {"class": "chair", "x": 1.018, "y": 0.932, "size_x": 0.81, "size_y": 0.84, "height": 0.68},
    {"class": "chair", "x": -0.266, "y": 0.446, "size_x": 0.48, "size_y": 0.48, "height": 0.48},
    {"class": "chair", "x": -1.69, "y": 0.438, "size_x": 0.54, "size_y": 0.78, "height": 0.64},
    {"class": "potted plant", "x": -2.304, "y": 0.601, "size_x": 0.45, "size_y": 0.33, "height": 0.79},
    {"class": "potted plant", "x": 1.728, "y": 1.233, "size_x": 0.24, "size_y": 0.36, "height": 0.25},
    {"class": "potted plant", "x": 1.212, "y": 1.549, "size_x": 0.33, "size_y": 0.24, "height": 0.16},
]
MAP = {"origin": [-3.26, -1.05], "res": 0.03, "width": 197, "height": 152}


def test_scene_file_roundtrip_and_validation(tmp_path):
    scene = schema.build_scene(LOUNGE, MAP, name="lounge")
    schema.save(scene, tmp_path / "s.json")
    assert schema.load(tmp_path / "s.json")["objects"][0]["id"] == "o1"
    with pytest.raises(schema.SceneError):
        schema.validate({"format": "nope"})
    bad = json.loads(json.dumps(scene))
    bad["objects"][0]["shape"] = "banana"
    with pytest.raises(schema.SceneError):
        schema.validate(bad)


def test_lounge_movable_things_are_only_the_two_small_plants():
    scene = schema.build_scene(LOUNGE, MAP)
    movable = [o["class"] + f"@{o['x']:.1f}" for o in scene["objects"] if o["movable"]]
    assert movable == ["potted plant@1.7", "potted plant@1.2"]


def test_lounge_twin_is_table_sized_with_furniture_as_obstacles():
    twin = to_twin.scene_to_twin(schema.build_scene(LOUNGE, MAP, name="lounge"))
    assert len(twin["objects"]) == 2 and all(o["shape"] == "cylinder" for o in twin["objects"])
    real_w, real_h = twin["window_m"]
    for o in twin["objects"]:
        assert abs(o["pos_cm"][0]) <= real_w * 50 and abs(o["pos_cm"][1]) <= real_h * 50   # on the table
    assert any(o["name"].startswith("chair") for o in twin["obstacles"])      # the chair next to the plants
    assert twin["left_out"] >= 3                                              # the far side of the room
    # the window is sized so that it fills the sim table
    assert abs(twin["sim_scale"] * real_w - to_twin.TABLE_SIM[0]) < 0.01


def test_things_on_a_table_pick_that_table():
    objs = [
        {"class": "dining table", "x": 0.0, "y": 0.0, "size_x": 1.2, "size_y": 0.7, "height": 0.74},
        {"class": "cup", "x": -0.2, "y": 0.1, "size_x": 0.08, "size_y": 0.08, "height": 0.1, "base_z": 0.74},
        {"class": "book", "x": 0.2, "y": -0.1, "size_x": 0.2, "size_y": 0.15, "height": 0.03, "base_z": 0.74},
        {"class": "cup", "x": 3.0, "y": 3.0, "size_x": 0.08, "size_y": 0.08, "height": 0.1, "base_z": 0.0},
    ]
    scene = schema.build_scene(objs, name="desk")
    on_table = [o["class"] for o in scene["objects"] if o["on"] == "o1"]
    assert on_table == ["cup", "book"]
    twin = to_twin.scene_to_twin(scene)
    assert sorted(o["name"] for o in twin["objects"]) == ["book", "cup 1"]     # the other cup is far away on the floor
    assert twin["obstacles"] == [] and twin["left_out"] == 1
    cup = next(o for o in twin["objects"] if o["name"] == "cup 1")
    assert cup["pos_cm"] == [-20.0, 10.0] and cup["size_cm"][2] == 10.0        # real centimetres


def test_nothing_movable_is_an_error():
    scene = schema.build_scene([{"class": "chair", "x": 0, "y": 0, "size_x": 0.5, "size_y": 0.5, "height": 0.5}])
    with pytest.raises(to_twin.NoTable):
        to_twin.scene_to_twin(scene)
