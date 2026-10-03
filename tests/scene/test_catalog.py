"""Class catalog: every YOLO class is known, sizes beat names, unknown names get a sensible guess."""
import pytest

from echotwin.scene import catalog


def test_all_80_coco_classes_are_covered():
    assert len(catalog.COCO_CLASSES) == 80
    for name in catalog.COCO_CLASSES:
        info = catalog.classify(name, 0.5, 0.5, 0.5)       # big enough that no name is overruled by its size
        assert info["known"] and info["shape"] in catalog.SHAPES


@pytest.mark.parametrize("alias,coco", [("phone", "cell phone"), ("Mug", "cup"), ("desk", "dining table"),
                                        ("sofa", "couch"), ("plant", "potted plant"), ("TV_monitor", "tv")])
def test_aliases(alias, coco):
    assert catalog.normalize(alias) == coco


def test_small_known_things_are_movable_and_furniture_is_not():
    assert catalog.classify("cup", 0.08, 0.08, 0.1)["movable"]
    assert catalog.classify("bottle", 0.07, 0.07, 0.25)["shape"] == "cylinder"
    assert catalog.classify("book", 0.2, 0.15, 0.03)["shape"] == "flat"
    assert not catalog.classify("chair", 0.5, 0.5, 0.5)["movable"]
    assert catalog.classify("dining table", 1.5, 0.8, 0.75)["surface"]


def test_size_beats_name():
    # a cup 2 m wide is a detection mistake: not something to pick up
    assert not catalog.classify("cup", 2.0, 1.0, 0.7)["movable"]
    # a potted plant is movable only when small
    assert catalog.classify("potted plant", 0.24, 0.36, 0.25)["movable"]
    assert not catalog.classify("potted plant", 0.45, 0.33, 0.79)["movable"]


def test_unknown_names_get_a_geometry_guess():
    box = catalog.classify("shoe box", 0.3, 0.2, 0.12)
    assert not box["known"] and box["shape"] == "box" and box["movable"]
    assert catalog.classify("chocolate bar", 0.1, 0.05, 0.01)["shape"] == "flat"
    assert catalog.classify("oil can", 0.08, 0.08, 0.2)["shape"] == "cylinder"
    assert not catalog.classify("wardrobe", 1.0, 0.6, 2.0)["movable"]


def test_a_tiny_dining_table_is_a_small_object_not_a_table():
    small = catalog.classify("dining table", 0.11, 0.06, 0.06)
    assert small["movable"] and not small["surface"] and small["label"] == "object" and small["shape"] == "box"
    assert catalog.classify("dining table", 0.07, 0.06, 0.02)["shape"] == "flat"
    big = catalog.classify("dining table", 0.83, 1.11, 0.23)
    assert big["surface"] and not big["movable"] and "label" not in big


def test_people_and_hands_do_not_become_part_of_the_scene():
    from echotwin.scene import schema
    sc = schema.build_scene([
        {"class": "person", "x": 0, "y": 0, "size_x": 0.2, "size_y": 0.2, "height": 0.3},
        {"class": "cup", "x": 1, "y": 0, "size_x": 0.08, "size_y": 0.08, "height": 0.1},
        {"class": "Hand", "x": 2, "y": 0, "size_x": 0.1, "size_y": 0.1, "height": 0.05}])
    assert [o["class"] for o in sc["objects"]] == ["cup"] and sc["objects"][0]["id"] == "o1"
    assert sc["dropped"] == ["person", "Hand"]
