"""Choosing and loading the detector (no models are loaded here)."""
import pytest

from echotwin.perception import detectors as D


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "MODELS", tmp_path)
    return tmp_path


def test_weights_are_looked_up_in_the_models_folder(models):
    assert D.weights_path("yolo11s-seg.pt") == models / "yolo11s-seg.pt"
    assert D.weights_path("yoloe-26s-seg.pt:text") == models / "yoloe-26s-seg.pt"        # the :text suffix is not part of the file
    absolute = models / "elsewhere" / "w.pt"
    assert D.weights_path(str(absolute)) == absolute


def test_auto_prefers_the_open_vocabulary_model_when_its_weights_exist(models):
    assert D.best_available() == D.FALLBACK                                              # nothing there: the plain YOLO
    (models / "yolo11s-seg.pt").write_bytes(b"x")
    assert D.best_available() == "yolo11s-seg.pt"
    (models / "yoloe-26s-seg.pt").write_bytes(b"x")
    assert D.best_available() == "yoloe-26s-seg.pt:text=objects365"          # a public vocabulary, not our own list


def test_missing_weights_are_not_downloaded_unless_asked(models):
    with pytest.raises(SystemExit) as e:
        D.load("yolo26s-seg.pt")
    assert "--download" in str(e.value)
    with pytest.raises(SystemExit):
        D.load("yoloe-26s-seg.pt:text")


def test_the_prompt_list_is_a_clean_generic_vocabulary():
    from echotwin.scene import catalog
    assert len(catalog.PROMPTS) >= 40 and len(set(catalog.PROMPTS)) == len(catalog.PROMPTS)
    assert all(p == p.lower() and p.strip() == p for p in catalog.PROMPTS)
    assert {"couch", "chair", "potted plant", "bottle", "cup"} <= set(catalog.PROMPTS)     # the common ones


def test_the_vocabulary_can_be_chosen(models, monkeypatch, tmp_path):
    assert D.split("yoloe.pt:text") == ("yoloe.pt", "catalog")
    assert D.split("yoloe.pt:text=lvis") == ("yoloe.pt", "lvis")
    assert D.split("yolo11s-seg.pt") == ("yolo11s-seg.pt", None)
    assert D.split("C:\\models\\yoloe.pt:text=my.txt") == ("C:\\models\\yoloe.pt", "my.txt")        # the drive colon is not a suffix
    (models / "yoloe-26s-seg.pt").write_bytes(b"x")
    monkeypatch.setenv("DETECT_PROMPTS", "lvis")
    assert D.best_available() == "yoloe-26s-seg.pt:text=lvis"
    words = tmp_path / "words.txt"
    words.write_text("Sofa\n# a comment\ncoffee_table\nsofa\n\n", encoding="utf-8")
    assert D.prompt_set(str(words)) == ["sofa", "coffee table"]                          # lower case, no comments, no repeats
    assert D.prompt_set("coco")[:2] == ["person", "bicycle"] and len(D.prompt_set("coco")) == 80
    with pytest.raises(SystemExit):
        D.prompt_set("no-such-list")
