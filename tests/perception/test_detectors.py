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
    assert D.best_available() == "yoloe-26s-seg.pt:text"


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
