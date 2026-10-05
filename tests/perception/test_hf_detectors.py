"""The Hugging Face detectors' plumbing, without any model: spec parsing, chunking, label matching, licences."""
from echotwin.perception import detectors as D
from echotwin.perception import hf_detectors as H


def test_specs_of_the_new_models_parse_like_the_yoloe_ones():
    assert D.split("owlv2:text=objects365") == ("owlv2", "objects365")
    assert D.split("gdino-tiny:text=catalog") == ("gdino-tiny", "catalog") and D.split("gdino-tiny:text") == ("gdino-tiny", "catalog")
    assert H.is_hf("owlv2") and H.is_hf("gdino-tiny") and not H.is_hf("yoloe-26s-seg.pt")


def test_licences_are_known_per_model():
    assert D.licence("owlv2:text=coco") == "Apache-2.0" and D.licence("gdino-tiny") == "Apache-2.0"
    assert D.licence("yoloe-26s-seg.pt:text=objects365") == "AGPL-3.0" and D.licence("yolo11s-seg.pt") == "AGPL-3.0"


def test_a_long_vocabulary_is_split_into_chunks_and_nothing_is_lost():
    names = [f"n{i}" for i in range(365)]
    parts = H.chunks(names, 40)
    assert len(parts) == 10 and sum(len(p) for p in parts) == 365 and parts[-1] == names[360:]
    assert [n for p in parts for n in p] == names


def test_grounding_dino_text_is_mapped_back_to_the_asked_for_names():
    names = ["cup", "wine glass", "bottle", "table"]
    assert H.match_name("cup", names) == "cup"
    assert H.match_name("a wine glass", names) == "wine glass"          # the longest asked-for name inside the text
    assert H.match_name("cup bottle", names) in ("cup", "bottle")
    assert H.match_name("Remote Control", names) == "remote control"     # not asked for: kept as it is (lowercase)


def test_scores_below_the_threshold_are_dropped_and_ids_become_names():
    out = H.dets_from_scores([0.9, 0.2, 0.5], [2, 0, 1], ["cup", "bottle", "table"], 0.25)
    assert out == [("table", 0.9), ("bottle", 0.5)]


def test_the_default_detector_is_still_yoloe(monkeypatch):
    monkeypatch.delenv("DETECT_PROMPTS", raising=False)
    assert D.PREFERRED[0].startswith("yoloe-26s-seg.pt")
