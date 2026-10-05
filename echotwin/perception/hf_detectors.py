"""Open-vocabulary detectors from Hugging Face that are not YOLO: OWLv2 and Grounding DINO (both Apache-2.0).

    python -m echotwin.perception.bench_detect owlv2:text=objects365 gdino-tiny:text=objects365 --download

They need `transformers` (not installed by default: `pip install transformers` in the perception environment) and their weights, which go
to `models/hf/` (git-ignored) when `--download` is given. They return boxes and names only, no masks: `bench_detect.py` scores names, and
the pipeline's `detect.py`, which labels the 3D points through masks, does not use them (see docs/RESULTS.md for what that would take).

The vocabulary is a list of names, as for YOLOE (`detectors.prompt_set`). Grounding DINO reads about 256 text tokens per image, so a long
list is asked in chunks and the detections are merged.
"""
from __future__ import annotations

from pathlib import Path

MODELS = Path(__file__).resolve().parents[2] / "models"
CACHE = MODELS / "hf"

FAMILIES = {
    "owlv2": {"repo": "google/owlv2-base-patch16-ensemble", "licence": "Apache-2.0", "chunk": 400},
    "gdino-tiny": {"repo": "IDEA-Research/grounding-dino-tiny", "licence": "Apache-2.0", "chunk": 40},
}


def is_hf(spec_head: str) -> bool:
    """Is this model name one of ours (the part of the spec before ':text')?"""
    return spec_head in FAMILIES


def chunks(names: list[str], n: int) -> list[list[str]]:
    return [names[i:i + n] for i in range(0, len(names), n)]


def match_name(label: str, names: list[str]) -> str:
    """Grounding DINO returns the words it matched, which may be one name, part of one, or a few run together ('cup bottle'):
    the longest asked-for name found in the text, else the text itself."""
    text = " ".join(str(label).lower().split())
    best = ""
    for n in names:
        if (f" {n} " in f" {text} ") and len(n) > len(best):
            best = n
    return best or text


def dets_from_scores(scores, label_ids, names: list[str], conf: float) -> list[tuple[str, float]]:
    """(name, confidence) for every score at or above `conf`; `label_ids` index `names`."""
    return [(names[int(i)], float(s)) for s, i in zip(scores, label_ids) if float(s) >= conf]


class HFDetector:
    """predict(path, conf) -> [(name, confidence)] for a fixed vocabulary."""

    def __init__(self, family: str, names: list[str], download: bool = False, device: str | None = None):
        import torch
        if family not in FAMILIES:
            raise SystemExit(f"Unknown detector {family!r}: {', '.join(FAMILIES)}.")
        try:
            import transformers
        except ImportError as e:
            raise SystemExit("This detector needs the 'transformers' library: pip install transformers (in the perception environment).") from e
        self.family, self.names, self.cfg = family, list(names), FAMILIES[family]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        CACHE.mkdir(parents=True, exist_ok=True)
        kw = {"cache_dir": str(CACHE), "local_files_only": not download}
        try:
            self.processor = transformers.AutoProcessor.from_pretrained(self.cfg["repo"], **kw)
            cls = transformers.Owlv2ForObjectDetection if family == "owlv2" else transformers.AutoModelForZeroShotObjectDetection
            self.model = cls.from_pretrained(self.cfg["repo"], **kw).to(self.device).eval()
        except OSError as e:
            raise SystemExit(f"{self.cfg['repo']} is not in {CACHE}. Run again with --download (about 0.6 to 0.7 GB).") from e

    def predict(self, img_path: Path, conf: float) -> list[tuple[str, float]]:
        import torch
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        out: list[tuple[str, float]] = []
        for part in chunks(self.names, self.cfg["chunk"]):
            if self.family == "owlv2":
                inputs = self.processor(text=[[f"a photo of a {n}" for n in part]], images=image, return_tensors="pt").to(self.device)
                with torch.no_grad(), torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                    res = self.model(**inputs)
                side = max(w, h)                                         # OWLv2 pads the image to a square
                post = getattr(self.processor, "post_process_object_detection", None) or self.processor.image_processor.post_process_object_detection
                r = post(res, threshold=conf, target_sizes=[(side, side)])[0]
                out += dets_from_scores(r["scores"].tolist(), r["labels"].tolist(), part, conf)
            else:
                prompt = " . ".join(part) + " ."
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    res = self.model(**inputs)
                r = self.processor.post_process_grounded_object_detection(res, inputs.input_ids, threshold=conf, text_threshold=conf,
                                                                          target_sizes=[(h, w)])[0]
                labels = r.get("text_labels") or r.get("labels") or []
                out += [(match_name(lab, part), float(s)) for lab, s in zip(labels, r["scores"].tolist()) if float(s) >= conf]
        return out
