"""Which detector model to use, and how to load it. Shared by the detect step and the benchmark.

A model is named by its weights file in models/ (git-ignored). Add ":text" to an open-vocabulary YOLOE model to
give it a text prompt list instead of its built-in names: ":text" uses the catalog's list, ":text=lvis" a public one
(coco, objects365, lvis) and ":text=my_words.txt" your own (one name per line). "auto" picks the best one you have:

    yoloe-26s-seg.pt:text=objects365   open vocabulary with the public Objects365 names (365 words): finds
                            cushions, rugs, speakers, coffee tables... (see docs/RESULTS.md)
    yolo11s-seg.pt          the 80 YOLO classes (downloaded on first use if missing)

Weights are never downloaded here unless `download=True`, except the plain YOLO fallback, which Ultralytics fetches
by name as before.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"
# The default vocabulary is a public one, not a list written for our photos. Change it with DETECT_PROMPTS in .env
# (coco | objects365 | lvis | catalog | a text file with one name per line).
PREFERRED = ("yoloe-26s-seg.pt:text=objects365", "yolo11s-seg.pt")      # best first
FALLBACK = "yolo11s-seg.pt"


def split(spec: str) -> tuple[str, str | None]:
    """'weights.pt:text=lvis' -> ('weights.pt', 'lvis'); 'weights.pt:text' -> ('weights.pt', 'catalog');
    'weights.pt' -> ('weights.pt', None). Only a ':text' suffix counts (Windows paths contain ':')."""
    head, sep, tail = spec.rpartition(":text")
    if sep and (tail == "" or tail.startswith("=")):
        return head, (tail[1:] or "catalog")
    return spec, None


def prompt_set(name: str) -> list[str]:
    """Names to ask for. 'catalog' | 'coco' | 'objects365' | 'lvis' | path of a text file (one name per line)."""
    if name == "catalog":
        from echotwin.scene import catalog
        return list(catalog.PROMPTS)
    if name == "coco":
        from echotwin.scene import catalog
        return list(catalog.COCO_CLASSES)
    if name in ("objects365", "lvis"):
        import ultralytics
        import yaml
        f = Path(ultralytics.__file__).parent / "cfg" / "datasets" / ("Objects365.yaml" if name == "objects365" else "lvis.yaml")
        raw = yaml.safe_load(f.read_text(encoding="utf-8"))["names"]
        raw = list(raw.values()) if isinstance(raw, dict) else list(raw)
        return _clean([str(n).split("/")[0] for n in raw])      # LVIS lists synonyms as 'rug/carpet': keep the first
    path = Path(name) if Path(name).is_absolute() else REPO / name
    if not path.exists():
        raise SystemExit(f"Unknown prompt list {name!r}: use catalog, coco, objects365, lvis or a text file.")
    return _clean(path.read_text(encoding="utf-8").splitlines())


def _clean(names) -> list[str]:
    out, seen = [], set()
    for n in names:
        n = " ".join(str(n).lower().replace("_", " ").split())
        if n and not n.startswith("#") and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def weights_path(spec: str) -> Path:
    w = Path(split(spec)[0])
    return w if w.parent != Path(".") else MODELS / w          # a bare file name always means models/


def best_available() -> str:
    """The first preferred model whose weights are on disk, else the plain YOLO fallback.
    DETECT_PROMPTS in the environment replaces the vocabulary of an open-vocabulary model."""
    for spec in PREFERRED:
        if weights_path(spec).exists():
            weights, vocab = split(spec)
            custom = os.environ.get("DETECT_PROMPTS", "").strip()
            return f"{weights}:text={custom}" if vocab and custom else spec
    return FALLBACK


def load(spec: str, prompts=None, download: bool = False):
    """-> an Ultralytics model ready to predict. spec: 'auto' | weights | 'weights:text'."""
    if spec == "auto":
        spec = best_available()
    weights, vocab = split(spec)
    path = weights_path(spec)
    if not path.exists() and not download and spec != FALLBACK:
        raise SystemExit(f"{path} is missing. Put the file there, or run again with --download.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if vocab:
        from ultralytics import YOLOE
        names = list(prompts or prompt_set(vocab))
        with contextlib.chdir(MODELS):      # Ultralytics looks for (and downloads) the text encoder in the current folder
            model = YOLOE(str(path.resolve()))
            model.set_classes(names, model.get_text_pe(names))
        return model
    from ultralytics import YOLO
    return YOLO(str(path) if path.exists() else weights)


def main(argv=None):
    """python -m echotwin.perception.detectors [--download]: show the model that will be used, or fetch it."""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--download", action="store_true",
                    help="download the recommended weights (yoloe-26s-seg.pt, 29 MB, and its text encoder, "
                         "mobileclip2_b.ts, 242 MB) from github.com/ultralytics/assets into models/ (AGPL-3.0)")
    a = ap.parse_args(argv)
    if a.download:
        load(PREFERRED[0], download=True)
    for spec in PREFERRED:
        print(f"{'found  ' if weights_path(spec).exists() else 'missing'}  {spec}")
    print(f"text encoder: {'found' if (MODELS / 'mobileclip2_b.ts').exists() else 'missing'}  mobileclip2_b.ts")
    print(f"used for detection: {best_available()}")


if __name__ == "__main__":
    main()
