"""Which detector model to use, and how to load it. Shared by the detect step and the benchmark.

A model is named by its weights file in models/ (git-ignored). Add ":text" to an open-vocabulary YOLOE model to
give it the catalog's text prompt list instead of its built-in names. "auto" picks the best one you have:

    yoloe-26s-seg.pt:text   open vocabulary, finds cushions, rugs, speakers, posters... (see docs/RESULTS.md)
    yolo11s-seg.pt          the 80 YOLO classes (downloaded on first use if missing)

Weights are never downloaded here unless `download=True`, except the plain YOLO fallback, which Ultralytics fetches
by name as before.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"
PREFERRED = ("yoloe-26s-seg.pt:text", "yolo11s-seg.pt")      # best first
FALLBACK = "yolo11s-seg.pt"


def split(spec: str) -> tuple[str, bool]:
    """'weights.pt:text' -> ('weights.pt', True). Only a trailing ':text' counts (Windows paths contain ':')."""
    return (spec[:-5], True) if spec.endswith(":text") else (spec, False)


def weights_path(spec: str) -> Path:
    w = Path(split(spec)[0])
    return w if w.parent != Path(".") else MODELS / w          # a bare file name always means models/


def best_available() -> str:
    """The first preferred model whose weights are on disk, else the plain YOLO fallback."""
    for spec in PREFERRED:
        if weights_path(spec).exists():
            return spec
    return FALLBACK


def load(spec: str, prompts=None, download: bool = False):
    """-> an Ultralytics model ready to predict. spec: 'auto' | weights | 'weights:text'."""
    if spec == "auto":
        spec = best_available()
    weights, text = split(spec)
    path = weights_path(spec)
    if not path.exists() and not download and spec != FALLBACK:
        raise SystemExit(f"{path} is missing. Put the file there, or run again with --download.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if text:
        from ultralytics import YOLOE
        from echotwin.scene import catalog
        names = list(prompts or catalog.PROMPTS)
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
