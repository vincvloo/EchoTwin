"""What kind of thing is it? Class name -> 3D shape, can the gripper move it, can it hold other things.

Covers the 80 COCO classes YOLO11 knows, a few common aliases (so a reviewer or an open-vocabulary detector
can say "mug" or "desk"), and a geometry fallback for any other name. Numpy-free on purpose.

Shapes are the ones the robot twin knows: flat | box | cylinder | round.
Sizes always win over the table: a "cup" measured 2 m wide is a detection mistake, not a cup, so it is
treated as furniture instead of something to pick up.
"""
from __future__ import annotations

SHAPES = ("flat", "box", "cylinder", "round")
MAX_MOVABLE_FOOTPRINT = 0.40     # metres, longest side of something the gripper can move
MAX_MOVABLE_HEIGHT = 0.40        # metres

# name: (shape, movable, surface). movable = a small thing you could pick up. surface = can hold other things.
_T, _F = True, False
_ENTRIES: dict[str, tuple[str, bool, bool]] = {
    # people, animals, vehicles, street: never moved
    "person": ("box", _F, _F), "bicycle": ("box", _F, _F), "car": ("box", _F, _F), "motorcycle": ("box", _F, _F),
    "airplane": ("box", _F, _F), "bus": ("box", _F, _F), "train": ("box", _F, _F), "truck": ("box", _F, _F),
    "boat": ("box", _F, _F), "traffic light": ("cylinder", _F, _F), "fire hydrant": ("cylinder", _F, _F),
    "stop sign": ("flat", _F, _F), "parking meter": ("cylinder", _F, _F), "bench": ("box", _F, _T),
    "bird": ("round", _F, _F), "cat": ("box", _F, _F), "dog": ("box", _F, _F), "horse": ("box", _F, _F),
    "sheep": ("box", _F, _F), "cow": ("box", _F, _F), "elephant": ("box", _F, _F), "bear": ("box", _F, _F),
    "zebra": ("box", _F, _F), "giraffe": ("box", _F, _F),
    # carried things and sports gear
    "backpack": ("box", _T, _F), "umbrella": ("cylinder", _T, _F), "handbag": ("box", _T, _F),
    "tie": ("flat", _T, _F), "suitcase": ("box", _T, _F), "frisbee": ("flat", _T, _F), "skis": ("flat", _F, _F),
    "snowboard": ("flat", _F, _F), "sports ball": ("round", _T, _F), "kite": ("flat", _T, _F),
    "baseball bat": ("cylinder", _T, _F), "baseball glove": ("box", _T, _F), "skateboard": ("flat", _T, _F),
    "surfboard": ("flat", _F, _F), "tennis racket": ("flat", _T, _F),
    # kitchen and food
    "bottle": ("cylinder", _T, _F), "wine glass": ("cylinder", _T, _F), "cup": ("cylinder", _T, _F),
    "fork": ("flat", _T, _F), "knife": ("flat", _T, _F), "spoon": ("flat", _T, _F), "bowl": ("cylinder", _T, _F),
    "banana": ("cylinder", _T, _F), "apple": ("round", _T, _F), "sandwich": ("box", _T, _F),
    "orange": ("round", _T, _F), "broccoli": ("round", _T, _F), "carrot": ("cylinder", _T, _F),
    "hot dog": ("cylinder", _T, _F), "pizza": ("flat", _T, _F), "donut": ("round", _T, _F),
    "cake": ("cylinder", _T, _F),
    # furniture and appliances
    "chair": ("box", _F, _F), "couch": ("box", _F, _T), "potted plant": ("cylinder", _T, _F),
    "bed": ("box", _F, _T), "dining table": ("box", _F, _T), "toilet": ("box", _F, _F), "tv": ("box", _F, _F),
    "laptop": ("box", _T, _F), "mouse": ("box", _T, _F), "remote": ("flat", _T, _F), "keyboard": ("flat", _T, _F),
    "cell phone": ("flat", _T, _F), "microwave": ("box", _F, _F), "oven": ("box", _F, _F),
    "toaster": ("box", _T, _F), "sink": ("box", _F, _F), "refrigerator": ("box", _F, _F),
    "book": ("flat", _T, _F), "clock": ("cylinder", _T, _F), "vase": ("cylinder", _T, _F),
    "scissors": ("flat", _T, _F), "teddy bear": ("round", _T, _F), "hair drier": ("box", _T, _F),
    "toothbrush": ("flat", _T, _F),
}
COCO_CLASSES = tuple(_ENTRIES)

_ALIASES = {
    "phone": "cell phone", "mobile phone": "cell phone", "smartphone": "cell phone", "mug": "cup",
    "glass": "wine glass", "table": "dining table", "desk": "dining table", "coffee table": "dining table",
    "sofa": "couch", "armchair": "chair", "plant": "potted plant", "monitor": "tv", "television": "tv",
    "ball": "sports ball", "notebook": "book", "tv monitor": "tv", "can": "bottle", "jar": "vase",
    "teddy": "teddy bear", "stool": "chair",
}
# nothing in the catalog: guess from the name
_FLAT_WORDS = ("paper", "card", "sheet", "wrapper", "envelope", "tablet", "bar")
_ROUND_WORDS = ("ball", "egg", "case")
_CYL_WORDS = ("bottle", "glass", "cup", "can", "tube", "pen", "jar")


def normalize(name: str) -> str:
    n = " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())
    return _ALIASES.get(n, n)


def classify(name: str, size_x: float, size_y: float, height: float | None) -> dict:
    """-> {"shape", "movable", "surface", "known"} for one detected object (sizes in metres).

    "label" is added when the name does not fit the size and a plainer name is better ("object").
    """
    n = normalize(name)
    h = float(height or 0.1)
    foot = max(float(size_x), float(size_y))
    small = foot <= MAX_MOVABLE_FOOTPRINT and h <= MAX_MOVABLE_HEIGHT
    if n in _ENTRIES:
        shape, movable, surface = _ENTRIES[n]
        if surface and small:
            # a "dining table" 11 x 6 cm is not a table: on a tabletop the detector often gives the same name
            # to the table and to the small things on it. Size beats name.
            return {"shape": _guess_shape("object", size_x, size_y, h), "movable": True, "surface": False,
                    "known": False, "label": "object"}
        return {"shape": shape, "movable": bool(movable and small), "surface": surface, "known": True}
    return {"shape": _guess_shape(n, size_x, size_y, h), "movable": small, "surface": False, "known": False}


def _guess_shape(name: str, sx: float, sy: float, h: float) -> str:
    words = name.split()
    if any(w in _CYL_WORDS for w in words):
        return "cylinder"
    if any(w in _FLAT_WORDS for w in words) or h < 0.03:
        return "flat"
    if any(w in _ROUND_WORDS for w in words):
        return "round"
    if min(sx, sy) > 0 and max(sx, sy) / min(sx, sy) < 1.3 and h > 1.5 * max(sx, sy):
        return "cylinder"          # tall and narrow
    return "box"


# one stable colour per class for the twin (the real colour is not in the scene file)
_PALETTE = ("#4287f5", "#34a853", "#8d6e63", "#ab47bc", "#78909c", "#fbc02d", "#e64a19", "#2e7d32", "#0288d1",
            "#d81b60", "#5d4037", "#f57c00")


def color_for(name: str) -> str:
    n = normalize(name)
    return _PALETTE[sum(map(ord, n)) % len(_PALETTE)]
