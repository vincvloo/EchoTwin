"""Check photos before building the twin: per photo, is it sharp and are there objects on the table."""
import cv2
import numpy as np

from . import everyday as E
from . import imageutil as V

MIN_SHARPNESS = 40


def check_photo(jpeg: bytes) -> tuple[dict, np.ndarray | None]:
    """-> (result, preview image: the photo next to the numbered objects found)."""
    img = V.decode(jpeg)
    if img is None:
        return {"ok": False, "reason": "Not an image I can read.", "found": []}, None
    sharp = V.sharpness(img)
    h = 300
    res = E.analyse(img, None)
    n = len(res["items"])
    marks = res["marks"]
    right = cv2.resize(marks, (int(marks.shape[1] * h / marks.shape[0]), h))
    left = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
    if sharp < MIN_SHARPNESS:
        ok, reason = False, "Too blurry. Hold the phone still."
    elif not n:
        ok, reason = False, "No objects found on the table. Is the whole table surface in view?"
    else:
        ok, reason = True, f"{n} object{'s' if n != 1 else ''} found (named by the AI when you build)."
    return {"ok": ok, "mode": "everyday", "reason": reason, "found": [f"object {i}" for i in range(1, n + 1)],
            "sharpness": round(sharp)}, np.hstack([left, right])
