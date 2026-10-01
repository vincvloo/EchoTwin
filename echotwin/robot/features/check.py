"""Check photos before building the twin: per photo, is the sheet found, what is detected, is it sharp."""
import cv2
import numpy as np

from . import vision as V

MIN_SHARPNESS = 40
LABELS = {"red": "red block", "blue": "blue block", "yellow": "yellow block", "green": "green zone", "tray": "blue tray"}


def check_photo(jpeg: bytes) -> tuple[dict, np.ndarray | None]:
    """-> (result, preview image: photo with sheet outline | top-down view with detections)."""
    img = V.decode(jpeg)
    if img is None:
        return {"ok": False, "reason": "Not an image I can read.", "found": []}, None
    sharp = V.sharpness(img)
    corners = V.find_sheet(img)
    cands = V.pose_candidates(corners, img.shape) if corners is not None else []
    shown = img.copy()
    if corners is not None:
        cv2.polylines(shown, [corners.astype(np.int32)], True, (0, 220, 120) if cands else (0, 0, 255), 4)
    h = 300
    left = cv2.resize(shown, (int(shown.shape[1] * h / shown.shape[0]), h))
    if not cands:
        # no A4 sheet: everyday-objects mode (glass, chocolate, case... on any table)
        from . import everyday as E
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
            ok, reason = True, f"Everyday objects: {n} found (named by the AI when you build)."
        return {"ok": ok, "mode": "everyday", "reason": reason, "found": [f"object {i}" for i in range(1, n + 1)],
                "sharpness": round(sharp)}, np.hstack([left, right])
    pose = V.choose_pose(cands, None)
    warped, valid = V.warp(img, pose["img_pts"])
    dets = V.detect(warped, valid)
    top = V.annotate(warped, dets)
    top = cv2.resize(top, (int(top.shape[1] * h / top.shape[0]), h))
    found = [LABELS[k] for k in ("red", "blue", "yellow", "green", "tray") if k in dets]
    missing = [LABELS[k] for k in ("red", "blue", "yellow", "green", "tray") if k not in dets]
    if sharp < MIN_SHARPNESS:
        ok, reason = False, "Too blurry. Hold the phone still."
    elif not found:
        ok, reason = False, "Sheet found, but no blocks or zones. Check colours and lighting."
    else:
        ok = True
        reason = "Good." if not missing else "Good, but missing: " + ", ".join(missing) + "."
    return {"ok": ok, "reason": reason, "found": found, "missing": missing, "sharpness": round(sharp),
            "camera_side": "front" if pose["cam"][1] < 0 else "back"}, np.hstack([left, top])
