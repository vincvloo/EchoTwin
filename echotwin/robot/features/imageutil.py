"""Small image helpers shared by the photo and video code."""
import cv2
import numpy as np


def decode(jpeg: bytes, max_side: int = 1280) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def sharpness(bgr) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = 480 / g.shape[1]
    g = cv2.resize(g, None, fx=s, fy=s)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())
