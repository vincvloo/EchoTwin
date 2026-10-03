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


def frames_from_video(data: bytes, n: int = 12, side: int = 1280) -> list[bytes]:
    """N evenly spaced frames of a whole video, as JPEG bytes (long side at most `side` px). [] if it cannot be read."""
    import tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        out = []
        for k in np.linspace(0, max(total - 2, 0), n).astype(int) if total else []:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, img = cap.read()
            if not ok:
                continue
            s = side / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                out.append(buf.tobytes())
        cap.release()
        return out
    finally:
        Path(path).unlink(missing_ok=True)


def sharpness(bgr) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = 480 / g.shape[1]
    g = cv2.resize(g, None, fx=s, fy=s)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())
