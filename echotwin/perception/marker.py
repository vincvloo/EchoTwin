"""A printed marker on the table gives the true scale and the table plane.

An AprilTag (36h11, id 0): a black square, by default 10 cm. Print the page, lay it flat on the table, and keep it
in view in a few photos. Its four corners are in the photo at known distances from each other, so a photo gives
metres per pixel on the table, and the camera height and angle above it.

    python -m echotwin.perception.marker --print out/marker.png      # the page to print (A4, 100 %, no "fit to page")
    python -m echotwin.perception.marker --find examples/table_photos # which photos show the marker

Needs only numpy, OpenCV (cv2.aruco) and, for --print, Pillow. Both Python envs have them.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

MARKER_ID = 0
DEFAULT_SIZE_MM = 100.0
F_FACTOR = 0.75           # focal length guess: 0.75 x the long side of the photo (a phone's main camera, about 69 degrees)


def size_m() -> float:
    """The printed square's side in metres (MARKER_SIZE_CM in .env, 10 by default)."""
    try:
        return float(os.environ.get("MARKER_SIZE_CM") or DEFAULT_SIZE_MM / 10) / 100.0
    except ValueError:
        return DEFAULT_SIZE_MM / 1000.0


def _dictionary():
    import cv2
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def make_marker(px: int) -> np.ndarray:
    """The tag, black border included, as a px x px grey image (px is rounded up to a multiple of 8 cells)."""
    import cv2
    px = int(np.ceil(px / 8) * 8)
    return cv2.aruco.generateImageMarker(_dictionary(), MARKER_ID, px)


def make_page(size_mm: float = DEFAULT_SIZE_MM, dpi: int = 300, page_mm=(210.0, 297.0)):
    """An A4 page with the tag in the middle and a ruler line of the same length under it.
    Returns (image, dpi_to_save): print at 100 %, then measure the black square and check it is size_mm."""
    import cv2
    px = int(np.ceil(size_mm / 25.4 * dpi / 8) * 8)
    dpi_eff = px * 25.4 / size_mm                       # so that the printed square is exactly size_mm
    W, H = int(round(page_mm[0] / 25.4 * dpi_eff)), int(round(page_mm[1] / 25.4 * dpi_eff))
    page = np.full((H, W), 255, np.uint8)
    y0, x0 = (H - px) // 2, (W - px) // 2
    page[y0:y0 + px, x0:x0 + px] = make_marker(px)
    ry = y0 + px + int(0.12 * px)
    page[ry - 3:ry + 3, x0:x0 + px] = 0
    for x in (x0, x0 + px - 3):
        page[ry - 20:ry + 20, x:x + 3] = 0
    cv2.putText(page, f"EchoTwin marker {size_mm:g} mm: print at 100 %, lay flat on the table",
                (max(10, x0 - px // 2), y0 - int(0.12 * px)),
                cv2.FONT_HERSHEY_SIMPLEX, px / 1100, 0, max(2, px // 400), cv2.LINE_AA)
    cv2.putText(page, f"this line is {size_mm:g} mm: check it with a ruler", (x0, ry + int(0.14 * px)),
                cv2.FONT_HERSHEY_SIMPLEX, px / 900, 0, max(2, px // 500), cv2.LINE_AA)
    return page, dpi_eff


def detect(image: np.ndarray) -> np.ndarray | None:
    """The four corners (top-left, top-right, bottom-right, bottom-left of the tag) as a (4, 2) float array, or None."""
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(_dictionary(), params).detectMarkers(gray)
    if ids is None:
        return None
    for c, i in zip(corners, ids.ravel()):
        if int(i) == MARKER_ID:
            return c.reshape(4, 2).astype(float)
    return None


def plane_points(side: float) -> np.ndarray:
    """The tag's corners in its own plane (metres): x to the right, y down, the origin at the top-left corner."""
    return np.array([[0, 0], [side, 0], [side, side], [0, side]], float)


def homography(corners: np.ndarray, side: float) -> np.ndarray:
    """3 x 3, from table-plane metres (tag frame) to photo pixels. Any point on the table plane can be placed in metres."""
    import cv2
    return cv2.getPerspectiveTransform(plane_points(side).astype(np.float32), corners.astype(np.float32))


def camera_matrix(shape, f: float | None = None) -> np.ndarray:
    h, w = shape[:2]
    f = f or F_FACTOR * max(w, h)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def pose(corners: np.ndarray, side: float, shape, f: float | None = None) -> dict:
    """Camera pose over the table from the tag: height above the table (m), angle of the viewing direction below the
    horizon (degrees, 90 = straight down), the camera position in the tag frame (centre origin, y up) and the
    reprojection error (pixels). The focal length is a guess unless `f` is given."""
    import cv2
    s = side / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)    # centre origin, y up (tag y is down)
    K = camera_matrix(shape, f)
    ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float32), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise ValueError("could not solve the marker pose")
    R, _ = cv2.Rodrigues(rvec)
    pos = (-R.T @ tvec).ravel()
    axis = R.T @ np.array([0.0, 0.0, 1.0])               # the viewing direction in the tag frame
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    err = float(np.linalg.norm(proj.reshape(4, 2) - corners, axis=1).mean())
    return {"height": float(abs(pos[2])), "pitch_deg": float(np.degrees(np.arcsin(min(1.0, abs(axis[2]))))),
            "position": pos.tolist(), "error_px": err}


def find_in_folder(folder: str) -> list[tuple[str, bool]]:
    import cv2
    out = []
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            img = cv2.imread(str(p))
            out.append((p.name, img is not None and detect(img) is not None))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", metavar="FILE", help="write the page to print (PNG, or PDF if the name ends in .pdf)")
    ap.add_argument("--size-mm", type=float, default=None, help="side of the black square (default MARKER_SIZE_CM or 100)")
    ap.add_argument("--find", metavar="FOLDER", help="say which photos in a folder show the marker")
    a = ap.parse_args(argv)
    if a.print:
        from PIL import Image
        mm = a.size_mm or size_m() * 1000
        page, dpi = make_page(mm)
        Path(a.print).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(page).save(a.print, dpi=(dpi, dpi), **({"resolution": dpi} if a.print.lower().endswith(".pdf") else {}))
        print(f"wrote {a.print}: print at 100 %, the black square must measure {mm:g} mm")
    if a.find:
        for name, seen in find_in_folder(a.find):
            print(f"  {name:32s} {'marker found' if seen else '-'}")
    if not (a.print or a.find):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
