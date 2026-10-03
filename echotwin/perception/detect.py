"""Label the points of a VGGT cloud with object classes, using YOLO11 segmentation on the same photos.

Runs in the VGGT environment (needs ultralytics), after echotwin.perception.reconstruct:

    ..\\vggt-env\\Scripts\\python.exe echotwin.perception.detect data/lounge_vggt.ply

video_to_ply.py stores, for every point, the photo and pixel it came from (<cloud>.pix.npz).
Each photo is segmented; a point gets the class of the most confident mask covering its pixel.
Writes <cloud>.labels.npz (per-point class id, confidence, frame) and <cloud>_detections/ (each photo
with its masks and labels drawn, for people to look at). Then build the labelled map
in the main environment with echotwin.perception.objects.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from echotwin.perception import detectors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cloud", help=".ply written by video_to_ply.py (its .pix.npz must be next to it)")
    ap.add_argument("--model", default="auto",
                    help="weights in models/, add :text for YOLOE with the catalog's text prompts; "
                         "auto = the best one you have (see detectors.py)")
    ap.add_argument("--download", action="store_true", help="let Ultralytics download missing weights")
    ap.add_argument("--conf", type=float, default=0.35, help="minimum detection confidence")
    a = ap.parse_args(argv)
    from PIL import Image

    pix = np.load(Path(a.cloud).with_suffix(".pix.npz"))
    frame, row, col = pix["frame"].astype(int), pix["row"].astype(int), pix["col"].astype(int)
    H, W = (int(v) for v in pix["model_hw"])
    n_frames = sum(1 for k in pix.files if k.startswith("frame_"))
    cls = np.full(len(frame), -1, np.int16)
    conf = np.zeros(len(frame), np.float32)
    print(f"detector: {detectors.best_available() if a.model == 'auto' else a.model}")
    model = detectors.load(a.model, download=a.download)
    names = model.names
    found = {}
    shots = Path(a.cloud).parent / (Path(a.cloud).stem + "_detections")
    for i in range(n_frames):
        img = pix[f"frame_{i}"]
        res = model.predict(img[..., ::-1].copy(), conf=a.conf, retina_masks=True, verbose=False)[0]  # BGR in
        shot = Image.fromarray(res.plot(line_width=2)[..., ::-1])   # the photo with masks and labels, for people
        shot.thumbnail((960, 960))
        shots.mkdir(exist_ok=True)
        shot.save(shots / f"photo_{i:02d}.jpg", quality=82)
        if res.masks is None:
            continue
        on = frame == i
        r, c = row[on], col[on]
        for m, k, p in zip(res.masks.data.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int),
                           res.boxes.conf.cpu().numpy()):
            small = np.asarray(Image.fromarray((m > 0.5).astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
            hit = np.nonzero(on)[0][small[r, c]]
            better = hit[conf[hit] < p]
            cls[better], conf[better] = k, p
            found[names[k]] = found.get(names[k], 0) + 1
    out = Path(a.cloud).with_suffix(".labels.npz")
    np.savez_compressed(out, cls=cls, conf=conf, frame=frame.astype(np.uint16),
                        names=json.dumps({int(k): v for k, v in names.items()}))
    print(f"{n_frames} frames, detections: " + ", ".join(f"{k} x{v}" for k, v in sorted(found.items(), key=lambda t: -t[1])))
    print(f"labelled {np.mean(cls >= 0):.0%} of {len(cls):,} points -> {out}")


if __name__ == "__main__":
    main()
