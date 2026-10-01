"""Room video -> coloured point cloud (.ply) with VGGT, for scans without ARCore.

Runs in the separate VGGT environment, not the main one:

    ..\\vggt-env\\Scripts\\python.exe echotwin.perception.reconstruct room.mp4 -o data/room_video.ply --frames 20

What it does: pick N frames evenly, run VGGT (camera + depth heads only, backbone in bf16 so it
fits a 4 GB GPU), unproject confident depth pixels into one cloud, then rotate it so +Y is up
(glTF / ARCore convention). Up comes from the cameras: a handheld phone has little roll, so the
camera x axes are horizontal and 'up' is the direction most perpendicular to all of them.

The result has NO metric scale. Convert it with a scale guess first, then fix it with --ref.
Always pass --up y: the cloud is already upright, and auto-detection can pick a wall when little
floor is visible.
    python -m echotwin.perception.mapping data/room_video.ply -o out/room --up y --scale <hint printed below>

On the RTX 2050 (4 GB): 8 frames ~1.5 min, 20 frames ~5.5 min (peak 4.8 GB spills into shared RAM).
Film with the floor in view (phone low, pointing down), otherwise the map has no free space.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

WEIGHTS = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def read_frames(path, n, long_side=518):
    """N evenly spaced RGB frames (or all images in a folder), resized so the long side is 518
    and both sides are multiples of 14. No crop, no padding: padding would become fake points."""
    import imageio.v3 as iio
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    p = Path(path)
    if p.is_dir():
        # Photos: apply the phone's EXIF rotation; shrink early (12 MP x 30 photos is a lot of RAM).
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        files = [files[i] for i in np.linspace(0, len(files) - 1, min(n, len(files))).round().astype(int)]
        frames = []
        for f in files:
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            im.thumbnail((2 * long_side, 2 * long_side))
            frames.append(np.asarray(im))
    else:
        # Two streaming passes (count, then keep N): a 1-minute 1080p clip does not fit in RAM.
        total = sum(1 for _ in iio.imiter(p))
        idx = set(np.linspace(0, total - 1, min(n, total)).round().astype(int).tolist())
        frames = [f[..., :3] for i, f in enumerate(iio.imiter(p)) if i in idx]
    # Mixed portrait / landscape shots cannot share one tensor: keep the majority orientation.
    portrait = [f.shape[0] > f.shape[1] for f in frames]
    keep = sum(portrait) * 2 >= len(frames)
    if not all(q == keep for q in portrait):
        print(f"skipping {sum(q != keep for q in portrait)} {'landscape' if keep else 'portrait'} frame(s); "
              f"using {sum(q == keep for q in portrait)} {'portrait' if keep else 'landscape'} ones")
        frames = [f for f, q in zip(frames, portrait) if q == keep]
    h, w = frames[0].shape[:2]
    s = long_side / max(h, w)
    H, W = (max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14))
    out = [F.interpolate(torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1)[None].float() / 255,
                         size=(H, W), mode="bilinear", antialias=True, align_corners=False) for f in frames]
    hi = []                                                    # sharper copies for object detection
    for f in frames:
        im = Image.fromarray(np.ascontiguousarray(f))
        im.thumbnail((1280, 1280))
        hi.append(np.asarray(im))
    return torch.cat(out).clamp(0, 1), hi                      # each frame resized on its own


def load_model(device, dtype):
    import torch
    from vggt.models.vggt import VGGT
    model = VGGT(enable_point=False, enable_track=False)
    sd = torch.hub.load_state_dict_from_url(WEIGHTS, map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith(("point_head.", "track_head."))}
    model.load_state_dict(sd)
    del sd
    model = model.to(device).eval()
    model.aggregator.to(dtype)          # the 1B backbone in bf16; heads stay float32 (they run outside autocast)
    return model


def _rot_to_y(up):
    """Rotation taking unit vector `up` onto +Y (Rodrigues)."""
    a = np.cross(up, [0, 1.0, 0])
    s, c = np.linalg.norm(a), up[1]
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1, -1])
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / s ** 2)


def _dominant_plane(pts, rng, iters=300, thresh=0.01, sample=30_000):
    """(unit normal, point on plane) of the largest plane, RANSAC; thresh relative to the cloud size."""
    p = pts[rng.choice(len(pts), min(sample, len(pts)), replace=False)]
    tol = thresh * np.linalg.norm(np.ptp(p, axis=0))
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        k = int((np.abs((p - a) @ n) < tol).sum())
        if k > best_n:
            best, best_n = (n, a), k
    return best


def upright(extrinsic, pts=None, rng=None):
    """Rotation (3x3) taking the estimated world 'up' to +Y. extrinsic: (S, 3, 4) camera-from-world, OpenCV.

    Normally from the cameras: a handheld phone has little roll, so camera x axes are horizontal
    and 'up' is the direction most perpendicular to all of them. When the shots all face the same
    way (a tabletop from one side) the x axes are nearly parallel and that is ambiguous; then the
    normal of the largest plane in `pts` is used instead. Either way 'up' points towards the cameras
    from that surface (you film a floor or table from above).
    """
    R = extrinsic[:, :, :3]
    xs, ys = R[:, 0, :], R[:, 1, :]                 # camera x (right) and y (down) axes in world
    w, v = np.linalg.eigh(xs.T @ xs)
    up = v[:, 0]                                     # most perpendicular to every camera x axis
    if up @ (-ys).mean(0) < 0:                       # image 'up' is -y in OpenCV
        up = -up
    if w[1] < 0.1 * len(xs) and pts is not None:     # x axes span no plane: use the dominant surface
        n, a = _dominant_plane(pts, rng or np.random.default_rng(0))
        cams = -np.einsum("sji,sj->si", R, extrinsic[:, :, 3])
        up = n if np.median((cams - a) @ n) > 0 else -n
    return _rot_to_y(up)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="video file, or a folder of images")
    ap.add_argument("-o", "--out", required=True, help="output .ply")
    ap.add_argument("--frames", type=int, default=20, help="frames to use (20 takes ~5.5 min on a 4 GB GPU)")
    ap.add_argument("--conf", type=float, default=50, help="drop the least confident N %% of pixels")
    ap.add_argument("--cam-height", type=float, default=0.45,
                    help="typical phone height above the floor during capture (m), for the scale hint")
    a = ap.parse_args(argv)
    import torch       # heavy imports here so the module (and upright()) loads without the VGGT env
    import trimesh
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    t0 = time.time()
    imgs, hi = read_frames(a.video, a.frames)
    print(f"{len(imgs)} frames at {imgs.shape[3]}x{imgs.shape[2]} | device {device} {dtype}")
    model = load_model(device, dtype)
    print(f"model loaded ({time.time()-t0:.0f} s)")
    while True:
        try:
            with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                pred = model(imgs.to(device))
                extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"].float(), imgs.shape[-2:])
            break
        except Exception as e:                                # OOM (AcceleratorError on 4 GB cards)
            if "out of memory" not in str(e) or len(imgs) <= 6:
                raise
            pred = None
            torch.cuda.empty_cache()
            keep_idx = np.linspace(0, len(imgs) - 1, len(imgs) - 3).round().astype(int)
            imgs, hi = imgs[keep_idx], [hi[i] for i in keep_idx]
            print(f"GPU out of memory: retrying with {len(imgs)} frames")
    if device == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    depth = pred["depth"][0].float().cpu().numpy()
    conf = pred["depth_conf"][0].float().cpu().numpy()
    extr, intr = extr[0].cpu().numpy(), intr[0].cpu().numpy()
    pts = unproject_depth_map_to_point_map(depth, extr, intr)             # (S, H, W, 3)
    keep = conf >= np.percentile(conf, a.conf)
    rgb = (imgs.permute(0, 2, 3, 1).numpy()[keep] * 255).astype(np.uint8)
    pts = pts[keep]
    src = np.nonzero(keep)                                        # (frame, row, col) of every point

    Rup = upright(extr, pts)
    pts = pts @ Rup.T
    cams = (-np.einsum("sji,sj->si", extr[:, :, :3], extr[:, :, 3])) @ Rup.T   # camera centres, Y-up
    floor = np.quantile(pts[:, 1], 0.02)
    cam_h = float(np.median(cams[:, 1] - floor))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(Path(a.out).with_suffix('.cams.npz'), extrinsic=extr, intrinsic=intr, up_rotation=Rup)
    # Where each point came from, for echotwin.perception.detect: frame index, pixel in the model-size
    # frame, and the frames themselves at up to 1280 px (object detection needs more than 518 px).
    np.savez_compressed(Path(a.out).with_suffix('.pix.npz'), frame=src[0].astype(np.uint16),
                        row=src[1].astype(np.uint16), col=src[2].astype(np.uint16),
                        model_hw=np.array(imgs.shape[-2:]), **{f"frame_{i}": f for i, f in enumerate(hi)})
    trimesh.PointCloud(pts, colors=np.column_stack([rgb, np.full(len(rgb), 255, np.uint8)])).export(a.out)
    ext = np.ptp(pts, axis=0)
    print(f"wrote {a.out}: {len(pts):,} points, extent {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} (VGGT units, Y up)")
    print(f"camera height above floor: {cam_h:.3f} units | scale hint if you held the phone at "
          f"{a.cam_height:.2f} m: --scale {a.cam_height / max(cam_h, 1e-6):.3f}  (then fix with --ref)")
    print(f"next: python -m echotwin.perception.mapping {a.out} -o out/{Path(a.out).stem} --up y "
          f"--scale {a.cam_height / max(cam_h, 1e-6):.3f}")
    print(f"total {time.time()-t0:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
