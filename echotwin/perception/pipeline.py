"""The steps from phone photos to a scene file, as a list of commands. Standard library only.

The robot server (a different Python environment) runs these steps as subprocesses while it shows progress,
so this module must import without numpy, torch or scipy.

    1. reconstruct   photos -> 3D cloud (VGGT)                        GPU environment
    2. detect        YOLO on the same photos, labels onto the cloud   GPU environment
    3. objects       cloud + labels -> objects on a map -> scene.json maps environment
    4. review        vision model checks the objects (needs a key)    maps environment

The scale of a photo reconstruction is a guess from the phone height (printed by step 1), so later steps
carry a "{scale}" placeholder that `fill` replaces.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCALE_RE = re.compile(r"--scale ([0-9.]+)\s+\(then fix")


@dataclass
class Step:
    name: str
    argv: list[str]
    typical_s: float          # usual duration, only for the progress bar
    optional: bool = False    # a failure here is not the end of the run
    env: dict | None = None   # extra environment variables for this command


def resolve(path: str) -> str:
    """A path from .env: relative paths are relative to the repository, so the same .env works anywhere."""
    path = (path or "").strip()
    return str(path if not path or Path(path).is_absolute() else REPO / path)


def rel(path: Path) -> str:
    """Show a path relative to the repository when it is inside it (commands run with the repo as cwd)."""
    p = Path(path)
    if p.is_absolute():
        try:
            return p.resolve().relative_to(REPO).as_posix()
        except ValueError:
            pass
    return str(p)


def plan(photos: Path, out: Path, *, gpu_py: str, maps_py: str | None = None, frames: int = 12,
         cam_height: float = 0.45, floor_offset: float = 0.0, review: bool = True,
         table: bool = True, vggt_path: str = "") -> list[Step]:
    """Commands for one run. `photos` is a folder of images, `out` the folder for everything produced.

    table=True is tuned for a tabletop: a fine map (1 cm cells) and small objects allowed.
    """
    maps_py = maps_py or gpu_py
    cloud, stem, photos = rel(out / "cloud.ply"), rel(out / "objects"), rel(photos)
    gpu_env = {"PYTHONPATH": vggt_path + os.pathsep + os.environ.get("PYTHONPATH", "")} if vggt_path else None
    objects = [maps_py, "-m", "echotwin.perception.objects", cloud, "-o", stem, "--up", "y",
               "--scale", "{scale}", "--floor-offset", str(floor_offset)]
    if table:
        objects += ["--res", "0.01", "--min-area", "0.0015"]
    steps = [
        Step("Building a 3D model", [gpu_py, "-m", "echotwin.perception.reconstruct", photos, "-o", cloud,
                                     "--frames", str(frames), "--cam-height", str(cam_height)], 150, env=gpu_env),
        Step("Finding objects", [gpu_py, "-m", "echotwin.perception.detect", cloud], 40, env=gpu_env),
        Step("Placing them on the table", objects, 15),
    ]
    if review:
        steps.append(Step("Checking what they are", [maps_py, "-m", "echotwin.perception.review", cloud,
                                                      f"{stem}_scene.json"], 40, optional=True))
    return steps


def scene_path(out: Path) -> Path:
    return Path(out) / "objects_scene.json"


def fill(argv: list[str], scale: float | None) -> list[str]:
    return [a.replace("{scale}", f"{scale:.4f}" if scale else "1.0") for a in argv]


def find_scale(output: str) -> float | None:
    m = SCALE_RE.search(output)
    return float(m.group(1)) if m else None


def settings(env=os.environ) -> dict:
    """What the robot server reads from the environment (.env)."""
    py = resolve(env.get("PERCEPTION_PY") or "")
    vggt = resolve(env.get("VGGT_PATH") or "third_party/vggt")
    return {"gpu_py": py, "maps_py": resolve(env.get("PERCEPTION_MAPS_PY") or "") or py,
            "vggt_path": vggt if Path(vggt).is_dir() else "",
            "mode": (env.get("SCAN_MODE") or "auto").strip().lower(),          # auto | quick | 3d
            "frames": int(env.get("SCAN_FRAMES") or 12),
            "cam_height": float(env.get("SCAN_CAM_HEIGHT") or 0.45),
            "min_photos": 3}


def usable(cfg: dict, n_photos: int) -> str | None:
    """None when the 3D pipeline can run, otherwise the reason it cannot (in plain words)."""
    if cfg["mode"] == "quick":
        return "quick mode is selected"
    if n_photos < cfg["min_photos"]:
        return f"3D needs at least {cfg['min_photos']} photos"
    if not cfg["gpu_py"] or not Path(cfg["gpu_py"]).exists():
        return "the perception environment is not set up (PERCEPTION_PY)"
    return None
