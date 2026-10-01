# Third-party components

EchoTwin's own code is under AGPL-3.0-or-later (see `LICENSE`). It uses the components below. None of
them is copied into this repository; you install or download them yourself. Check each licence before you
use or distribute a build, especially commercially. This list is a summary, not legal advice.

| Component | Used for | Licence | What to know |
|---|---|---|---|
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (YOLO11 weights and library) | Object detection and segmentation (`echotwin/perception/detect.py`) | AGPL-3.0, or a paid Ultralytics enterprise licence | Running EchoTwin with it, including serving it over a network, brings AGPL duties (offer the source, §13). This is why EchoTwin is AGPL. Weights download on first use. |
| [VGGT](https://github.com/facebookresearch/vggt) (Meta) | 3D reconstruction from photos (`echotwin/perception/reconstruct.py`) | VGGT License (Meta), custom | Not vendored. The code currently downloads the original `VGGT-1B` checkpoint, which is **non-commercial**. Only `VGGT-1B-Commercial` (access form required) allows commercial use, and military use is excluded. Redistribution must carry Meta's licence. |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Robot simulation | Apache-2.0 | |
| NVIDIA API (build.nvidia.com), optional | Object naming, phrasing | NVIDIA API terms | Needs your own key in `.env`. Photos you send leave your machine. |
| ElevenLabs, optional | Voice | ElevenLabs terms | Needs your own key in `.env`. |
| numpy, scipy, matplotlib, trimesh, OpenCV, Pillow, FastAPI, uvicorn, httpx | Core libraries | BSD / MIT / Apache-2.0 | Permissive; compatible with AGPL. |
| PyTorch | Runs VGGT and YOLO | BSD-3-Clause | |

## Data

- `examples/` holds phone photos taken by the project owner.
- Scaniverse and ARKitScenes captures are **not** in this repository. ARKitScenes (Apple) is
  non-commercial; if you add it yourself, follow its licence.

## For commercial users

You would need: an Ultralytics enterprise licence (or a different detector), the commercial VGGT checkpoint
(or a different reconstruction model), and to check the NVIDIA and ElevenLabs terms for your use.
