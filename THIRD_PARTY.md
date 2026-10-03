# Third-party components

EchoTwin's own code is under AGPL-3.0-or-later (see `LICENSE`). It uses the components below. None of
them is copied into this repository; you install or download them yourself. Check each licence before you
use or distribute a build, especially commercially. This list is a summary, not legal advice.

| Component | Used for | Licence | What to know |
|---|---|---|---|
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (YOLO11 weights and library) | Object detection and segmentation (`echotwin/perception/detect.py`) | AGPL-3.0, or a paid Ultralytics enterprise licence | Running EchoTwin with it, including serving it over a network, brings AGPL duties (offer the source, §13). This is why EchoTwin is AGPL. Weights download on first use. |
| YOLOE (Ultralytics) and its text encoder MobileCLIP2 (Apple) | Open-vocabulary detection with text prompts (`models/yoloe-26s-seg.pt`, `models/mobileclip2_b.ts`) | YOLOE: AGPL-3.0. MobileCLIP2 weights: Apple's own model licence | The prompt words are the public Objects365 class names (shipped with Ultralytics). Downloaded with `python -m echotwin.perception.detectors --download`. As far as we know Apple's licence for the MobileCLIP weights allows research use only: **check it before any commercial use**. |
| [VGGT](https://github.com/facebookresearch/vggt) (Meta) | 3D reconstruction from photos (`echotwin/perception/reconstruct.py`) | VGGT License (Meta), custom | Not vendored. The code currently downloads the original `VGGT-1B` checkpoint, which is **non-commercial**. Only `VGGT-1B-Commercial` (access form required) allows commercial use, and military use is excluded. Redistribution must carry Meta's licence. |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Robot simulation | Apache-2.0 | |
| SO-ARM100 model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (The Robot Studio, Google DeepMind), optional | An arm you can switch to (`ARM=so_arm100`) | Apache-2.0 | Not in this repository. `python -m echotwin.robot.arm --download so_arm100` fetches the files at a pinned commit into `models/arms/` (git-ignored) with their licence. Credit: The Robot Studio and Google DeepMind. |
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
