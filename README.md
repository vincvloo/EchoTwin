# EchoTwin

Scan a place with a phone. EchoTwin finds the objects in it, lets a sonar robot localize itself in the map,
and builds a digital twin where a gripper robot learns to move those objects from a few demonstrations.

Built to work with little data: a few photos instead of a LiDAR scan, a few demos instead of a training set,
cheap ultrasonic sensors instead of a laser.

## The idea in eight steps

1. **Capture**: 13 or so phone photos, or one video of the place.
2. **Reconstruct**: VGGT turns the photos into a 3D point cloud.
3. **Level and scale**: find the floor, align the walls, set the size from the phone height.
4. **Detect**: YOLO finds objects in each photo and votes them into 3D.
5. **Review**: an NVIDIA vision model looks at each object again and corrects names, shapes and false detections (needs a key; skipped without one).
6. **Scene file**: every object gets a shape and a "can I move it" flag; one `scene.json` that both halves read.
7. **Use the map**: (a) a sonar robot localizes in the room; (b) a table becomes a digital twin.
8. **Act and learn**: tell the robot what to move ("put the glass next to the chocolate"). It does it, says it is already done, or asks you to show it once.

Details: [docs/PIPELINE.md](docs/PIPELINE.md). Other capture methods and why photos are the main one:
[docs/APPROACHES.md](docs/APPROACHES.md). Roadmap: [docs/TASKS.md](docs/TASKS.md).

## Setup

Two Python environments (the GPU stack and the simulator do not need to share one). Every path below and in
`.env` is relative to the repository, so the same setup works on any machine.

```powershell
# Perception: reconstruction, detection, maps, sonar localization, perception web app
python -m venv .venv-perception
.venv-perception\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # your CUDA
.venv-perception\Scripts\pip install -r requirements-perception.txt

# VGGT (Meta): clone it into third_party/ (git-ignored). Pin a commit you have checked.
git clone https://github.com/facebookresearch/vggt third_party/vggt
.venv-perception\Scripts\pip install -r third_party/vggt/requirements.txt

# Robot: MuJoCo twin, dashboard, phone page
uv venv --python 3.12 .venv-robot
uv pip install --python .venv-robot -r requirements-robot.txt

copy .env.example .env      # then set PERCEPTION_PY and ROBOT_PY (relative paths are fine)
```

On Linux or macOS use `bin/python` instead of `Scripts\python.exe`.

Everything runs offline without API keys. `AI_API_KEY` (NVIDIA) adds object naming, phrasing and the review
of detections; `ELEVENLABS_API_KEY` adds a voice.

### Phone scan: 3D or quick

With `PERCEPTION_PY` set, a sweep scan from the phone builds a 3D model of the table (VGGT, YOLO, optional review).
It takes a few minutes on a laptop GPU; the dashboard and the phone show progress and a **Skip, use quick mode**
button. Without `PERCEPTION_PY`, with fewer than 3 photos, or if a step fails, the quick one-photo method runs
instead. Set `SCAN_MODE=quick` to always use it.

## Run

```powershell
.\launch.ps1        # perception app http://127.0.0.1:8765, robot dashboard http://localhost:8000
```

| Goal | Command |
|---|---|
| Synthetic flat for tests and benchmark | `python -m echotwin.perception.sample_scan` |
| Map + sonar localization GIF | `python -m echotwin.navigation.demo --no-bench` |
| Photos -> cloud (VGGT) | `python -m echotwin.perception.reconstruct examples/lounge_photos -o data/lounge.ply --cam-height 1.3` |
| Cloud -> YOLO labels | `python -m echotwin.perception.detect data/lounge.ply` |
| Labels -> map + objects | `python -m echotwin.perception.objects data/lounge.ply -o out/lounge --up y --scale 3.333 --floor-offset 0.29` |
| Review with a vision model (optional, needs `AI_API_KEY`) | `python -m echotwin.perception.review data/lounge.ply out/lounge_scene.json` |
| Objects -> robot twin | `python -m echotwin.scene.to_twin out/lounge_scene.json --out data/twin.zip` |
| Robot server | `<robot python> -m echotwin.robot.server` |
| Tests | `python -m pytest -q tests/perception` and `<robot python> -m pytest -q tests/robot tests/scene` |

Phone: open `https://<laptop-ip>:8443/phone` on the same Wi-Fi and accept the certificate warning once
(HTTPS is needed for camera, tilt and microphone).

## Where things are

```
echotwin/perception/   photos -> cloud -> map -> objects
echotwin/navigation/   sonar model, robot, particle filter
echotwin/scene/        contract between perception and robot
echotwin/robot/        digital twin, server, skills, voice
apps/                  perception web app, robot dashboard and phone pages
docs/                  pipeline, approaches, capture guide, results, pitch deck
examples/              lounge and table photos
```

Sonar results and limits: [docs/RESULTS.md](docs/RESULTS.md). Sources: [docs/RESEARCH.md](docs/RESEARCH.md).

## Licence

EchoTwin is released under the [GNU AGPL-3.0-or-later](LICENSE). It uses Ultralytics YOLO (AGPL-3.0) and the
VGGT model, whose original weights are **non-commercial**. See [THIRD_PARTY.md](THIRD_PARTY.md) before you use
it commercially or run it as a network service.
