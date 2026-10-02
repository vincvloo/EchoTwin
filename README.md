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
5. **Review** *(planned)*: an NVIDIA vision model confirms or corrects each detection.
6. **Scene file**: every object gets a shape and a "can I move it" flag; one `scene.json` that both halves read.
7. **Use the map**: (a) a sonar robot localizes in the room; (b) a table becomes a digital twin.
8. **Act and learn**: tell the robot what to move ("put the glass next to the chocolate"). It does it, says it is already done, or asks you to show it once.

Details: [docs/PIPELINE.md](docs/PIPELINE.md). Other capture methods and why photos are the main one:
[docs/APPROACHES.md](docs/APPROACHES.md). Roadmap: [docs/TASKS.md](docs/TASKS.md).

## Setup

Two Python environments (the GPU stack and the simulator do not need to share one).

```powershell
# Perception: maps, detection, sonar localization, perception web app (GPU optional for maps)
pip install -r requirements-perception.txt        # install torch for your CUDA first
pip install git+https://github.com/facebookresearch/vggt.git     # pin a commit

# Robot: MuJoCo twin, dashboard, phone page
uv venv --python 3.12 .venv-robot
uv pip install --python .venv-robot -r requirements-robot.txt

copy .env.example .env      # set PERCEPTION_PY and ROBOT_PY; AI keys are optional
```

Everything runs offline without API keys. `AI_API_KEY` (NVIDIA) adds object naming and phrasing;
`ELEVENLABS_API_KEY` adds a voice.

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
