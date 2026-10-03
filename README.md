# EchoTwin

Photograph a table with your phone. EchoTwin builds a digital twin of it, finds the objects, and a simulated robot arm
learns to move them from a few demonstrations: "put the glass next to the chocolate".

Built to work with little data: a few phone photos instead of a LiDAR scan, a few demonstrations instead of a
training set, any object a detector can name instead of a fixed list.

## How it works, in six steps

1. **Capture**: about 10 phone photos or a short video of a table ([how](docs/CAPTURE_GUIDE.md)), with an optional printed marker on the table for true sizes.
2. **Reconstruct**: VGGT turns the photos into a 3D point cloud.
3. **Level and scale**: find the table plane, set the size from the phone height.
4. **Detect**: an open-vocabulary detector names the objects in each photo and votes them into 3D.
5. **Review** *(optional)*: an NVIDIA vision model looks at each object again and corrects names, shapes and false
   detections (needs a key).
6. **Act and learn**: every object gets a shape and a "can I move it" flag, the table becomes a twin, and you tell the
   robot what to move. It does it, says it is already done, or asks you to show it once
   ([how to teach it](docs/TEACHING.md)).

Details: [docs/PIPELINE.md](docs/PIPELINE.md). Other capture methods and why photos are the main one:
[docs/APPROACHES.md](docs/APPROACHES.md). Roadmap: [docs/TASKS.md](docs/TASKS.md).

## Setup

Two Python environments (the GPU stack and the simulator do not need to share one). Every path below and in
`.env` is relative to the repository, so the same setup works on any machine.

```powershell
# Perception: reconstruction, detection, object map (needs a GPU for the 3D model)
python -m venv .venv-perception
.venv-perception\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # your CUDA
.venv-perception\Scripts\pip install -r requirements-perception.txt

# VGGT (Meta): clone it into third_party/ (git-ignored). Pin a commit you have checked.
git clone https://github.com/facebookresearch/vggt third_party/vggt
.venv-perception\Scripts\pip install -r third_party/vggt/requirements.txt

# Robot: MuJoCo twin, the dashboard and the phone page (this is the app you open)
uv venv --python 3.12 .venv-robot
uv pip install --python .venv-robot -r requirements-robot.txt

copy .env.example .env      # then set PERCEPTION_PY and ROBOT_PY (relative paths are fine)
```

On Linux or macOS use `bin/python` instead of `Scripts\python.exe`.

Everything runs offline without API keys. `AI_API_KEY` (NVIDIA) adds object naming, phrasing and the review
of detections; `ELEVENLABS_API_KEY` adds a voice.

### Detection weights

The detect step uses the best model you have in `models/` (git-ignored). The recommended one is YOLOE with text
prompts and a public list of 365 names (Objects365); it names furniture and everyday things correctly
(`docs/RESULTS.md`). Fetch it once (about 270 MB from github.com/ultralytics/assets):

```powershell
.venv-perception\Scripts\python -m echotwin.perception.detectors --download
```

Without it, the plain YOLO11 model is used and downloaded on first use. To look for other things, put one name per
line in a text file and set `DETECT_PROMPTS=my_words.txt` in `.env` (or `lvis`, `coco`, `catalog`). Small objects on a
table are still hard to name for every detector; the review step (`AI_API_KEY`) helps.

### 3D or quick scan

With `PERCEPTION_PY` set, a scan builds a 3D model of the table (VGGT, detector, optional review). It takes a few
minutes on a laptop GPU; the dashboard and the phone show progress and a **Skip, use quick mode** button. Without
`PERCEPTION_PY`, with fewer than 3 photos, or if a step fails, the quick one-photo method runs instead. Set
`SCAN_MODE=quick` to always use it.

## Run

```powershell
.\launch.ps1        # one server: dashboard http://localhost:8000, phone https://<laptop-ip>:8443/phone
```

Open the dashboard and use the example photos, upload your own photos or a video, or scan with the phone (open the
phone address on the same Wi-Fi and accept the certificate warning once; HTTPS is needed for camera, tilt and
microphone). Then tell the robot what to move.

Each step also runs by itself:

| Goal | Command |
|---|---|
| Photos -> cloud (VGGT) | `python -m echotwin.perception.reconstruct examples/table_photos -o data/table.ply --cam-height 0.45` |
| Cloud -> detector labels | `python -m echotwin.perception.detect data/table.ply` |
| Labels -> objects (`scene.json`) | `python -m echotwin.perception.objects data/table.ply -o out/table --up y --scale <hint> --res 0.01 --min-area 0.0015` |
| Review with a vision model (optional) | `python -m echotwin.perception.review data/table.ply out/table_scene.json` |
| Objects -> robot twin | `python -m echotwin.scene.to_twin out/table_scene.json --out data/twin.zip` |
| Which detector is best? | `python -m echotwin.perception.bench_detect yolo11s-seg.pt yoloe-26s-seg.pt:text=objects365` |
| Run on a real arm (or the mock one) | `BACKEND=real` in `.env`, see [docs/REAL_ARM.md](docs/REAL_ARM.md) |
| Use another arm | `python -m echotwin.robot.arm --list` and [docs/ARMS.md](docs/ARMS.md) |
| Tests | `python -m pytest -q tests/perception tests/scene` and `<robot python> -m pytest -q tests/robot tests/scene` |

## Where things are

```
echotwin/perception/   photos -> cloud -> objects, detectors, benchmark
echotwin/scene/        the contract between perception and the robot (scene.json, class catalog, twin builder)
echotwin/robot/        digital twin with a physical arm (`arms/`), server, skills, voice
apps/robot_ui/         the dashboard and the phone page
docs/                  pipeline, approaches, capture and teaching guides, results, history
examples/              table photos
```

What was tried and removed (colour blocks, the sonar mobile base): [docs/history](docs/history/README.md).

## Licence

EchoTwin is released under the [GNU AGPL-3.0-or-later](LICENSE). It uses Ultralytics YOLO (AGPL-3.0) and the
VGGT model, whose original weights are **non-commercial**. See [THIRD_PARTY.md](THIRD_PARTY.md) before you use
it commercially or run it as a network service.
