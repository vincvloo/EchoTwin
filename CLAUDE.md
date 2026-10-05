# CLAUDE.md

Context for Claude Code in this repo. Read `docs/PIPELINE.md` first.

EchoTwin: photograph a table with a phone, build a digital twin of it, find its objects, and let a simulated gripper
robot learn to move them. Built for the Physical AI Hackathon (Swiss {ai} Weeks). Owner: Vincent.
Hardware: phone + laptop (RTX 2050, 4 GB). No physical robot yet; everything robot-side is simulated.
Direction: make the simulation physically honest and the skills transferable, so it can later run on a real arm
(see `docs/TASKS.md`).

## Layout
- `echotwin/perception/`: photos/video -> VGGT cloud -> level and scale -> detector -> object map. GPU env.
- `echotwin/scene/`: the contract between perception and robot (no heavy dependencies): `schema.py` (scene.json),
  `catalog.py` (class -> shape / movable), `to_twin.py`, `review.py`.
- `echotwin/robot/`: MuJoCo twin with a physical arm (`arm.py`, descriptors in `arms/`, see `docs/ARMS.md`), the server (dashboard, phone page, scans), router, skills, voice. Robot env. Sizes are real metres; a grasp is friction and force, so skills can fail: `python -m echotwin.robot.skillcheck`. Skills talk to a back-end (`backend.py`: `sim` = `World`, `real` = `RealBackend` with a twin and a driver; `BACKEND=` in `.env`, docs/REAL_ARM.md); the servo driver is untested on hardware. The closed loop (`prop_skills.loop_start`, `observe.py`, `features/locate.py`; `CLOSED_LOOP=`) looks, checks and retries, and looks at its own gripper (`ALIGN=`, `find_pads`) to line it up with the object; no real camera has been tested. A neural policy (`policy.py`, `policy_obs.py`, `demos.py`, `train_policy.py`; docs/LEARNING.md) can make the moves instead of the scripted skill (`POLICY=`); it is trained with torch in the perception env and runs on numpy. Skills are keyed by measurements (`features/measure.py`), not by shape or name; `--transfer` tests that.
- `apps/robot_ui/`: the dashboard and the phone page, served by the robot server. This is the only app.
- `data/ out/ models/ certs/ third_party/` are local and git-ignored.
- `docs/history/`: removed features (colour blocks, the sonar mobile base) and where to find their code.

## Commands
```bash
python -m pytest -q tests/perception tests/scene               # perception env
<robot python> -m pytest -q tests/robot tests/scene            # robot env
<robot python> -m echotwin.robot.server                        # http://localhost:8000 (or .\launch.ps1)
<robot python> -m echotwin.robot.skillcheck                    # success rate of each default skill on the arm
<robot python> -m echotwin.robot.arm --check [NAME]            # test an arm descriptor
python -m echotwin.perception.detectors                        # which detector will be used
python -m echotwin.perception.marker --print out/marker.png    # the calibration marker to print (true scale)
```
Two Python envs: perception (torch, VGGT, ultralytics, scipy, matplotlib) and robot (MuJoCo, FastAPI, OpenCV).
`.env` holds `PERCEPTION_PY`, `ROBOT_PY` and the API keys (never commit it). Paths in `.env` are relative to the repo.

## Rules
- Run the tests before saying something works; add a test for each new behaviour.
- Keep the shared `echotwin/scene` code free of heavy dependencies.
- No machine-specific paths in the repo; paths from `.env` are relative to the repository.
- One pipeline: the dashboard, the phone and the command line all use `echotwin/perception/pipeline.py`.
- Plain, direct wording in docs. No em-dashes, no marketing phrasing (owner preference).
- Work in small PRs: one concern per PR, tests green in both envs.
