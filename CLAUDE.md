# CLAUDE.md

Context for Claude Code in this repo. Read `docs/PIPELINE.md` first.

EchoTwin: scan a place with a phone, find its objects, localize a sonar robot in the map, and let a gripper
robot learn to move the objects. Built for the Physical AI Hackathon (Swiss {ai} Weeks). Owner: Vincent.
Hardware: phone + laptop (RTX 2050, 4 GB). No physical robot; everything robot-side is simulated.

## Layout
- `echotwin/perception/`: photos/video -> VGGT cloud -> level and scale -> YOLO -> object map. GPU env.
- `echotwin/navigation/`: sonar model, differential-drive robot, particle filter, demo run.
- `echotwin/scene/`: shared contract between perception and robot (no heavy dependencies): `schema.py` (scene.json), `catalog.py` (class -> shape / movable), `to_twin.py`.
- `echotwin/robot/`: MuJoCo twin, server, router, skills, voice, twin import. Robot env.
- `apps/perception_web` (port 8765), `apps/robot_ui` (dashboard + phone pages, served by the robot server).
- `data/ out/ runs/ models/ certs/` are local and git-ignored.

## Commands
```bash
python -m pytest -q tests/perception                           # perception env
<robot python> -m pytest -q tests/robot tests/scene            # robot env
python -m echotwin.perception.sample_scan                      # synthetic flat -> data/
python -m echotwin.navigation.demo --no-bench                  # map + sonar localization GIF -> out/
python apps/perception_web/server.py                           # http://127.0.0.1:8765
<robot python> -m echotwin.robot.server                        # http://localhost:8000
```
Two Python envs: perception (torch, VGGT, ultralytics, scipy, matplotlib) and robot (MuJoCo, FastAPI, OpenCV).
`.env` holds `PERCEPTION_PY`, `ROBOT_PY` and the API keys (never commit it).

## Rules
- Run the tests before saying something works; add a test for each new behaviour.
- Keep the shared `echotwin/scene` code free of heavy dependencies.
- Particle loops stay NumPy-vectorised; stochastic functions take an `np.random.Generator`.
- When the sensor model changes, re-run the benchmark and update `docs/RESULTS.md`.
- Plain, direct wording in docs. No em-dashes, no marketing phrasing (owner preference).
- Work in small PRs: one concern per PR, tests green in both envs.
