# Task backlog for Claude Code

Work top to bottom. Each task: goal, where to work, done-when. Run `python -m pytest -q` after
every task and add tests for new behaviour. Update `CLAUDE.md` "Current status" when a task is done.

Status legend: [ ] open, [x] done

---

## [x] T0. Core pipeline in simulation
Scan -> grid -> simulated sonar robot -> particle filter, with benchmark and tests. Done 2026-09-29.

---

## [ ] T1. Run the pipeline on a real OnePlus 12 scan  (priority: highest)
**Goal:** prove the map step works on real phone data.
**Input:** Vincent's scan in `data/` (follow `docs/CAPTURE_GUIDE.md`).
**Work:**
- `python -m sonarloc.mesh_to_grid data/<scan> -o out/<room>`; inspect the PNG.
- Handle what real files break: multi-mesh `.glb` scenes, vertex colours, huge files (downsample
  with a voxel grid before RANSAC), wrong up axis (auto-detect: the up axis is the one along which
  the largest flat plane has its normal).
- Add `--truth ""` path in `run_demo.py` so the demo runs with world = map when no truth exists
  (already supported; verify).
**Done when:** the real map shows walls and furniture legs; reference distances match within 3 cm;
the demo GIF runs on the real map.

## [ ] T2. Interactive browser simulator  (priority: high, this is the hackathon demo)
**Goal:** a single self-contained `web/index.html` that shows the robot, sonar cones and particles
live, runs on a phone browser, needs no server.
**Work:**
- Port `gridmap.raycast`, `SonarRig`, `Robot`, `Wander`, `MCL` to plain JS (no framework).
  Distance field: Felzenszwalb 2D EDT. Typed arrays for particles.
- Embed the default map as base64 PNG: `out/web_maps.png` channels R = map occupied,
  G = map known, B = ground truth occupied (generate with a small script, commit it as `scripts/export_web_map.py`).
- Controls: play/pause, speed, restart localization, kidnap robot, 2 vs 4 sonars, beam width,
  noise level, auto-wander vs manual driving (arrow keys + on-screen buttons),
  click on the map to drop an obstacle that is **not** in the map (tests robustness live).
- Panel: position error, heading error, particle count, spread, status pill
  (Searching / Converging / Localized), error sparkline.
- Upload: load a `map.png` + resolution produced by `mesh_to_grid` to run on your own room.
- Light and dark theme; works at 400 px wide.
**Done when:** converges on the sample map in under ~10 s at 1x speed in Chrome desktop and
Android Chrome; kidnap recovers; frame time under 16 ms with 3000 particles.

## [ ] T3. Close the loop: navigate to a goal  ("Act" part of the hackathon theme)
**Goal:** once localized, drive to a clicked goal using the estimated pose only.
**Work:** A* on the grid inflated by robot radius (`dist > 0.15`), pure-pursuit controller on the
estimated pose, sonar-based emergency stop. Add `sonarloc/planner.py` + tests.
**Done when:** 10/10 goals reached in simulation without collisions after convergence.

## [ ] T4. Make the simulation less optimistic
**Goal:** benchmark numbers that predict reality better.
**Work:**
- Simulated world uses a different sensor model than the filter: side lobes (secondary cone at
  low probability), multipath (occasional reading = longer path), cross-talk between sonars,
  60 ms sequential firing so sonars are not synchronous, wheel slip bursts.
- Likelihood-field or precomputed expected-range lookup table (x, y, theta bins) for speed.
- Report results in `docs/RESULTS.md` next to the current ones.
**Done when:** benchmark includes an "adversarial" config and the README states real-world expectations based on it.

## [ ] T5. Custom Android capture app (optional)
**Goal:** raw data (RGB + ARCore pose + depth + IMU) instead of an app's black-box mesh.
**Work:** Kotlin app from `google-ar/arcore-android-sdk` samples; see `githabideri/bildfang` for a
session-folder format. Export `frames/*.jpg`, `depth/*.png` (16-bit mm), `poses.csv`,
`intrinsics.json`. Add `sonarloc/fuse_depth.py` to back-project depth into a point cloud.
**Done when:** a OnePlus 12 session converts to a point cloud that passes T1 checks.

## [ ] T6. ROS 2 bridge (optional)
**Goal:** same map and sonars in ROS 2 so `nav2_amcl` can be compared with our filter.
**Work:** map already in map_server format; publish each sonar as `sensor_msgs/Range` and as a
sparse `LaserScan`; launch file for `nav2_amcl` + RViz.
**Done when:** AMCL localizes in the sample map from a bag recorded by the simulator.

## [ ] T7. Real robot (only if hardware appears)
ESP32 + 4x HC-SR04 (fire sequentially, ~60 ms apart) + encoder motors, streaming over Wi-Fi to the
laptop running the same filter. Measure mount offsets into `SonarRig.mounts`.
