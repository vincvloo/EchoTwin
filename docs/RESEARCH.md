# Research notes

Checked 2026-09-29. Re-verify links and device support before relying on them.

## Is the OnePlus 12 usable?

Yes, with one limit.

- **ARCore:** the OnePlus 12 (and 12R) is on Google's ARCore supported-devices list, marked
  "Supports Depth API". Source: https://developers.google.com/ar/devices
- **No time-of-flight / LiDAR sensor.** Rear cameras are 50 MP main (Sony LYT-808), 64 MP 3x tele,
  48 MP ultrawide. Source: https://en.wikipedia.org/wiki/OnePlus_12
- **What that means:** ARCore gives accurate, metric camera poses (camera + IMU tracking). Depth
  comes from "depth from motion" (comparing frames as you move), not from a depth sensor. It is
  noisier and weak on plain, textureless walls and glass. Expect 1-3 % scale error and a few cm of
  noise, which is what `scripts/make_sample_scan.py` simulates (1.5 % scale drift, 8 mm noise).
- **Practical checks on the phone:** install "Google Play Services for AR" from the Play Store; if it
  installs and updates, ARCore works.

## GitHub repositories and tools, by pipeline step

### 1. Capture on Android (poses + images/depth)

| Repo / tool | What it gives | Fit for this project |
|---|---|---|
| [google-ar/arcore-android-sdk](https://github.com/google-ar/arcore-android-sdk) | Official SDK + samples, incl. Recording & Playback API | Base for a custom capture app (task T5) |
| [googlesamples/arcore-depth-lab](https://github.com/googlesamples/arcore-depth-lab) | Depth API samples (point clouds, occlusion) | Reference for reading depth images |
| [githabideri/bildfang](https://github.com/githabideri/bildfang) | Logs video, ARCore pose, IMU, intrinsics to session folders; MIT; APK via CI | Very early (0 stars, targets Pixel). Good template, not production |
| [introlab/rtabmap](https://github.com/introlab/rtabmap) | Full RGB-D SLAM; Android app source has an ARCore camera | Play Store build is the 2018 Tango-only version; ARCore needs building from source |
| [remmel/recorder-3d](https://github.com/remmel/recorder-3d) | RGB + depth + pose recorder | Huawei AREngine ToF phones only. **Not usable on OnePlus 12** |
| Polycam / Scaniverse / KIRI Engine (apps, closed source) | Meshes or splats with real scale | Fastest route for task T1. Check which export formats are free |

### 2. Scan -> 2D occupancy grid

| Repo | Notes |
|---|---|
| [jkk-research/pointcloud_to_grid](https://github.com/jkk-research/pointcloud_to_grid) | ROS 1 + ROS 2 node: PointCloud2 -> OccupancyGrid by height/intensity |
| [Taeyoung96/OctoMap-ROS2](https://github.com/Taeyoung96/OctoMap-ROS2) | 3D map -> 2D occupancy grid via OctoMap (ROS 2 Humble) |
| This repo, `sonarloc/mesh_to_grid.py` | Plain Python, no ROS, outputs map_server PNG + YAML |

### 3. Localization with sparse range sensors

| Repo | Notes |
|---|---|
| [jamesjackson/ev3-localization](https://github.com/jamesjackson/ev3-localization) | Closest analogue: LEGO EV3, 2 ultrasonic sensors, particle filter, Python, real robot videos |
| [sjobeek/robostats_mcl](https://github.com/sjobeek/robostats_mcl) | Python MCL (laser data), clean reference |
| [ydsf16/particle_filter_localization](https://github.com/ydsf16/particle_filter_localization) | Particle filter in an occupancy grid |
| [ros-navigation/navigation2](https://github.com/ros-navigation/navigation2) | `nav2_amcl`: production AMCL; sonars must be published as a sparse `LaserScan` |
| GitHub topics: [monte-carlo-localization](https://github.com/topics/monte-carlo-localization), [particle-filter](https://github.com/topics/particle-filter?l=python) | More examples |

## Reference

- Thrun, Burgard, Fox, *Probabilistic Robotics* (2005): ch. 5 odometry motion model, ch. 6 beam
  sensor model, ch. 8 MCL and augmented MCL. `sonar.py` and `mcl.py` follow these chapters.
