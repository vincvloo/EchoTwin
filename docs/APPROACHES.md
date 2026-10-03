# Approaches

Photos or video through VGGT and YOLO is the main path. The others are documented because they were built and
tested, or because they are the reference to compare against.

| Approach | Where | Metric | Object labels | Result | Role |
|---|---|---|---|---|---|
| **Photos / video -> VGGT -> open-vocabulary detector -> multi-view object map** | `perception/reconstruct.py`, `detect.py`, `objects.py` | Phone height (about 5 % off) | **Yes, in 3D, voted over photos** | Cleanest lounge map (armchairs, sofa, lamps) | **Main path** |
| Scaniverse Gaussian splat (phone app) | removed (was in `mapping.py`) | Yes | No | Worked, noisy; stage height matches the photo map within 5 % | Comparison (info button) |
| LiDAR mesh (ARKitScenes, iPad) | removed | Yes | No | 93 % floor coverage, table legs visible | Comparison (info button) |
| 3D mesh from a scanning app as scenery | `robot/twin_import/meshes.py` | Yes | No | Works; point clouds are refused | Visual backdrop |
| One photo, tilt camera, NVIDIA naming | `robot/features/everyday.py` | Approximate | AI names | Works, approximate | Quick mode, no GPU |
| Hand video: what moved where | `robot/features/video_everyday.py` | n/a | n/a | Works | Teaching, not mapping |

Scaniverse and LiDAR perform well but cost more to capture (a LiDAR device, or an app plus a long scan).
Their data is not in this repo; the comparison is in the dashboard (the info button on the photo upload).

Known limits of the main path (table test): the VGGT table surface is flat to only 2 to 3 cm, so objects
under about 5 cm vanish, and a hand in the frame becomes a fake obstacle.

## History

Colour blocks and the sonar mobile base were removed: see [history/README.md](history/README.md).
