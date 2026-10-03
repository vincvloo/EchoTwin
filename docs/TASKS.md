# Task backlog

[x] done, [ ] open

## Done
- [x] PR1 Merge the two projects into EchoTwin, remove the blocks demo
- [x] PR2 Shared `scene.json`, class catalog for any detected object, twin builder, licence files
- [x] PR3 NVIDIA review of detections
- [x] PR4 Phone scan runs the 3D pipeline, with a "skip, quick mode" button
- [x] PR5 Teaching guide, capture comparison
- [x] PR6 Detector benchmark; text-prompt YOLOE with a public vocabulary is the preferred detector
- [x] PR7 One solution: the sonar mobile base is removed, one app (the dashboard) for scan and robot

## Next: from the simulation to a real robot with transferable skills
In this order. Each step is only meaningful once the one before it is solid.

- [ ] **PR8 A physically honest simulation.** Real scale (no 2x table), an arm model (a low-cost 5 to 6 joint arm such as
      the SO-101 is a likely first target; to verify), a contact grasp with friction and mass instead of gluing the object
      to the hand, then re-check every current skill against it. What fails here is what would fail on hardware.
- [ ] **PR9 Skills from measurements.** Describe each object by grasp width, height, footprint and estimated weight
      instead of four shapes (flat, box, cylinder, round), so a skill learned on a mug carries over to similar
      objects and any detected name works. Needs PR8: only a physical grasp makes these numbers meaningful.
- [ ] **PR10 Calibration.** A printed marker (AprilTag) on the table gives a true scale and the camera pose relative to
      the table, so perception is metric (today the scale is a guess from the phone height). Also builds the table
      texture in 3D mode (the cloud loader drops colours today).
- [ ] **PR11 One skill interface, two back-ends.** Skills output end-effector poses and gripper commands in metres;
      the simulator and a real arm both implement it. A record of success rate in sim vs real on the same task.
- [ ] **PR12 Closed loop.** Re-detect the object before the grasp, check the gripper closed on something, retry.
      The existing "imagine first, then ask for help" behaviour fits this.
- [ ] **PR13 Learning.** Demos are already stored in a LeRobot-style format. Train a policy (for example ACT) on real
      teleoperated demos, with simulated demos as extra data.

## Open, not scheduled
- [ ] A non-YOLO detector (OWLv2, Grounding DINO: permissive licences, would also remove the AGPL dependency); small
      tabletop objects are still named wrongly by every detector tested (`docs/RESULTS.md`).
- [ ] The MobileCLIP2 text encoder licence (Apple) should be checked before any commercial use.
