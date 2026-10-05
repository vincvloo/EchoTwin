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

- [x] **PR8 A physically honest simulation.** Real scale, a swappable 5-joint arm (built-in, SO-ARM100, or your own
      descriptor), a contact grasp with friction and mass instead of gluing the object to the hand, spoken refusals
      (too wide, too thin, out of reach), and every default skill re-checked (`docs/RESULTS.md`). Open: the SO-ARM100's
      single jaw does not hold the full table yet; tall objects are the weak spot.
- [x] **PR9 Skills from measurements.** Each object is described by grip width, height, length and estimated weight;
      a skill is planned from the demos of similar sizes, so any detected name works. Result (`docs/RESULTS.md`): same
      success as the shape label, plus the robot asks only when the size is new. Open: learned skills do not beat the tuned
      defaults, because practice keeps every success at random styles (for PR13).
- [x] **PR10 Calibration.** A printed marker (AprilTag, `perception/marker.py`) on the table gives the true scale, the
      table plane and the origin in 3D scans, and the camera height and angle in quick mode; 3D scans also get a table
      texture painted from the photos. Tested on synthetic scenes only. Open: check it on a real capture with a printed
      marker (error against a measured object), and place the real arm's base by the marker (PR11).
- [x] **PR11 One skill interface, two back-ends.** The robot contract is written down (`robot/backend.py`); the simulation and
      a real arm (servo driver + the simulation as its twin) both fulfil it, switched by `BACKEND=sim|real`. No arm yet: the real
      back-end is tested on a mock arm and a fake servo bus, and the sim-vs-mock success rate is in `docs/RESULTS.md`.
      Open: the servo driver has never touched hardware (`docs/REAL_ARM.md`).
- [x] **PR12 Closed loop.** Before gripping the robot parks the arm out of the way and looks (a camera frame, the blob that
      differs from the table near where the object should be), plans again from what it sees, checks that the jaws closed on
      something, that it lifted, and where it ended up, and retries twice before asking to be shown. On by default for the
      real arm (`CLOSED_LOOP`). Tested on a render of the simulation and the mock arm; open: a real camera and its calibration.
- [x] **PR13 Learning.** A chunk policy (state in, the next 10 actions out, temporal ensembling; torch to train, numpy to run) learns the
      move by behaviour cloning of simulated demos and your own recorded ones (`docs/LEARNING.md`). Result (`docs/RESULTS.md`): 73 % of
      the 12 tasks with 600 demos against 94 % for the scripted skill, and about 20 % on the mock arm. Open: images, real teleoperated
      demos, randomising the simulation so it carries to an arm that lags, ACT proper (LeRobot). The demo files only use LeRobot's
      field names; they are not a LeRobot dataset and no export exists.

- [x] **PR14 Visual servoing.** The robot can look at its own gripper (`find_pads`, `ALIGN=on`) and line it up with the object before
      going down. Result (`docs/RESULTS.md`): the position error at the grip halves (6.8 to 3.7 mm) but the tall cylinder improves only when the
      arm is badly off, so it is off by default. Open: a better measurement of the tool (a mark on the hand, a side view).

## Open, not scheduled
- [x] A non-YOLO detector test (OWLv2, Grounding DINO: Apache-2.0): neither beats YOLOE with the public vocabulary and both are slower
      (`docs/RESULTS.md`). Open: a mask source (box as mask, or SAM) if the AGPL dependency must go; small
      tabletop objects are still named wrongly by every detector tested (`docs/RESULTS.md`).
- [ ] The MobileCLIP2 text encoder licence (Apple) should be checked before any commercial use.
