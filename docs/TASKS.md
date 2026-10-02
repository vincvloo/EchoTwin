# Task backlog

[x] done, [ ] open

- [x] Core pipeline in simulation (map, sonar, particle filter, benchmark)
- [x] Real data: photos -> VGGT -> map and objects (lounge, table)
- [x] Perception web app, bridge to robot twin, robot server with everyday objects
- [x] PR1 Merge Sonar and Robot into EchoTwin, remove the blocks demo
- [x] PR2 Shared `scene.json`, class catalog for any detected object, bridge fix (table window, real scale, obstacles), licence files
- [ ] Table texture: colour top-down image of the table from the cloud (the loader drops colours today)
- [ ] PR3 NVIDIA review of YOLO detections
      Idea: the review also judges grasp traits (fragile, hollow, soft), not only a shape from the list of four.
- [ ] PR4 Phone scan runs the 3D pipeline, with a "skip, quick mode" button
- [ ] PR5 Teaching guide (how to film and how it is processed) and the capture-comparison info button
- [ ] PR6 Detector benchmark: YOLO11 vs open-vocabulary and newer models
      Idea: open-vocabulary detection removes the 80-class limit.
- [ ] Sonar extras (browser simulator, goal navigation, harsher sensor model, ROS 2): see `TASKS_SONAR.md`

## Idea: describe objects by measurements, not four shapes

Robot skills are learned per shape (flat, box, cylinder, round). That is too coarse, and adding a class to the
catalog cannot fix it, because every class still maps onto one of the four. Instead, describe each object by
numbers (height, footprint, elongation, thinness, plus traits from the review) and let a skill learned on one
object carry over to similar ones. The shape becomes a label. This changes `prop_skills.py` and needs its own PR,
after PR3 and PR6.
