# Task backlog

[x] done, [ ] open

- [x] Core pipeline in simulation (map, sonar, particle filter, benchmark)
- [x] Real data: photos -> VGGT -> map and objects (lounge, table)
- [x] Perception web app, bridge to robot twin, robot server with everyday objects
- [x] PR1 Merge Sonar and Robot into EchoTwin, remove the blocks demo
- [x] PR2 Shared `scene.json`, class catalog for any detected object, bridge fix (table window, real scale, obstacles), licence files
- [ ] Table texture: colour top-down image of the table from the cloud (the loader drops colours today)
- [x] PR3 NVIDIA review of YOLO detections
- [x] PR4 Phone scan runs the 3D pipeline, with a "skip, quick mode" button
- [x] PR5 Teaching guide (how to film and how it is processed) and the capture-comparison info button
- [x] PR6 Detector benchmark: YOLO11 vs open-vocabulary and newer models (text-prompt YOLOE is now the preferred detector)
      Result: open vocabulary with text prompts removes the 80-class limit (`docs/RESULTS.md`). Still open: a non-YOLO
      model (OWLv2, Grounding DINO: permissive licences, would also remove the AGPL dependency) and small tabletop objects.
- [ ] Sonar extras (browser simulator, goal navigation, harsher sensor model, ROS 2): see `TASKS_SONAR.md`
