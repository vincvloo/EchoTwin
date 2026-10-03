# History

Things that were built, worked, and were taken out. The code is still in git; the notes are kept here.

## Colour blocks and the A4 sheet (removed in PR1)

The first robot demo moved three coloured cubes into a green zone and a blue tray, with an A4 sheet as the scale
reference and a keypoint imitation policy. The everyday-object path (any object, learning per shape) replaced it.

## The sonar mobile base (removed in PR7)

A differential-drive robot that localised itself in the scanned room with four ultrasonic sensors and a particle
filter, in simulation. It worked (about 1 cm error once localised on the real maps) and was benchmarked, but it was a
separate product from the table-top arm, and EchoTwin is about the arm. Removed: the sonar model, the robot, the
particle filter, the run and benchmark scripts, the synthetic test room, the pitch deck and the browser demo.

- Last commit with all of it: `8317b82` (`git checkout 8317b82`).
- [sonar/RESULTS.md](sonar/RESULTS.md): benchmark and what it said.
- [sonar/RESEARCH.md](sonar/RESEARCH.md): sources and related projects.
- [sonar/TASKS.md](sonar/TASKS.md): the old task list (browser simulator, goal navigation, harsher sensor model, ROS 2).

What stayed because the arm side needs it: levelling a cloud and finding the floor (`perception/mapping.py`), the floor
plan grid the object map is built on (`perception/gridmap.py`), and the object map itself.
