# Arms: use the one you have

An arm sits on a base: fixed at the table edge, or on wheels. Robots (a base and an arm) are in `docs/ROBOTS.md`.

The simulated robot is an arm with a two-pad gripper. The arm is not hard-coded: a small JSON file describes it, and
`echotwin/robot/arm.py` builds it into the scene. Pick one with `ARM=` in `.env`.

| `ARM=` | What it is |
|---|---|
| (empty) or `builtin` | A 5-joint arm written in code, SO-101 sized, two parallel pads, opens 8 cm. Works offline. All numbers in `docs/RESULTS.md` are for this one. |
| `so_arm100` | The SO-ARM100 from MuJoCo Menagerie (Apache-2.0), one moving jaw. Fetch it first (below). Passes `--check`, not yet the full skill table. |
| `path/to/my_arm.json` | Your own arm (relative to the repository, or absolute). |

```bash
python -m echotwin.robot.arm --list                    # which arms are described and which files are missing
python -m echotwin.robot.arm --download so_arm100      # about 3.4 MB into models/arms/ (git-ignored), nothing else downloads
python -m echotwin.robot.arm --check so_arm100         # loads it, prints its reach, tries a 4 cm cube and a 6 cm cylinder
```

## What the robot does with an arm

- The skills still say "tool to (x, y, z), jaws open or closed". Inverse kinematics (tool position plus "tool points
  down", the wrist roll lines the jaws up with the object) turns that into joint targets.
- The reachable ring is measured once per arm. A request outside it, or an object the jaws cannot hold, is declined
  with the reason ("it is 12 cm wide and my gripper opens 8 cm", "it is out of my reach").
- An object counts as held only when two pads squeeze it. It can slip, drop or tip over, as it would on hardware.

## Describe your own arm

Copy `echotwin/robot/arms/builtin.json` and change it. You need an MJCF (MuJoCo XML) of the arm with position
actuators, and these fields:

| Field | Meaning |
|---|---|
| `mjcf` | Path of the MJCF, relative to the repository, or `builtin` |
| `base_body` | The root body of the arm (it is attached to the table at the front edge) |
| `mount_yaw_deg` | Turn the arm about the vertical axis so it faces into the table (try 0 or 180) |
| `joints`, `actuators` | Exactly five each, in order: base rotation, three bending joints, wrist roll |
| `gripper` | `mode` `parallel` (two sliding pads) or `single` (one moving jaw, set `fixed_side` to 1 or -1), the gripper `actuators`, and their `open` and `closed` values |
| `tool` | `body` and `pos` (the point between the pads, in that body's frame), `point_axis` (the tool's axis that must point down), `close_axis` (the direction the jaws close along) |
| `pads` | Parts of the pad geom names (used to see what is held) |
| `home` | Five joint values to start from |
| `max_opening` | How far the jaws open, in metres (objects wider than this are declined) |
| `download` | Optional: repo, commit, path and files for `--download` |

Names are looked up in your MJCF, so the order of things inside it does not matter. A wrong name gives an error that
says which one. Then run `python -m echotwin.robot.arm --check path/to/my_arm.json`. If the cube or cylinder fails,
look at the grip first: the pads need friction (1.5 or more), the closing force a few newtons, and the tool point
a few millimetres above the lowest tip of the pads.

Credit: the SO-ARM100 model is by The Robot Studio and Google DeepMind (MuJoCo Menagerie), Apache-2.0.

## Switching the arm, and what size of arm it would take

In the dashboard, open Advanced, then "Arm size: what would it take?" (or press "What arm size would grab it?" when Pip
declines an object). The arm list shows every descriptor in `echotwin/robot/arms/` (an arm whose files are missing is
greyed out until you download it); switching rebuilds the twin with the other arm and keeps the table. It takes a few
seconds, and the choice is not saved: `ARM=` in `.env` is what the server starts with. Skills learned with one arm are not
checked on another.

**Arm size.** The slider makes the arm 0.75 to 4 times its described size; the table and the objects keep their sizes. For
each object the card says what size it needs: big enough for the jaws to open around it and to reach it, not so big that it
cannot bend down to it, or why no size works (thinner than 15 mm cannot be pinched from a table). "Size for every object it
can grab" applies the smallest size that takes them all.

How the arm is resized (`arm.resize`): lengths times k, masses times k^3, inertias times k^5; torque gains, damping and
force limits of the joints times k^5 (a sliding joint k^3 and k^4), so the bigger arm moves the same way relative to its
size. The jaws are the exception: their squeeze only grows with k, because it holds the object, whose weight does not
change (a gripper squeezing k^4 times harder pops a light round thing out). The reach grid and the base's distance from the
table edge grow with k.

Measured with the scripted skill (`PS.imagine`, the same check as `--check`):

| Arm | Sizes | Result |
|---|---|---|
| builtin, objects scaled with the arm (4 cm cube, 6 cm cylinder) | 0.75, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 4 | all picked and placed |
| builtin, a 15 cm box and a 12 cm round case at their size | 1 / 1.5 | declined (wider than the jaws) |
| builtin, the same | 2, 2.5, 3, 4 | all picked and placed |
| so_arm100, objects scaled with the arm | 1, 1.5, 2 | the cube yes; the cylinder lands 6 to 25 cm off from 1.5 |
| so_arm100 | 3 | fails |

So the size a card shows is a geometric answer for any arm; that the resized arm then also moves the object is measured for
the built-in arm only. Below 0.75 the built-in arm can hardly reach the table, so the slider stops there.

Sizes of the objects: from one photo they are estimates. If you know how wide one object really is, "Rescale the table"
scales the table, its objects and the camera by one factor; the arm does not change. Any factor is fine as long as the
result is plausible: every thing between 5 mm and 2 m, the mapped area at most 20 m wide.
