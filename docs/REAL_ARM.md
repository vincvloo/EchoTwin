# Running on a real arm (and what is not verified)

EchoTwin has one robot contract (`echotwin/robot/backend.py`) and two ways to fulfil it. Skills, practice, replay and the
dashboard do not know which one they talk to.

| `BACKEND=` | What moves | Needs |
|---|---|---|
| `sim` (default) | The simulation is the robot | Nothing |
| `real` with no `REAL_PORT` | The **mock arm**: a second, perturbed simulation (lag, encoder offsets, heavier objects, half speed) | Nothing |
| `real` with `REAL_PORT=COM3` (or `/dev/ttyACM0`) | An SO-ARM100 / SO-101 on a Feetech servo bus | The arm, `pip install feetech-servo-sdk`, a calibrated descriptor |

If the port cannot be used (no SDK, port busy, arm not calibrated) the app starts on the simulation and says why. The dashboard
shows `sim`, `mock arm` or `REAL ARM` next to the connection badges.

## How the real back-end works

The simulation keeps running beside the arm as its **twin**. Each tick (20 times a second):

1. The twin's controller turns the wanted tool motion into five joint setpoints and a gripper command (the same code as the sim).
2. The driver sends them to the servos and waits one tick.
3. The measured joints are read back. The tool position is the forward kinematics of those joints.
4. The twin's physics follows the measured joints, so it also predicts what the pads hold and where a carried object goes.

Moves are imagined on the twin first, as always. A real arm moves at **half the simulation's tool speed**. "Stop" (button, word,
or shaking the phone) turns the servos' torque off. The first move after connecting waits for **"arm the robot"** (a button on the
dashboard appears), unless `REAL_REQUIRE_GO=0`.

## What is verified, and what is not

Verified here (tests, mock arm, fake servo bus): the contract is complete on both back-ends; the default skills run on the mock
arm (`python -m echotwin.robot.skillcheck --backend both`, in `docs/RESULTS.md`); stop and resume; the arm gate; the fallbacks; the
calibration maths and the command flow of the servo driver; every executed move is logged per back-end (`python -m echotwin.robot.runlog`).

**Not verified: any real hardware.** The servo driver (`feetech.py`) was written from the protocol and the SO-ARM100's layout and
has never talked to a servo. The calibration of a physical arm, real friction and grasps, and the speed limits are untested. Treat
the first connection as a test of the driver, with the arm clamped down, your hand on the power switch, and a clear table.

## Connecting an SO-ARM100 / SO-101 (checklist)

1. Mount the arm facing into the table, at the position the simulation expects (the front edge, `BASE_INSET` in `world.py`;
   7 cm in from the front edge of the table, centred). Clamp it. Placing the base by the marker is not done yet.
2. `pip install feetech-servo-sdk` in the robot environment. Find the port (Windows: Device Manager, COM number).
3. Fill in the `real` block of `echotwin/robot/arms/so_arm100.json` (or copy the file for your arm and set `ARM=path/to/it.json`):
   servo `ids`, `gripper_id`, `baudrate`.
4. **Calibrate.** Hold the arm by hand in the simulation's home pose (`home` in the descriptor: upright, forearm folded),
   run `python -m echotwin.robot.feetech --port COM3 --read` (it switches torque off), and copy each reading into `zero_ticks`.
   Move one joint by hand at a time and check that the reading rises when the simulation's angle rises; set `direction` to
   `-1` where it falls. Read the gripper at fully open and fully closed into `gripper_ticks`.
5. Set `"calibrated": true`. Until then the driver refuses to move anything.
6. In `.env`: `BACKEND=real` and `REAL_PORT=COM3`. Start the server. The dashboard shows `REAL ARM`; press **Arm the robot**.
7. First tests, slowly: say "go home", then ask for one small move of a light object. Watch the arm, not the screen.
8. Stop works: press STOP during a move and check that the arm goes limp.

Safety notes: joint targets are clamped to the simulation's joint limits and to the reachable workspace, the table edge is
a limit, and the speed is halved. None of this replaces a physical power switch within reach. A limp arm falls: support it when you
press stop.

## Objects on a real table

The real back-end *believes* where objects are (the twin's contacts predict them). Real objects drift, slip, and sit slightly off
the scan, so with `CLOSED_LOOP` on (the default for a real arm) every move looks first, checks as it goes, and retries:

1. The arm parks beside its base, tool high, out of the camera's view of the reachable table.
2. `observe`: a frame from the camera, a figure-ground search for the object near where it is believed to be
   (`features/locate.py`: the blob that differs from the table around it), its place on the table through a `PlaneMap`. The
   twin's belief is moved to what was seen. If the object is not seen the robot says so and stops.
3. The pick is planned from what was seen. After the grip it checks that the real jaws closed on something (they must stop where
   the twin's jaws stop on the object; jaws that go further met nothing), after the lift that it is still held, and at the end it looks again.
4. A failed check opens the jaws, lifts, parks and looks again: at most 2 retries, then it asks to be shown.

**Lining the gripper up (`ALIGN=on`, off by default).** Joint readings are never exact: an encoder that is a degree off puts the tool a centimetre away from where
the arm thinks it is. Before the descent the robot therefore looks at the gripper itself: at the hover over the object, just above it, and at
the grip height it finds the two pads in the camera image (the darkest compact things near the tool, about a jaw opening apart), takes the
midpoint, compares it with where the joints say the tool is, and moves the pick by the difference (up to 3 looks per level). If the gripper
is more than 3 cm off, or cannot be seen, it says so (and carries on without, or asks to be shown). **On a real arm the jaws must be easy to
see from the camera: two dark or brightly coloured pads, or a mark on each jaw.** The pad finder is a threshold on grey level tuned on the
render; real images will need it re-tuned.

What exists for a real camera: `CameraSource(index)` (OpenCV) and `PlaneMap.from_homography(H)`. **What does not:** nothing wires
them into `backend.make`, and no calibration produces `H` (the marker of PR10 could, with the table frame of the twin: not done). The
simulation's own camera is the only one tested. Without a camera the real arm still runs the jaw checks and the retries, but "look"
returns what the twin believes.
