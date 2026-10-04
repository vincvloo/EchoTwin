# Learning a policy from demonstrations

Until now the robot's moves were a hand-written state machine whose four numbers (grip height, lift, drop, speed) were learned
from demos (`docs/TEACHING.md`). This page is about the other kind of learning: a **policy**, a small neural network that looks at the
state of the table and the arm at 20 Hz and says what to do next. It is trained by behaviour cloning on demonstrations.

It is a first, honest version. What it is and is not:

| | |
|---|---|
| **It is** | A state-based policy in the spirit of ACT: it predicts a chunk of the next 10 actions (half a second) and averages overlapping chunks at run time. Trained with an L1 loss on demonstrations. Runs without torch. |
| **It is not** | Full ACT (no images, no CVAE, no transformer): there is no camera data in the demos, and the expert is deterministic given the state. |
| **Data** | Thousands of steps made in bulk by the scripted skill in the simulation, plus the demos you record (human, video, practice) if you want. There are **no real-robot demos** in this repository. |

## What the policy sees and does

18 numbers per step (`echotwin/robot/policy_obs.py`): the tool position and commanded grip, the object's position, the goal, whether the pads
hold the object, the object's width, height and length, whether the goal is "on top of", the jaws' yaw, the time since the move started, and
the top of the thing it stacks on. All of these exist on a real arm with a camera (object position from the twin's belief or a camera).
It outputs the tool velocity (vx, vy, vz), the grip (0 or 1) and the jaws' yaw.

## Make the data

```bash
# 600 successful moves by the scripted skill; the executed motion is noisy, the recorded action is the expert's clean one (DART)
<robot python> -m echotwin.robot.demos --n 600 --noise 0.3 --out data/policy/sim_noise

# the demos you recorded in the dashboard (data/robot/episodes), with their source (human, video, practice)
<robot python> -m echotwin.robot.demos --episodes data/robot/episodes --out data/policy/my_demos
```

Old episodes (before PR8) have no yaw and no object measurements; they are marked `legacy`, filled with zeros and still usable.
About 1 s per move, so 600 take roughly 10 minutes.

## Train

Training needs torch, which lives in the perception environment (`PERCEPTION_PY`); the GPU is used if there is one (a 4 GB RTX 2050 is enough:
the network has about 0.3 million weights).

```bash
<perception python> -m echotwin.robot.train_policy data/policy/sim_noise data/policy/my_demos --human-weight 5 --out data/policy/act_lite.npz
```

`--human-weight 5` counts your own demonstrations five times as much as the simulated ones. The result is a plain numpy `.npz` and a
`.json` with the training and validation loss.

## Use

```bash
<robot python> -m echotwin.robot.skillcheck --policy data/policy/act_lite.npz --trials 12             # the 12-cell table, simulation
<robot python> -m echotwin.robot.skillcheck --policy data/policy/act_lite.npz --backend mock          # on the mock arm
```

In the app, put `POLICY=data/policy/act_lite.npz` in `.env`: "do" moves are then made by the policy. Imagining the move first and asking to be
shown still use the scripted skill. STOP stops the policy like any other move.

## What to expect, and what is missing

The numbers are in `docs/RESULTS.md`. Things to keep in mind:

- A policy trained on simulated, scripted demos copies the scripted skill; it cannot be better than its teacher, and it has no reason to work on a
  real arm. To get there you need real teleoperated demos (and images, so that it can see), then LeRobot's ACT is the natural next tool.
- The demo files use LeRobot's field names (`observation.state`, `action`) but are **not** a LeRobot dataset (no parquet, no metadata); no export exists.
- The policy has no memory of what it did beyond the time since the start. If it gets stuck it stays stuck until the time limit (600 ticks, 30 s).
- Millimetre tolerances (a 7 cm cylinder in 8.2 cm jaws) are hard to clone from a few hundred demos.
