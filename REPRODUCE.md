# Reproduce — NASA Valkyrie walking + obstacle course

Everything needed to reproduce the headline result
(`Videos/valkyrie_obstacle_course.mp4`) from scratch: build the robot, train the
policy, and record the rollout. Trained on one RTX 5090 (32 GB), `num_envs` up to
4096, PPO. Simulator step is 0.005 s, control step 0.02 s (50 Hz control).

## 0. Environment
```bash
# from the repo root
uv sync
# all commands below use `env -u PYTHONPATH uv run ...` so the bundled venv is used
# and any system PYTHONPATH is ignored.
```

## 1. Build the robot (URDF -> MJCF)
Valkyrie is not shipped with mjlab/IsaacLab; it is built from NASA's public
description.
```bash
git clone https://github.com/NASA-JSC-Robotics/val_description \
  src/mjlab/asset_zoo/robots/nasa_valkyrie/val_description
env -u PYTHONPATH uv run python \
  src/mjlab/asset_zoo/robots/nasa_valkyrie/build_mjcf.py
```
This converts the 75 Collada meshes to OBJ, strips Gazebo/transmission/sensor
tags, welds the finger + Hokuyo joints, fixes degenerate sensor-frame inertials,
and writes a clean MJCF (free-joint pelvis, feet-only collision, foot sites,
5-sensor pelvis IMU). Result: 135.9 kg, 32 actuators. Tasks auto-register on
import as `Mjlab-Velocity-Flat-NASA-Valkyrie` and `Mjlab-Velocity-Rough-NASA-Valkyrie`.

## 2. Train — three stages
The heavy humanoid will not step under a plain velocity reward (it sits in the
"stand and balance" optimum), and naive rough-terrain training regresses to
**marching in place** (see §4). The pipeline that works:

### Stage A — flat walker (from scratch)
```bash
WANDB_MODE=offline env -u PYTHONPATH uv run train \
  Mjlab-Velocity-Flat-NASA-Valkyrie \
  --env.scene.num-envs 4096 --agent.max-iterations 15000
```
The gait-phase clock (a shared phase in `[0,1)` that advances while the command
is non-zero, offset per foot to left/right anti-phase) + the `gait_phase_contact`
reward are what make it start walking rather than balance in place.

### Stage B — rough terrain + stair curriculum
Warm-start from the flat walker and ramp terrain difficulty by iteration
(near-flat +/-1 cm at the start, +1 level per ~1000 iters):
```bash
TERRAIN_RAMP=1e-3 LIN_VEL_ABS_W=-1.0 FOOTHOLD_W=2.0 \
WANDB_MODE=offline env -u PYTHONPATH uv run train \
  Mjlab-Velocity-Rough-NASA-Valkyrie \
  --env.scene.num-envs 4096 --agent.max-iterations 20000 \
  --agent.resume --agent.load-run <flat_run_dir>
```
- `LIN_VEL_ABS_W=-1.0` — L1 velocity-tracking penalty; the anti-marching term.
- `FOOTHOLD_W=2.0` — perceptive foothold reward, **gated on actual forward speed
  projected on the command** (not on the command magnitude — see §4).

### Stage C — anti-crouch flat cleanup
The foothold/swing shaping bleeds a permanent knee-bent crouch into flat walking.
Add a one-sided, stance-relative pelvis-height reward and continue, but keep it
**gentle** — strong anti-crouch + strong stagger drives the stiff-legged march
back (see §4):
```bash
FOOTHOLD_W=2.0 BASE_H_W=-5 STAGGER_W=-0.5 LIN_VEL_ABS_W=-1.0 BASE_H_TARGET=1.02 \
WANDB_MODE=offline env -u PYTHONPATH uv run train \
  Mjlab-Velocity-Rough-NASA-Valkyrie \
  --agent.resume --agent.load-run <rough_run_dir> \
  --agent.load-checkpoint <walking_checkpoint>.pt
```
Measured init-pose pelvis height above the stance foot is 1.056 m; the target is
1.02 m so a small nominal knee bend is not charged. Save the walking checkpoint
(the one used for the videos here walks flat 1.09 / pyramid 1.07 / ridge 0.63,
all confirmed by §3).

### Reward weights (rough task, as used for the headline result)
| reward | weight | | reward | weight |
|---|---|---|---|---|
| track_linear_velocity | 2.0 | | gait_phase_contact | 2.0 |
| track_angular_velocity | 2.0 | | gait_swing_lift | 0.5 |
| upright | 1.0 | | gait_mirror | -0.5 |
| pose | 1.0 | | feet_stagger | -1.5 (-> -0.5 in stage C) |
| body_ang_vel | -0.05 | | foot_flat_landing | 2.0 |
| angular_momentum | -0.02 | | base_height_above_feet | -20 (-> -5 in stage C) |
| dof_pos_limits | -1.0 | | arm_swing | 0.4 |
| action_rate_l2 | -0.1 | | foot_clearance | -2.0 |
| foot_swing_height | -0.25 | | foot_slip | -0.1 |
| self_collisions | -1.0 | | track_lin_vel_abs (L1) | -1.0 |

## 3. Verify it walks (not marches) — MANDATORY
ep_length and reward can stay high while the robot marches in place. Always
confirm real translation with a fixed forward command before trusting a
checkpoint:
```bash
MUJOCO_GL=egl env -u PYTHONPATH uv run python \
  src/mjlab/asset_zoo/robots/nasa_valkyrie/eval_track.py \
  --ckpt <checkpoint>.pt --vx 0.9 --steps 300 --envs 64
```
Reads body-frame forward speed projected on the command + world displacement and
prints a ratio: **>0.6 WALKS, 0.3-0.6 WEAK, <0.3 MARCHING.**

## 4. The marching trap (why the above is shaped this way)
Rough-terrain training kept regressing to a high-reward march in place. Two causes,
both fixed:
1. **Foothold reward gated on the command, not the motion.** The robot earned the
   full foothold reward (~+2.8/step) while stepping in place on hard stairs,
   dwarfing the velocity penalty. Fix: gate the reward on
   `track = clamp(proj / cmd_speed, 0, 1)` where `proj` is the actual velocity
   projected on the command direction — no motion, no foothold pay.
2. **Anti-crouch too strong.** `base_height_above_feet=-20` + `feet_stagger=-1.5`
   run long rewards a stiff, high-pelvis march. Keep anti-crouch ~ -5 and stagger
   ~ -0.5, and watch §3 every few hundred iters; kill if the ratio drops below 0.6.

## 5. Record the obstacle course (the headline video)
The recorder builds a single continuous course from env vars: approach -> stairs
up / top / down -> +/-10 cm random-box rough field -> ramp up-and-over -> long
flat exit. Camera tracks the torso.
```bash
REC_COURSE=1 SCAN_VIZ=0 MUJOCO_GL=egl \
STAIR_H=0.14 STAIR_TREAD=0.30 STAIR_NSTEPS=5 STAIR_TOPFLAT=2.0 STAIR_APPROACH=2.5 \
COURSE_GAP=1.0 ROUGH_LEN=6.0 ROUGH_CELL=0.5 ROUGH_H=0.10 \
RAMP_RUN=3.0 RAMP_H=0.6 COURSE_EXIT=6.0 \
env -u PYTHONPATH uv run python \
  src/mjlab/asset_zoo/robots/nasa_valkyrie/record_walk.py \
  --ckpt <walking_checkpoint>.pt --out course_full.mp4 \
  --task Mjlab-Velocity-Rough-NASA-Valkyrie \
  --steps 1400 --vx 0.9 --fix-yaw --azimuth 90 --width 1280 --height 720
# The policy traverses the whole course by ~n=1200; trim before it walks off the
# far edge of the exit pad:
ffmpeg -i course_full.mp4 -t 24.0 -c:v libx264 -pix_fmt yuv420p -crf 18 -an \
  Videos/valkyrie_obstacle_course.mp4
```
Recording knobs: `ROUGH_H` = +/- rough cell height (0.10 = +/-10 cm),
`STAIR_H`/`STAIR_NSTEPS` = step height / count, `RAMP_H`/`RAMP_RUN` = ramp rise /
run per side, `COURSE_EXIT` = flat exit length. `SCAN_VIZ=0` hides the height-scan
debug markers. Other showcase terrains use `REC_STAIRS=1` (inverted-pyramid stairs)
and `REC_RIDGE=1` (up -> flat -> down ridge) with the same `STAIR_*` knobs.
```

## Provenance of the shipped video
`Videos/valkyrie_obstacle_course.mp4` is the stage-B walking checkpoint (before the
stage-C anti-crouch cleanup) recorded with the §5 command and trimmed to 24 s. It
walks flat 1.09 / inverted-pyramid stairs 1.07 / up-flat-down ridge 0.63 by the §3
metric — all in the WALKS band.
