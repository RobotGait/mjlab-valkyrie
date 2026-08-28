# NASA Valkyrie in mjlab — velocity walking

> **Fork of [mjlab](https://github.com/mujocolab/mjlab) (Apache-2.0).**
> Added by me: integration of the **NASA Valkyrie (R5)** humanoid + a velocity-tracking
> training environment, a gait-phase clock + gait/arm-swing reward shaping, a rough-terrain
> stair-climbing curriculum, and trained walking policies with rollout videos.
> mjlab itself (the RL framework / sim / training loop) is upstream work, not mine.

## What I added (my contribution)
Valkyrie is **not** a robot that ships with mjlab or IsaacLab. I brought it in from scratch:

- **URDF → MJCF integration.** Took the public NASA Valkyrie description
  (`val_description`, NASA Open Source Agreement v1.3), converted 75 Collada meshes to OBJ,
  stripped Gazebo/transmission/sensor tags, welded the 26 finger joints and the Hokuyo
  (not needed for locomotion), fixed degenerate sensor-frame inertials, and rebuilt a clean
  MJCF with a free-joint pelvis, **feet-only collision**, foot sites, and a 5-sensor pelvis IMU
  (gyro / velocimeter / accelerometer / frame-z-axis / subtree angular momentum).
  Total mass **135.9 kg** (matches the real robot ≈129–136 kg → the inertials are right).
  **32 actuators** (legs 12 / torso 3 / arms 14 / neck 3), effort/velocity limits from the URDF.
- **Velocity-tracking tasks** registered as `Mjlab-Velocity-Flat-NASA-Valkyrie` and
  `Mjlab-Velocity-Rough-NASA-Valkyrie` (asset zoo + task config; PD gains and a
  crouch-stand keyframe tuned for Valkyrie).
- **Gait phase clock.** A heavy humanoid trained with a plain velocity reward sits in the
  "stand and balance" local optimum and never steps. I added an optional phase clock to the
  velocity command (a shared phase in `[0,1)` that advances while the command is non-zero,
  offset per foot to left/right anti-phase), exposed it in the observation, and added a
  contact-vs-schedule reward. That is what makes it start walking.
- **Reward shaping.** A left/right mirror cost (penalizes one-sided stepping / roll wobble)
  and a natural arm counter-swing reward (each shoulder pitch tracks the negative of the
  same-side hip pitch, so the arms swing anti-phase with the legs).
- **Rough-terrain stair curriculum.** Terrain that is mostly random boxes (>50% of tiles) plus
  climbable stairs and a little uniform noise, with an **iteration-based difficulty ramp**:
  starts near-flat (±1 cm) and raises the box/stair height by one level every ~1000 iterations.
- **Trained walking policies** (flat + rough) and recorded rollouts.

Files I added / touched:
- `src/mjlab/asset_zoo/robots/nasa_valkyrie/` — MJCF build script, constants, MJCF + meshes,
  and a headless multi-camera rollout recorder (`record_walk.py`).
- `src/mjlab/tasks/velocity/config/nasa_valkyrie/` — env + PPO configs; auto-registers on import.
- `src/mjlab/tasks/velocity/mdp/` — small, self-contained additions to the velocity task MDP:
  a gait phase clock on the velocity command, the `gait_phase_sincos` observation, the
  `gait_phase_contact` / `gait_phase_swing_height` / `gait_lr_mirror_cost` / `arm_swing`
  rewards, and the `terrain_levels_time` iteration-based terrain curriculum.

## Result
https://github.com/RobotGait/mjlab-valkyrie/blob/main/Videos/valkyrie_stairs_10cm_perception.mp4

`Videos/valkyrie_stairs_10cm_perception.mp4` — the perceptive policy climbing a 10 cm
staircase using the terrain height-scan; the cyan dots are the scan points the robot senses.
`Videos/valkyrie_boxfield_5cm_perception.mp4` — the same policy walking across a field of
ankle-height (±5 cm) random boxes.

Trained on RTX 5090, `num_envs=4096`, PPO with the `terrain_levels_time` curriculum. Starting
from a flat walker, the terrain height-scan is spliced into the network (zero-initialised, so
day-one behaviour is unchanged) and the terrain difficulty is ramped so the robot learns to
see and step. A fixed forward-velocity rollout confirms the policy actually tracks the command
— real translation at ~0.4–0.6 m/s with zero falls — rather than stepping in place.

## Reproduce
First fetch NASA's public Valkyrie description and build the MJCF:
```bash
# NASA val_description (NOSA v1.3) — public, redistribution + modification permitted
git clone https://github.com/NASA-JSC-Robotics/val_description \
  src/mjlab/asset_zoo/robots/nasa_valkyrie/val_description
uv run python src/mjlab/asset_zoo/robots/nasa_valkyrie/build_mjcf.py
# (or set VAL_DESCRIPTION=/path/to/val_description)
```
Then train / play:
```bash
# train on flat ground
WANDB_MODE=offline env -u PYTHONPATH uv run train Mjlab-Velocity-Flat-NASA-Valkyrie \
  --env.scene.num-envs 4096 --agent.max-iterations 15000

# train on rough terrain + stairs (iteration-based difficulty ramp)
TERRAIN_RAMP=1e-3 WANDB_MODE=offline env -u PYTHONPATH uv run train \
  Mjlab-Velocity-Rough-NASA-Valkyrie --env.scene.num-envs 4096 --agent.max-iterations 20000

# record a front+side showcase clip (forward / strafe / turn)
MUJOCO_GL=egl WANDB_MODE=offline env -u PYTHONPATH uv run python \
  src/mjlab/asset_zoo/robots/nasa_valkyrie/record_walk.py \
  --ckpt <model.pt> --out showcase.mp4 --showcase --fix-yaw \
  --azimuth 90 --width 1280 --height 720
```

## Licensing / attribution
- Framework: **mjlab** — Apache-2.0 (upstream). The RL framework / sim / training loop are
  unchanged; my only edits to shared files are small, self-contained additions to the velocity
  task MDP (listed above). See `LICENSE`.
- Robot model: **NASA Valkyrie `val_description`** — NASA Open Source Agreement (NOSA) v1.3.
  Redistribution + modification are permitted; my modifications are noted and the original
  NASA copyright is retained. See `NOTICE`.
- My additions (Valkyrie asset integration, velocity env, training config): released under the
  same terms as the surrounding files.

---
**Have a robot URDF? I can get it walking in sim like this.** — robot RL / Sim2Real contract work.
