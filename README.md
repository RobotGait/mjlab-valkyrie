# NASA Valkyrie in mjlab — velocity walking

> **Fork of [mjlab](https://github.com/mujocolab/mjlab) (Apache-2.0).**
> Added by me: integration of the **NASA Valkyrie (R5)** humanoid + a velocity-tracking
> training environment + a trained walking policy + a rollout video.
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
- **Trained a walking policy** (PPO, flat terrain) and recorded a rollout.

Files I added:
- `src/mjlab/asset_zoo/robots/nasa_valkyrie/` — MJCF build script, constants, MJCF + meshes.
- `src/mjlab/tasks/velocity/config/nasa_valkyrie/` — env + PPO configs; auto-registers on import.

## Result
![Valkyrie walking in mjlab](Videos/valkyrie_walk.mp4)

Trained on flat terrain, RTX 5090, `num_envs=4096`, PPO.

<!-- TODO before final publish: swap in the mature-checkpoint clip and fill the numbers below.
     Current clip is an interim early-training rollout. -->
<!-- final iter / mean episode length / track-velocity reward / fall-rate / best ckpt -->

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
# train
WANDB_MODE=offline env -u PYTHONPATH uv run train Mjlab-Velocity-Flat-NASA-Valkyrie \
  --env.scene.num-envs 4096 --agent.max-iterations 15000
# play / record
env -u PYTHONPATH uv run play Mjlab-Velocity-Flat-NASA-Valkyrie --video True
```

## Licensing / attribution
- Framework: **mjlab** — Apache-2.0 (upstream; core unmodified). See `LICENSE`.
- Robot model: **NASA Valkyrie `val_description`** — NASA Open Source Agreement (NOSA) v1.3.
  Redistribution + modification are permitted; my modifications are noted and the original
  NASA copyright is retained. See `NOTICE`.
- My additions (Valkyrie asset integration, velocity env, training config): released under the
  same terms as the surrounding files.

---
**Have a robot URDF? I can get it walking in sim like this.** — robot RL / Sim2Real contract work.
