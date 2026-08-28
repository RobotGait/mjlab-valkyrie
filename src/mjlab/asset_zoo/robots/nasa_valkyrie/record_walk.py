"""Headless rollout recorder for the NASA Valkyrie velocity policy.

Loads the play env (flat), a trained checkpoint, drives a scripted command
schedule (forward / strafe / turn), renders rgb_array frames from a configurable
camera and writes an mp4. No viewer, exits cleanly.

Examples (from mjlab root):
  # single steady forward clip
  MUJOCO_GL=egl WANDB_MODE=offline env -u PYTHONPATH uv run python \
    src/mjlab/asset_zoo/robots/nasa_valkyrie/record_walk.py \
    --ckpt <model.pt> --out out.mp4 --steps 400 --vx 0.8

  # high-res showcase (forward+strafe+turn), side camera
  ... record_walk.py --ckpt <model.pt> --out side.mp4 --showcase \
      --width 1280 --height 720 --azimuth 90 --elevation -10 --distance 3.5
"""

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# label, vx, vy, wz, n_steps  (50 steps ~= 1 s at step_dt 0.02)
SHOWCASE = [
  ("forward 1.0 m/s", 1.0, 0.0, 0.0, 130),
  ("strafe left", 0.0, 0.6, 0.0, 100),
  ("strafe right", 0.0, -0.6, 0.0, 100),
  ("turn left", 0.2, 0.0, 0.6, 110),
  ("turn right", 0.2, 0.0, -0.6, 110),
  ("forward 1.0 m/s", 1.0, 0.0, 0.0, 100),
]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--steps", type=int, default=500)
  ap.add_argument("--vx", type=float, default=0.8)
  ap.add_argument(
    "--showcase",
    action="store_true",
    help="run the forward/strafe/turn schedule instead of steady vx",
  )
  ap.add_argument("--width", type=int, default=320)
  ap.add_argument("--height", type=int, default=240)
  ap.add_argument("--azimuth", type=float, default=90.0)
  ap.add_argument("--elevation", type=float, default=-20.0)
  ap.add_argument("--distance", type=float, default=3.5)
  ap.add_argument(
    "--fix-yaw",
    action="store_true",
    help="spawn facing +x (deterministic camera framing)",
  )
  ap.add_argument("--task", default="Mjlab-Velocity-Flat-NASA-Valkyrie")
  args = ap.parse_args()

  import mjlab.tasks  # noqa: F401  (registers tasks)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.viewer.viewer_config import ViewerConfig

  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1

  # Camera: track the torso, high-res, near-level framing.
  v = env_cfg.viewer
  v.origin_type = ViewerConfig.OriginType.ASSET_BODY
  v.entity_name = "robot"
  v.body_name = "torso"
  v.width = args.width
  v.height = args.height
  v.azimuth = args.azimuth
  v.elevation = args.elevation
  v.distance = args.distance
  v.enable_shadows = True
  v.enable_reflections = True

  # Deterministic facing so a fixed camera azimuth gives a clean front/side.
  if args.fix_yaw and "reset_base" in env_cfg.events:
    pr = env_cfg.events["reset_base"].params["pose_range"]
    pr["yaw"] = (0.0, 0.0)
    pr["x"] = (0.0, 0.0)
    pr["y"] = (0.0, 0.0)
    pr["z"] = (0.03, 0.03)  # fully deterministic spawn -> front/side stay in sync

  # Never auto-resample; we inject the command by hand each step.
  twist = env_cfg.commands["twist"]
  twist.resampling_time_range = (1e9, 1e9)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(args.task).clip_actions)

  agent_cfg = load_rl_cfg(args.task)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(args.ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  print(f"[INFO] loaded {args.ckpt}")

  term = env.unwrapped.command_manager.get_term("twist")

  def set_cmd(vx, vy, wz):
    term.is_standing_env[:] = False
    term.is_heading_env[:] = False
    term.is_world_env[:] = False
    term.is_forward_env[:] = False
    c = torch.tensor([[vx, vy, wz]], device=device, dtype=term.vel_command_b.dtype)
    term.vel_command_b[:] = c
    term.vel_command_w[:] = c

  if args.showcase:
    plan = SHOWCASE
  else:
    plan = [(f"forward {args.vx} m/s", args.vx, 0.0, 0.0, args.steps)]

  obs, _ = env.reset()
  frames = []
  segments = []  # (label, start_frame, end_frame)
  for label, vx, vy, wz, n in plan:
    start = len(frames)
    for _ in range(n):
      set_cmd(vx, vy, wz)
      with torch.inference_mode():
        act = policy(obs)
      obs, _, _, _ = env.step(act)
      set_cmd(vx, vy, wz)  # keep obs/reward command fixed for next step too
      frame = env.unwrapped.render()
      if frame is not None:
        frames.append(np.asarray(frame))
    segments.append((label, start, len(frames)))
  print(f"[INFO] captured {len(frames)} frames")

  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  import imageio.v2 as imageio

  fps = int(round(1.0 / env.unwrapped.step_dt))
  imageio.mimwrite(args.out, frames, fps=fps, quality=9)
  # Emit segment boundaries (for ffmpeg drawtext labelling downstream).
  seg_path = Path(args.out).with_suffix(".segments.txt")
  seg_path.write_text("\n".join(f"{s}\t{e}\t{fps}\t{lbl}" for lbl, s, e in segments))
  print(f"[INFO] wrote {args.out} ({len(frames)} frames @ {fps}fps)")
  for lbl, s, e in segments:
    print(f"[SEG] {s / fps:5.1f}-{e / fps:5.1f}s  {lbl}")
  os._exit(0)


if __name__ == "__main__":
  main()
