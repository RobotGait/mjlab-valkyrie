"""Headless rollout recorder for the NASA Valkyrie velocity policy.

Loads the play env (flat), a trained checkpoint, commands a steady forward
walk, renders rgb_array frames and writes an mp4. No viewer, exits cleanly.

Usage (from mjlab root):
  MUJOCO_GL=egl WANDB_MODE=offline env -u PYTHONPATH uv run python \
    src/mjlab/asset_zoo/robots/nasa_valkyrie/record_walk.py \
    --ckpt <path/model_XXXX.pt> --out <out.mp4> --steps 500 --vx 0.8
"""
import argparse
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--vx", type=float, default=0.8)
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-NASA-Valkyrie")
    args = ap.parse_args()

    import mjlab.tasks  # noqa: F401  (registers tasks)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1

    # Lock command to a steady forward walk for a clean portfolio clip.
    twist = env_cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (args.vx, args.vx)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(twist, "rel_standing_envs"):
        twist.rel_standing_envs = 0.0

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(args.task).clip_actions)

    agent_cfg = load_rl_cfg(args.task)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(args.ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    print(f"[INFO] loaded {args.ckpt}")

    obs, _ = env.reset()
    frames = []
    for i in range(args.steps):
        with torch.inference_mode():
            act = policy(obs)
        obs, _, _, _ = env.step(act)
        frame = env.unwrapped.render()
        if frame is not None:
            frames.append(np.asarray(frame))
    print(f"[INFO] captured {len(frames)} frames")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    fps = int(round(1.0 / env.unwrapped.step_dt))
    imageio.mimwrite(args.out, frames, fps=fps, quality=8)
    print(f"[INFO] wrote {args.out} ({len(frames)} frames @ {fps}fps)")
    os._exit(0)


if __name__ == "__main__":
    main()
