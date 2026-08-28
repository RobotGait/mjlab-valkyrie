from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  valkyrie_flat_env_cfg,
  valkyrie_rough_env_cfg,
)
from .rl_cfg import valkyrie_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-NASA-Valkyrie",
  env_cfg=valkyrie_rough_env_cfg(),
  play_env_cfg=valkyrie_rough_env_cfg(play=True),
  rl_cfg=valkyrie_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-NASA-Valkyrie",
  env_cfg=valkyrie_flat_env_cfg(),
  play_env_cfg=valkyrie_flat_env_cfg(play=True),
  rl_cfg=valkyrie_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
