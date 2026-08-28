"""NASA Valkyrie velocity environment configurations.

Mirrors the G1 velocity config but remaps names to Valkyrie's kinematics
(torso body, leftFoot/rightFoot, camelCase joints, feet-only collision).
"""

import os

from mjlab.asset_zoo.robots import (
  VALKYRIE_ACTION_SCALE,
  get_valkyrie_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TORSO = "torso"


def valkyrie_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create NASA Valkyrie rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_valkyrie_robot_cfg()}

  # Raycast sensor frame -> pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = ("left_foot_collision", "right_foot_collision")

  # Foot height scan wired to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.05, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(leftFoot|rightFoot)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    # Rough terrain: boxes as the MAIN surface (>50% of columns) plus stairs to
    # climb and a little uniform noise. Difficulty is time-ramped, not
    # distance-based: a gait-clocked all-direction robot cancels net travel, so
    # the walked-distance promotion never fires. Start ~flat (level 0) and raise
    # one level (= +0.5 cm box height / +stair height) every 1000 iterations via
    # terrain_levels_time (set TERRAIN_RAMP=1e-3 at launch).
    from mjlab.terrains.primitive_terrains import BoxRandomGridTerrainCfg

    tg = cfg.scene.terrain.terrain_generator
    tg.curriculum = True
    tg.num_rows = 11  # levels 0..10 -> box height 1cm..6cm in 0.5cm steps
    tg.size = (8.0, 8.0)
    cfg.scene.terrain.max_init_terrain_level = 0  # everyone starts near-flat

    subs = tg.sub_terrains
    # Box grid = main terrain. grid_height interpolates HMIN..HMAX across levels:
    # level 0 = +/-1cm, level 10 = +/-6cm (0.5cm per level).
    subs["boxes"] = BoxRandomGridTerrainCfg(
      proportion=float(os.environ.get("BOX_PROP", "0.55")),
      grid_width=float(os.environ.get("BOX_GRID_W", "0.30")),
      grid_height_range=(
        float(os.environ.get("BOX_HMIN", "0.01")),
        float(os.environ.get("BOX_HMAX", "0.06")),
      ),
      platform_width=1.5,
      merge_similar_heights=True,
    )
    # Stairs to climb (up + inverted/down), step height grows with level.
    _sh = float(os.environ.get("STAIR_HMAX", "0.12"))
    for k in ("pyramid_stairs", "pyramid_stairs_inv"):
      if k in subs:
        subs[k].step_height_range = (0.0, _sh)
        subs[k].proportion = 0.125
    # A little uniform noise (rough ground) so feet learn to clear unevenness.
    if "random_rough" in subs:
      subs["random_rough"].proportion = 0.10
      subs["random_rough"].noise_range = (0.005, 0.05)
    if "flat" in subs:
      subs["flat"].proportion = 0.10
    # Drop slopes/waves — focus on boxes + stairs.
    for k in ("hf_pyramid_slope", "hf_pyramid_slope_inv", "wave_terrain"):
      if k in subs:
        subs[k].proportion = 0.0

    # Iteration-based difficulty ramp instead of walked-distance promotion.
    if "terrain_levels" in cfg.curriculum:
      cfg.curriculum["terrain_levels"].func = mdp.terrain_levels_time

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = VALKYRIE_ACTION_SCALE

  cfg.viewer.body_name = TORSO

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.30

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO,)

  # Per-joint pose regularization std (Valkyrie camelCase joint names).
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*HipPitch": 0.3,
    r".*HipRoll": 0.15,
    r".*HipYaw": 0.15,
    r".*KneePitch": 0.35,
    r".*AnklePitch": 0.25,
    r".*AnkleRoll": 0.1,
    r"torsoYaw": 0.2,
    r"torsoRoll": 0.08,
    r"torsoPitch": 0.1,
    r".*ShoulderPitch": 0.15,
    r".*ShoulderRoll": 0.15,
    r".*ShoulderYaw": 0.1,
    r".*ElbowPitch": 0.15,
    r".*ForearmYaw": 0.2,
    r".*Wrist.*": 0.3,
    r".*NeckPitch": 0.2,
    r"neckYaw": 0.2,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*HipPitch": 0.5,
    r".*HipRoll": 0.2,
    r".*HipYaw": 0.2,
    r".*KneePitch": 0.6,
    r".*AnklePitch": 0.35,
    r".*AnkleRoll": 0.15,
    r"torsoYaw": 0.3,
    r"torsoRoll": 0.08,
    r"torsoPitch": 0.2,
    r".*ShoulderPitch": 0.5,
    r".*ShoulderRoll": 0.2,
    r".*ShoulderYaw": 0.15,
    r".*ElbowPitch": 0.35,
    r".*ForearmYaw": 0.3,
    r".*Wrist.*": 0.3,
    r".*NeckPitch": 0.2,
    r"neckYaw": 0.2,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = (TORSO,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (TORSO,)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # --- Gait clock (bipedal): the key driver that makes Valkyrie start walking.
  # Without a phase clock the policy sits in the "stand and balance" local
  # optimum (air_time reward alone is too weak for a heavy humanoid). A shared
  # phase in [0,1) advances while the command is non-zero (frozen when
  # standing); per-leg phase = clock + offset. Left/right feet run anti-phase
  # (0.0, 0.5). The per-leg sin/cos is added to the observation so the policy
  # can time its steps, and a contact-vs-schedule reward pays for stepping on
  # beat. Order (left, right) matches site_names / feet_ground_contact.
  twist_cmd.gait_freq = float(os.environ.get("GAIT_FREQ", "1.4"))
  twist_cmd.gait_offsets = (0.0, 0.5)  # left, right (anti-phase)

  # Per-leg phase (sin/cos x2 legs = 4 dims) in actor + critic obs.
  for group in ("actor", "critic"):
    cfg.observations[group].terms["gait_clock"] = ObservationTermCfg(
      func=mdp.gait_phase_sincos,
      params={"command_name": "twist"},
    )

  # Reward feet whose contact matches the schedule (stance while phase<0.5,
  # swing while phase>=0.5). This is what breaks the standing local optimum.
  cfg.rewards["gait_phase_contact"] = RewardTermCfg(
    func=mdp.gait_phase_contact,
    weight=float(os.environ.get("GAIT_PHASE_W", "2.0")),
    params={"sensor_name": feet_ground_cfg.name},
  )
  # Continuous swing-lift: pay foot height during the scheduled swing window so
  # a foot that never leaves the ground earns nothing (humanoid clearance ~10cm).
  cfg.rewards["gait_swing_lift"] = RewardTermCfg(
    func=mdp.gait_phase_swing_height,
    weight=float(os.environ.get("GAIT_LIFT_W", "0.5")),
    params={
      "target_height": float(os.environ.get("GAIT_LIFT_H", "0.10")),
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )

  # L/R mirror: penalize left/right asymmetry of the foot swing (one-sided
  # high-stepping / roll wobble / a dragging leg). Tracks per-foot swing-height
  # and swing-speed EMAs and costs the |left-right| gap; anti-phase stepping
  # itself is untouched. Fades out for lateral/yaw commands (legit asymmetry).
  cfg.rewards["gait_mirror"] = RewardTermCfg(
    func=mdp.gait_lr_mirror_cost,
    weight=float(os.environ.get("GAIT_MIRROR_W", "-0.5")),
    params={
      "sensor_name": feet_ground_cfg.name,
      "pairs": ((0, 1),),  # (left_foot, right_foot)
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )

  # Arm swing: reward natural counter-swing — each shoulder pitch tracks the
  # negative of the same-side hip pitch, so arms swing anti-phase with the legs
  # (and anti-phase with each other). Gated to moving commands. Flip
  # ARM_SWING_GAIN's sign if arms end up co-swinging with the legs.
  cfg.rewards["arm_swing"] = RewardTermCfg(
    func=mdp.arm_swing,
    weight=float(os.environ.get("ARM_SWING_W", "0.4")),
    params={
      "gain": float(os.environ.get("ARM_SWING_GAIN", "0.6")),
      "std": float(os.environ.get("ARM_SWING_STD", "0.5")),
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          "leftShoulderPitch",
          "rightShoulderPitch",
          "leftHipPitch",
          "rightHipPitch",
        ),
        preserve_order=True,
      ),
    },
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def valkyrie_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create NASA Valkyrie flat terrain velocity configuration."""
  cfg = valkyrie_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.5)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
