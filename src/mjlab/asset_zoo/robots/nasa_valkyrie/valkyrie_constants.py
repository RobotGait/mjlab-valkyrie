"""NASA Valkyrie (R5) constants for mjlab.

Model source: NASA JSC `val_description` (NASA Open Source Agreement v1.3),
converted URDF->MJCF for this repo. Effort / velocity limits below are taken
verbatim from the official URDF joint limits. Stiffness/damping are tuned PD
gains for position control (Valkyrie's exact rotor inertias are not public).
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF.
##

VALKYRIE_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "nasa_valkyrie" / "xmls" / "valkyrie.xml"
)
assert VALKYRIE_XML.exists(), VALKYRIE_XML


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(VALKYRIE_XML))


##
# Actuators. effort/velocity from URDF; PD gains tuned for RL position control.
##

_ARM = 0.15  # nominal reflected inertia for stability (large SEA joints)

ACT_LEG_HIP_KNEE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*HipRoll", ".*HipPitch", ".*KneePitch"),
    stiffness=350.0, damping=10.0, effort_limit=350.0, armature=0.6,
)
ACT_LEG_HIPYAW = BuiltinPositionActuatorCfg(
    target_names_expr=(".*HipYaw",),
    stiffness=250.0, damping=8.0, effort_limit=190.0, armature=0.4,
)
ACT_ANKLE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*AnklePitch", ".*AnkleRoll"),
    stiffness=200.0, damping=6.0, effort_limit=205.0, armature=0.3,
)
ACT_TORSO = BuiltinPositionActuatorCfg(
    target_names_expr=("torsoYaw", "torsoPitch", "torsoRoll"),
    stiffness=300.0, damping=8.0, effort_limit=170.0, armature=0.4,
)
ACT_SHOULDER = BuiltinPositionActuatorCfg(
    target_names_expr=(".*ShoulderPitch", ".*ShoulderRoll", ".*ShoulderYaw"),
    stiffness=150.0, damping=5.0, effort_limit=150.0, armature=0.2,
)
ACT_ELBOW = BuiltinPositionActuatorCfg(
    target_names_expr=(".*ElbowPitch", ".*ForearmYaw"),
    stiffness=100.0, damping=4.0, effort_limit=65.0, armature=0.15,
)
ACT_WRIST = BuiltinPositionActuatorCfg(
    target_names_expr=(".*WristRoll", ".*WristPitch"),
    stiffness=40.0, damping=2.0, effort_limit=14.0, armature=0.05,
)
ACT_NECK = BuiltinPositionActuatorCfg(
    target_names_expr=("lowerNeckPitch", "neckYaw", "upperNeckPitch"),
    stiffness=40.0, damping=2.0, effort_limit=26.0, armature=0.05,
)

VALKYRIE_ACTUATORS = (
    ACT_LEG_HIP_KNEE, ACT_LEG_HIPYAW, ACT_ANKLE, ACT_TORSO,
    ACT_SHOULDER, ACT_ELBOW, ACT_WRIST, ACT_NECK,
)

##
# Keyframe: validated crouch standing pose (feet flat).
##

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.05),
    joint_pos={
        ".*HipPitch": -0.45,
        ".*KneePitch": 0.90,
        ".*AnklePitch": -0.45,
        "leftShoulderRoll": -1.30, "rightShoulderRoll": 1.30,
        "leftShoulderPitch": 0.20, "rightShoulderPitch": 0.20,
        "leftElbowPitch": -1.00, "rightElbowPitch": 1.00,
    },
    joint_vel={".*": 0.0},
)

##
# Collision: feet-only (foot collision meshes; body self-collision disabled).
##

FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(left|right)_foot_collision$",),
    contype=1,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.8,),
)

##
# Final config.
##

VALKYRIE_ARTICULATION = EntityArticulationInfoCfg(
    actuators=VALKYRIE_ACTUATORS,
    soft_joint_pos_limit_factor=0.9,
)


def get_valkyrie_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=VALKYRIE_ARTICULATION,
    )


# action scale per joint (0.25 * effort / stiffness), mirrors mjlab convention.
VALKYRIE_ACTION_SCALE: dict[str, float] = {}
for _a in VALKYRIE_ACTUATORS:
    assert isinstance(_a, BuiltinPositionActuatorCfg)
    for _n in _a.target_names_expr:
        VALKYRIE_ACTION_SCALE[_n] = 0.25 * _a.effort_limit / _a.stiffness


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_valkyrie_robot_cfg())
    print("Valkyrie entity built OK")
    m = robot.spec.compile()
    print("nq", m.nq, "nv", m.nv, "nu", m.nu)
