"""Build a clean, mjlab-style Valkyrie MJCF from the preprocessed URDF.

- floating base freejoint on pelvis
- visual mesh geoms -> group 2, no contact
- feet-only collision (foot collision meshes enabled, all other body collision disabled)
- foot sites (left_foot/right_foot) + pelvis imu site
- meshdir = assets  (self-contained under xmls/)

Run from mjlab root:  uv run python src/mjlab/asset_zoo/robots/nasa_valkyrie/build_mjcf.py
"""
import os
import pathlib
import mujoco

HERE = pathlib.Path(__file__).parent
# Point this at a local checkout of NASA's public `val_description`
# (https://github.com/NASA-JSC-Robotics/val_description, NOSA v1.3).
# Override with the VAL_DESCRIPTION env var if it lives elsewhere.
VAL_DESCRIPTION = pathlib.Path(
    os.environ.get("VAL_DESCRIPTION", str(HERE / "val_description"))
)
URDF = VAL_DESCRIPTION / "model/robots/valkyrie_mj.urdf"
OUT = HERE / "xmls" / "valkyrie.xml"

FOOT_BODIES = {"leftFoot", "rightFoot"}

spec = mujoco.MjSpec.from_file(str(URDF))
# compile against the real asset dir; the written xml uses relative "assets"
ASSETS_ABS = str((HERE / "xmls" / "assets").resolve())
spec.meshdir = ASSETS_ABS
spec.modelname = "nasa_valkyrie"

# floating base
spec.body("pelvis").add_freejoint(name="floating_base_joint")

# classify + configure geoms
for b in spec.bodies:
    bname = b.name
    vi = ci = 0
    for g in b.geoms:
        is_visual = (g.contype == 0 and g.conaffinity == 0)
        if is_visual:
            g.group = 2
            g.density = 0.0
            g.contype = 0
            g.conaffinity = 0
            if not g.name:
                g.name = f"{bname}_visual_{vi}"; vi += 1
        else:
            ci += 1
            # collision mesh
            g.group = 3
            if bname in FOOT_BODIES:
                side = "left" if bname.startswith("left") else "right"
                g.name = f"{side}_foot_collision"
                g.contype = 1
                g.conaffinity = 1
                g.condim = 3
                g.priority = 1
                g.friction = [0.8, 0.02, 0.001]
            else:
                # feet-only: disable all other body collisions
                g.name = f"{bname}_collision_{ci}"
                g.contype = 0
                g.conaffinity = 0

# foot sites (used by locomotion env) + pelvis imu site
for bname in ("leftFoot", "rightFoot"):
    side = "left" if bname.startswith("left") else "right"
    s = spec.body(bname).add_site()
    s.name = f"{side}_foot"
    s.pos = [0.045, 0.0, -0.06]
    s.size = [0.02, 0.02, 0.02]
imu = spec.body("pelvis").add_site()
imu.name = "imu_in_pelvis"
imu.pos = [0.0, 0.0, 0.0]
imu.size = [0.01, 0.01, 0.01]

# IMU sensors on the pelvis site (matches mjlab velocity-env observation terms)
S = mujoco.mjtSensor
OBJ = mujoco.mjtObj
def add_sensor(name, stype, objtype, objname, reftype=None, refname=None):
    s = spec.add_sensor()
    s.name = name
    s.type = stype
    s.objtype = objtype
    s.objname = objname
    if reftype is not None:
        s.reftype = reftype
        s.refname = refname
    return s
add_sensor("imu_ang_vel", S.mjSENS_GYRO, OBJ.mjOBJ_SITE, "imu_in_pelvis")
add_sensor("imu_lin_vel", S.mjSENS_VELOCIMETER, OBJ.mjOBJ_SITE, "imu_in_pelvis")
add_sensor("imu_lin_acc", S.mjSENS_ACCELEROMETER, OBJ.mjOBJ_SITE, "imu_in_pelvis")
add_sensor("imu_upvector", S.mjSENS_FRAMEZAXIS, OBJ.mjOBJ_BODY, "world",
           OBJ.mjOBJ_SITE, "imu_in_pelvis")
add_sensor("root_angmom", S.mjSENS_SUBTREEANGMOM, OBJ.mjOBJ_BODY, "pelvis")

model = spec.compile()
print("compiled OK: nq", model.nq, "nv", model.nv, "nbody", model.nbody,
      "ngeom", model.ngeom, "nsite", model.nsite)
# contact-enabled geoms
nc = sum(1 for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i])
print("contact-enabled geoms:", nc,
      "->", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
             for i in range(model.ngeom)
             if model.geom_contype[i] or model.geom_conaffinity[i]])

# to_xml re-resolves meshes, so keep the absolute meshdir, then rewrite the
# emitted absolute path to the portable relative "assets".
xml = spec.to_xml()
xml = xml.replace(ASSETS_ABS, "assets")
OUT.write_text(xml)
print("wrote", OUT, "(meshdir -> assets)")
