"""Liest die CSV des SOMA Retargeters und macht daraus MuJoCo-qpos fuer die MicroDuck."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parent.parent
DUCK_XML = ROOT / "robots/pollen/microduck/desc/microduck.xml"
CSV_FPS = 120.0

SCENE_XML = """
<mujoco model="duck_stage">
  <include file="microduck.xml"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global offwidth="1920" offheight="1080" azimuth="140" elevation="-18"/>
    <quality shadowsize="4096"/>
    <map znear="0.005"/>
  </visual>
  <statistic center="0 0 0.08" extent="0.45"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.02 0.02 0.03" rgb2="0 0 0" width="64" height="64"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.11 0.11 0.11" rgb2="0.075 0.075 0.075"
             mark="edge" markrgb="0.91 0.2 0.43" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="20 20" reflectance="0.15"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.3 1.0" dir="-0.3 0.25 -1" diffuse="0.9 0.9 0.9" castshadow="true"/>
    <light pos="-0.5 0.5 0.6" dir="0.5 -0.5 -0.6" diffuse="0.5 0.12 0.26" castshadow="false"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="grid" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""


def load_csv(path):
    """-> (fps, qpos[T, 7+14], Gelenknamen). qpos im MuJoCo-Layout (Quaternion wxyz)."""
    with open(path, encoding="utf-8") as f:
        names = f.readline().strip().split(",")[7:]
        data = np.loadtxt(f, delimiter=",", ndmin=2)
    T = len(data)
    qpos = np.zeros((T, 7 + len(names)))
    qpos[:, 0:3] = data[:, 1:4] * 0.01
    xyzw = R.from_euler("xyz", data[:, 4:7], degrees=True).as_quat()
    qpos[:, 3] = xyzw[:, 3]
    qpos[:, 4:7] = xyzw[:, 0:3]
    qpos[:, 7:] = np.deg2rad(data[:, 7:])
    return CSV_FPS, qpos, names


def load_model():
    import mujoco
    stage = DUCK_XML.with_name("stage.xml")   # neben dem Modell, damit meshdir stimmt
    if not stage.exists() or stage.read_text(encoding="utf-8") != SCENE_XML:
        stage.write_text(SCENE_XML, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(stage))
    return model, mujoco.MjData(model)


def check_joint_order(model, names):
    import mujoco
    got = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
    if got != list(names):
        raise ValueError(f"Gelenkreihenfolge passt nicht: CSV {names} / Modell {got}")
