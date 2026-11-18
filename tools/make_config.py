"""Erzeugt Scaler-, Retargeter- und Post-Processing-Konfiguration fuer ein Ziel,
so wie es der Robot Configurator des SOMA Retargeters tut, nur ohne Fenster.

    uv run python ../../tools/make_config.py --check-g1     # Verfahren gegen die G1-Konfig pruefen
    uv run python ../../tools/make_config.py                # Enten-Konfig schreiben
"""
import argparse, json, pathlib, sys
import numpy as np
import warp as wp
import newton

import soma_retargeter.io.bvh as bvh_utils
import soma_retargeter.pipelines.utils as pipeline_utils
import soma_retargeter.utils.newton_utils as newton_utils
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.robotics.robot_registry import register_robots_path, registry
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType

HERE = pathlib.Path(__file__).resolve().parent
ROBOTS = HERE.parent / "robots"

# SOMA-Gelenk -> (Koerper der Ente, Positionsgewicht, Rotationsgewicht)
DUCK_MAP = {
    "Hips":      ("trunk_base",      15.0, 10.0),
    "Head":      ("jaw_soft",         1.0,  3.0),
    "LeftLeg":   ("upper_leg_left",   1.5,  0.15),
    "LeftShin":  ("leg",             10.0,  1.0),
    "LeftFoot":  ("ankle_left",      30.0,  2.0),
    "RightLeg":  ("upper_leg_right",  1.5,  0.15),
    "RightShin": ("leg_2",           10.0,  1.0),
    "RightFoot": ("ankle_right",     30.0,  2.0),
}


def q_inv(q): return wp.quat_inverse(q)


def local_offset(src, dst):
    q = wp.normalize(wp.mul(q_inv(src.q), dst.q))
    p = wp.quat_rotate(q_inv(dst.q), dst.p - src.p)
    return wp.transform(p, q)


def soma_zero_pose(scale, z_rot):
    path = pipeline_utils.get_source_zero_pose_asset_path(pipeline_utils.SourceType.SOMA)
    skel, anim = bvh_utils.load_bvh(path)
    xf = SpaceConverter(FacingDirectionType.MUJOCO).transform(wp.transform_identity())
    if z_rot:
        xf = wp.mul(wp.transform(wp.vec3(0, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.radians(z_rot))), xf)
    inst = SkeletonInstance(skel, [0, 0, 0], xf)
    inst.set_local_transforms(anim.get_local_transforms(0))
    return skel, inst.compute_global_transforms(scale=scale)


def robot_bodies(target, joint_q=None):
    builder = pipeline_utils.create_robot_builder(target)
    model = builder.finalize()
    state = model.state()
    if joint_q is not None:
        model.joint_q.assign(np.asarray(joint_q, dtype=np.float32))
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    names = [newton_utils.get_name_from_label(l) for l in builder.body_label]
    return names, state.body_q.numpy(), model.joint_q.numpy(), builder


def offsets_for(target, mapping, scale, z_rot, joint_q=None):
    names, body_q, jq, _ = robot_bodies(target, joint_q)
    skel, soma = soma_zero_pose(scale, z_rot)
    out = {}
    for joint, body in mapping.items():
        s = wp.transform(*soma[skel.joint_index(joint)])
        r = wp.transform(*body_q[names.index(body)])
        o = local_offset(s, r)
        out[joint] = [[round(float(v), 4) for v in o.p], [round(float(v), 4) for v in o.q]]
    return out, jq


def check_g1():
    entry = registry.get("unitree_g1")
    base = pathlib.Path(entry.manifest_dir)
    ret = json.load(open(base / "configs/soma_to_g1_retargeter_config.json"))
    sc = json.load(open(base / "configs/soma_to_g1_scaler_config.json"))
    mapping = {j: m["t_body"] for j, m in ret["ik_match_table"].items()}
    for z in (0.0, 90.0, -90.0, 180.0):
        got, _ = offsets_for("unitree_g1", mapping, sc["human_scale_ratio"], z, sc["reference_joint_q"])
        err_p = max(np.abs(np.array(got[j][0]) - np.array(sc["human_joint_offsets"][j][0])).max() for j in mapping)
        err_q = max(min(np.abs(np.array(got[j][1]) - np.array(sc["human_joint_offsets"][j][1])).max(),
                        np.abs(np.array(got[j][1]) + np.array(sc["human_joint_offsets"][j][1])).max()) for j in mapping)
        print(f"z_rot {z:6.1f}: max Abweichung Position {err_p:.4f} m, Quaternion {err_q:.4f}")


def write_duck(z_rot):
    target = "microduck"
    names, body_q, jq, builder = robot_bodies(target)
    skel, soma1 = soma_zero_pose(1.0, z_rot)
    hip_z = float(body_q[names.index("trunk_base")][2])
    scale = hip_z / float(soma1[skel.joint_index("Hips")][2])
    mapping = {j: v[0] for j, v in DUCK_MAP.items()}
    offs, jq = offsets_for(target, mapping, scale, z_rot)
    cfg = ROBOTS / "pollen/microduck/configs"
    json.dump({
        "human_type": "soma", "human_root": "Hips", "robot_type": target, "robot_root": "trunk_base",
        "human_scale_ratio": scale, "reference_joint_q": [round(float(v), 8) for v in jq],
        "human_joint_offsets": offs,
    }, open(cfg / "soma_to_microduck_scaler_config.json", "w"), indent=4)
    json.dump({
        "num_initialization_frames": 10, "num_stabilization_frames": 5,
        "human_robot_scaler_config": "configs/soma_to_microduck_scaler_config.json",
        "ik_iterations": 24, "joint_limit_weight": 10.0, "smooth_joint_filter_weight": 0.0,
        "enable_post_processing": True, "enable_contact_processing": True,
        "post_processing": {"robot_config": "configs/microduck_post_processing_config.json"},
        "ik_match_table": {j: {"t_body": b, "r_body": b, "t_weight": tw, "r_weight": rw}
                           for j, (b, tw, rw) in DUCK_MAP.items()},
    }, open(cfg / "soma_to_microduck_retargeter_config.json", "w"), indent=4)
    json.dump({
        "contact_correction": {"transition_frames": 4, "propagation_ratio": 0.4,
                               "rotation_propagation_ratio": 0.15, "enable_flatten_foot_plant": True},
        "limb_stabilizer": {
            "ik_iterations": 20, "joint_limit_weight": 10.0,
            "effectors": {"trunk_base": [30.0, 8.0],
                          "upper_leg_left": [1.5, 0.15], "leg": [1.0, 1.0], "ankle_left": [10.0, 2.0],
                          "upper_leg_right": [1.5, 0.15], "leg_2": [1.0, 1.0], "ankle_right": [10.0, 2.0]},
            "ik_root": 0, "root_correction_ratio": 1.0, "root_smooth_alpha": 0.8,
            "ik_limbs": {"LeftFoot": {"effectors": [1, 2, 3], "hint_reference": 2, "hint_offset": [0.03, 0.0, 0.0]},
                         "RightFoot": {"effectors": [4, 5, 6], "hint_reference": 5, "hint_offset": [0.03, 0.0, 0.0]}},
        },
    }, open(cfg / "microduck_post_processing_config.json", "w"), indent=4)
    print(f"Koerper: {names}")
    print(f"Skalierung {scale:.4f} (Rumpfhoehe Ente {hip_z:.3f} m), {len(jq)} Koordinaten")
    print("SOMA-Gelenke:", skel.joint_names if hasattr(skel, 'joint_names') else '')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-g1", action="store_true")
    ap.add_argument("--z-rot", type=float, default=90.0)
    a = ap.parse_args()
    register_robots_path(str(ROBOTS))
    check_g1() if a.check_g1 else write_duck(a.z_rot)
