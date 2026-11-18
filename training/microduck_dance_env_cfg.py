"""MicroDuck dance task: track one retargeted clip as a one-shot skill.

Built on Pollen's ground-pick task (same 61-D observation contract, same
actuator model and domain randomisation), so the exported ONNX drops into the
same one-shot slot: the command is [cos(2*pi*phase), sin(2*pi*phase), 0] and
the episode starts standing at phase 0.

The reference comes from Duck Retarget ("Export motion"): an .npz at 50 Hz with
qpos in MuJoCo layout. Its path is read from DUCK_DANCE_MOTION.

One period = blend from HOME into the clip, the clip, blend back to HOME, rest.
"""
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from mjlab.managers import CurriculumTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
    MicroduckGroundPickRlCfg,
    make_microduck_ground_pick_env_cfg,
)

MOTION_PATH = Path(os.environ.get(
    "DUCK_DANCE_MOTION", Path(__file__).resolve().parents[5] / "training" / "reference.npz"))

BLEND_IN_S = 0.6
BLEND_OUT_S = 0.6
REST_S = 0.4
STAND_HEIGHT = 0.125
_ROBOT = SceneEntityCfg("robot")


class Reference:
    """The clip resampled over one command period, as torch tables on the sim device."""

    def __init__(self, path, home, joint_names, device):
        d = np.load(path, allow_pickle=False)
        fps = float(d["fps"])
        names = [str(n) for n in d["joint_names"]]
        order = [names.index(n) for n in joint_names]          # by name, not by position
        q = d["joint_pos"][:, order].astype(np.float64)
        pos = d["root_pos"].astype(np.float64)
        rot = R.from_quat(d["root_quat_wxyz"][:, [1, 2, 3, 0]])

        dt = 1.0 / fps
        vel_w = np.gradient(pos, dt, axis=0)
        vel_b = rot.inv().apply(vel_w)
        rel = (rot[:-1].inv() * rot[1:]).as_rotvec() / dt        # body-frame angular velocity
        ang_b = np.vstack([rel, rel[-1:]])
        grav_b = rot.inv().apply(np.array([0.0, 0.0, -1.0]))

        n_in, n_out, n_rest = (int(round(s * fps)) for s in (BLEND_IN_S, BLEND_OUT_S, REST_S))
        home = np.asarray(home, dtype=np.float64)
        down = np.array([0.0, 0.0, -1.0])

        def ramp(a, b, n):
            s = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
            s = s * s * (3 - 2 * s)
            return a[None] * (1 - s) + b[None] * s

        def stitch(clip, rest_value):
            rest_value = np.asarray(rest_value, dtype=np.float64)
            return np.vstack([ramp(rest_value, clip[0], n_in), clip, ramp(clip[-1], rest_value, n_out),
                              np.repeat(rest_value[None], n_rest, axis=0)])

        t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
        self.joint_pos = t(stitch(q, home))
        self.height = t(stitch(pos[:, 2:3], [STAND_HEIGHT]))[:, 0]
        self.gravity = t(stitch(grav_b, down))
        self.lin_vel = t(stitch(vel_b[:, :2], [0.0, 0.0]))
        self.yaw_rate = t(stitch(ang_b[:, 2:3], [0.0]))[:, 0]
        self.frames = self.joint_pos.shape[0]

    def index(self, env, command_name):
        cmd = env.command_manager.get_command(command_name)
        phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
        return torch.clamp((phase * self.frames).long(), max=self.frames - 1)


def period_seconds(path=MOTION_PATH):
    d = np.load(path, allow_pickle=False)
    return len(d["joint_pos"]) / float(d["fps"]) + BLEND_IN_S + BLEND_OUT_S + REST_S


def _ref(env):
    ref = getattr(env, "_dance_ref", None)
    if ref is None:
        asset = env.scene["robot"]
        ref = Reference(MOTION_PATH, asset.data.default_joint_pos[0].cpu().numpy(),
                        list(asset.joint_names), env.device)
        env._dance_ref = ref
    return ref


def dance_joint_track(env, command_name="twist", std=0.3):
    ref = _ref(env)
    err = env.scene["robot"].data.joint_pos - ref.joint_pos[ref.index(env, command_name)]
    return torch.nan_to_num(torch.exp(-(err / std) ** 2).mean(dim=-1))


def dance_joint_track_l1(env, command_name="twist"):
    ref = _ref(env)
    err = env.scene["robot"].data.joint_pos - ref.joint_pos[ref.index(env, command_name)]
    return torch.nan_to_num(-err.abs().mean(dim=-1), nan=-1.0)


def dance_gravity_track(env, command_name="twist", std=0.25):
    ref = _ref(env)
    err = env.scene["robot"].data.projected_gravity_b - ref.gravity[ref.index(env, command_name)]
    return torch.nan_to_num(torch.exp(-err.square().sum(dim=-1) / std ** 2))


def dance_height_track(env, command_name="twist", std=0.03):
    ref = _ref(env)
    err = env.scene["robot"].data.root_link_pos_w[:, 2] - ref.height[ref.index(env, command_name)]
    return torch.nan_to_num(torch.exp(-(err / std) ** 2))


def dance_lin_vel_track(env, command_name="twist", std=0.12):
    ref = _ref(env)
    err = env.scene["robot"].data.root_link_lin_vel_b[:, :2] - ref.lin_vel[ref.index(env, command_name)]
    return torch.nan_to_num(torch.exp(-err.square().sum(dim=-1) / std ** 2))


def dance_yaw_rate_track(env, command_name="twist", std=1.5):
    ref = _ref(env)
    err = env.scene["robot"].data.root_link_ang_vel_b[:, 2] - ref.yaw_rate[ref.index(env, command_name)]
    return torch.nan_to_num(torch.exp(-(err / std) ** 2))


def trunk_too_low(env, min_height=0.045):
    return env.scene["robot"].data.root_link_pos_w[:, 2] < min_height


def make_microduck_dance_env_cfg(play: bool = False):
    cfg = make_microduck_ground_pick_env_cfg(play=play, rough=False)
    period = period_seconds()

    # Ground-pick specifics out: mouth targets, the payload in the mouth, feet glued to the floor.
    for name in ("mouth_ground_proximity", "mouth_perpendicular_to_ground", "ground_pick_return_pose_legs",
                 "ground_pick_return_pose_neck", "return_upright", "neck_vel_descent", "mouth_payload_force",
                 "feet_grounded", "feet_flat", "upright"):
        cfg.rewards.pop(name, None)
    cfg.events.pop("sample_mouth_payload", None)

    cfg.rewards["dance_joint_track"] = RewardTermCfg(func=dance_joint_track, weight=6.0, params={"std": 0.3})
    cfg.rewards["dance_joint_track_l1"] = RewardTermCfg(func=dance_joint_track_l1, weight=2.0)
    cfg.rewards["dance_gravity_track"] = RewardTermCfg(func=dance_gravity_track, weight=2.0, params={"std": 0.25})
    cfg.rewards["dance_height_track"] = RewardTermCfg(func=dance_height_track, weight=1.0, params={"std": 0.03})
    cfg.rewards["dance_lin_vel_track"] = RewardTermCfg(func=dance_lin_vel_track, weight=1.5, params={"std": 0.12})
    cfg.rewards["dance_yaw_rate_track"] = RewardTermCfg(func=dance_yaw_rate_track, weight=1.0, params={"std": 1.5})

    # A dance is faster than a careful reach: lighter smoothing than ground-pick, head may hit nothing.
    cfg.rewards["action_rate_l2"].weight = -0.3
    cfg.rewards["neck_action_rate_l2"].weight = -0.3
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={"reward_name": "action_rate_l2",
                "weight_stages": [{"step": 0, "weight": -0.3}, {"step": 1500 * 24, "weight": -0.6}]},
    )

    # One episode is one performance, starting from standing like the runtime's one-shot slot.
    twist = cfg.commands["twist"]
    twist.period = period
    twist.randomize_phase = False
    cfg.episode_length_s = period
    cfg.terminations["trunk_too_low"] = TerminationTermCfg(func=trunk_too_low, time_out=False)
    return cfg


def make_rl_cfg():
    from copy import deepcopy
    rl = deepcopy(MicroduckGroundPickRlCfg)
    rl.experiment_name = "dance"
    rl.run_name = "dance"
    rl.max_iterations = 6000
    return rl


MicroduckDanceRlCfg = make_rl_cfg()
