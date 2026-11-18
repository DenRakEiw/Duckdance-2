# Duck Retarget

Human motion capture in, a dancing [MicroDuck](https://github.com/pollen-robotics/microduck) out.

A Windows app that runs motion-capture clips through the
[NVIDIA SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter) onto the
MicroDuck and plays the result in 3D. The retargeter ships configurations for
humanoids only; this repo adds the one for an armless duck, a viewer, and a
training task that turns a clip into a policy.

It is the offline companion to [DuckDance](https://github.com/DenRakEiw/DuckDance),
which runs in the browser and steers the duck's walking policy from a video.

## What is in here

| Path | What it does |
| --- | --- |
| `app/main.py` | The app: clip list, GPU retargeting, 3D preview, video and motion export |
| `app/motion.py`, `app/render.py` | Retargeter CSV to MuJoCo `qpos`, offscreen rendering |
| `robots/pollen/microduck/` | The retargeter target: model, meshes, scaler / IK / post-processing configs |
| `tools/make_config.py` | Generates those configs headlessly, the way NVIDIA's Robot Configurator does |
| `training/microduck_dance_env_cfg.py` | mjlab task that tracks one exported clip as a one-shot skill |
| `tools/install_training.py` | Installs that task into Pollen's `microduck_rl` |

## Setup

Needs Windows 10/11, an NVIDIA GPU (driver 545 or newer), Git with Git LFS, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/DenRakEiw/Duckdance-2.git
cd Duckdance-2

# the app's own environment
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# the retargeter, in its own environment (about 700 MB with sample motions)
git clone https://github.com/NVIDIA/soma-retargeter.git vendor/soma-retargeter
cd vendor/soma-retargeter && git lfs pull && uv sync && cd ../..
```

Start it with `DuckRetarget.bat`.

## Using it

1. Tick clips in the list. NVIDIA's ten sample motions are there from the start;
   **Add BVH files** takes your own. They must use the SOMA skeleton, a generic
   BVH will not load correctly.
2. **Retarget to duck** runs the retargeter on the GPU. The first run compiles
   kernels and takes a few minutes; after that about 85 s for one clip.
3. Click a clip to watch it. Drag orbits, the wheel zooms.
4. **Save video** writes an MP4, **Export motion** a 50 Hz reference (`.npz`, `.json`) to `export/`.

## How the duck mapping works

The retargeter matches SOMA joints to robot bodies and solves IK per frame.
The duck gets eight of them: hips to the trunk, thigh, shin and foot per leg,
and the head. Arms, spine and hands stay unmapped, which the retargeter allows.
Motion is scaled by trunk height, about 0.12 of human size.

`tools/make_config.py` computes the per-joint offsets from the SOMA zero pose
and the duck's default pose. Run against the bundled Unitree G1 it reproduces
NVIDIA's shipped rotations exactly and positions to within 4 cm.

The result is kinematic: the duck is posed frame by frame, with no physics and
no balance. In the sample hip-hop clip about 16 % of joint values sit at a limit,
mostly hip roll, because a human stance is wider than short duck legs can follow.

## From clip to policy

To make the duck actually perform a clip, a policy has to learn to track it.
`training/` holds a task for Pollen's [microduck_rl](https://github.com/pollen-robotics/microduck_rl)
(mjlab, PPO). It reuses their ground-pick task, so observations, actuator model
and domain randomisation match their other one-shot skills.

```bash
git clone https://github.com/pollen-robotics/microduck_rl.git vendor/microduck_rl
cd vendor/microduck_rl && uv sync
# on Windows uv resolves the CPU build of PyTorch; swap in the CUDA one
uv pip install --python .venv/Scripts/python.exe "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch
cd ../..
python tools/install_training.py

# export a clip in the app, then point the task at it
set DUCK_DANCE_MOTION=%CD%\export\<clip>.npz
set PYTHONUTF8=1
cd vendor/microduck_rl
uv run --no-sync train Mjlab-Dance-Flat-MicroDuck --env.scene.num-envs 4096 --agent.logger tensorboard
```

`--no-sync` keeps uv from putting the CPU build back. Without `DUCK_DANCE_MOTION`
the task reads `training/reference.npz`.

This part is experimental: the task trains, whether the result dances is not yet shown.

## Licences

Code in this repository: Apache License 2.0.
The MicroDuck meshes are Pollen Robotics' and CC BY-NC-SA 4.0, see
[`robots/pollen/microduck/NOTICE.md`](robots/pollen/microduck/NOTICE.md).
The SOMA Retargeter and its sample motions are NVIDIA's and are not included here.
