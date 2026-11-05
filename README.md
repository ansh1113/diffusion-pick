# diffusion-pick

Imitation learning from pixels for **"put the red cube in the bowl"** with a MuJoCo Franka Panda
(MuJoCo Menagerie). A scripted IK expert produces 200 demonstrations; a **CNN Diffusion Policy**
(ResNet18 encoders, action chunking, DDIM sampling) and an **ACT** baseline are trained from
**RGB (wrist + front) + proprioception only** and evaluated on 100 randomized rollouts plus
robustness tests (distractors, lighting, camera shift). The policies are exported to ONNX.

Everything here — data, training, evaluation, GIFs, ONNX — was produced on a **4-core CPU-only VM**
(software OpenGL via EGL/llvmpipe) with the reduced 64x64 configs. A full-resolution config for a
single A10 GPU is included.

<!-- RESULTS -->

## Task & environment (`dpick/env.py`)

| | |
|---|---|
| Robot | Franka Emika Panda from MuJoCo Menagerie (`assets/franka_emika_panda`, Apache-2.0), gravity-compensated position servos |
| Scene | Table, 3 cubes (red / green / blue, 4 cm) with random position + yaw, a bowl (static, random position) |
| Randomization | cube & bowl positions over a 26 x 44 cm workspace (min. separations enforced), cube yaw in [-45°, 45°] (= all orientations, cube is 90°-symmetric), initial arm joints ~ home + N(0, 0.06 rad) |
| Observations | `front_rgb` (fixed third-person cam), `wrist_rgb` (hand-mounted cam), `proprio` (12-D: TCP xyz, sin/cos yaw, finger width, joints 1-6) |
| Action (10 Hz) | absolute TCP target `[x, y, z, yaw, grip]`; a 250 Hz damped-least-squares IK loop (with null-space posture regulation) converts it to joint targets |
| Success | red cube center inside the bowl (radius 6.5 cm from center), resting on the bowl floor, and released by the gripper |
| Rendering | `MUJOCO_GL=egl` (default, set by the package) or `osmesa`. Panda visual meshes are decimated 134k -> 17k triangles (`scripts/make_lite_meshes.py`, `PANDA_VISUAL=lite`, default) which makes llvmpipe rendering ~4.5x faster; collision geometry is untouched. `PANDA_VISUAL=full` uses the original meshes (re-generate data if you switch). |

Robustness conditions (same seeds -> identical cube/bowl layouts across conditions; perturbations use a separate RNG):

* **distractors** – 3 extra objects (yellow cube, purple cylinder, orange bar) placed among the cubes
* **lighting** – key-light intensity x[0.35, 1.7], color tint ±25%, random light direction/position, headlight/ambient x[0.3, 1.5]
* **camera_shift** – front camera translated by up to ±3 cm per axis and rotated by up to ±3° (wrist camera unchanged)

None of these perturbations appear in the training data (training uses random-shift + color-jitter augmentation only).

## Expert & data (`dpick/expert.py`, `scripts/generate_demos.py`)

A waypoint state machine on absolute TCP targets: approach above the red cube (yaw-aligned to the
cube) → descend → close → lift → transfer (with a random lateral detour) → lower → release → retreat.
Per-episode randomization: speed 0.18–0.32 m/s, approach / lift / release heights, grasp height and
xy offsets, placement offset inside the bowl, gripper close / open dwell times. DART-style noise
(σ = 4 mm) is added to the *executed* action during approach and transfer while the clean action is
recorded, so the data contains small recoveries. Only successful episodes are kept.

`data/demos_64.zarr`: **200 demos, 10,035 steps** (38–68 steps per episode, mean 50), 64x64 images,
expert success rate under DART noise 200/233 (86%).

```
data/front_rgb (N,64,64,3) uint8   data/wrist_rgb (N,64,64,3) uint8
data/proprio   (N,12) float32      data/action    (N,5) float32
meta/episode_ends (200,) int64     attrs: seeds, img_size, noise, ...
```

## Policies

<!-- ARCH -->

## Reproduce

```bash
pip install -r requirements.txt
# Ubuntu: apt-get install -y libegl1 libgl1-mesa-dri libosmesa6   (headless rendering)
```

```bash
python scripts/generate_demos.py --n 200 --img-size 64 --out data/demos_64.zarr
```

```bash
python scripts/train.py --config configs/dp_reduced.yaml
```

```bash
python scripts/train.py --config configs/act_reduced.yaml
```

```bash
python scripts/evaluate.py --ckpt runs/dp_reduced/best.pt --n 100 --out results/dp_reduced
```

```bash
python scripts/export_onnx.py --ckpt runs/dp_reduced/best.pt --out exports/dp_reduced.onnx
```

```bash
python scripts/make_report.py results/dp_reduced results/act_reduced
```

### Full-resolution run (A10)

```bash
python scripts/generate_demos.py --n 200 --img-size 128 --out data/demos_128.zarr
```

```bash
python scripts/train.py --config configs/dp_full.yaml
```

```bash
python scripts/train.py --config configs/act_full.yaml
```

`get_device()` picks `cuda` > `mps` > `cpu` automatically (override with `--device`). On CUDA,
training uses bf16 autocast. Rendering on the GPU box also works with `MUJOCO_GL=egl` (NVIDIA EGL).
Evaluation runs policy inference on CPU workers (one per core), so it is the same command.

## Repository layout

```
assets/franka_emika_panda/   Menagerie Panda (+ assets_lite/ decimated visual meshes)
dpick/env.py                 MuJoCo env, IK controller, randomization, robustness perturbations
dpick/expert.py              scripted expert
dpick/dataset.py             zarr I/O, normalizers, sequence dataset
dpick/models/common.py       device selection, GroupNorm ResNet18, spatial softmax, augmentation, EMA
dpick/models/diffusion.py    1D conditional UNet, cosine DDPM schedule, DDIM sampler, DiffusionPolicy
dpick/models/act.py          ACT (CVAE + transformer encoder/decoder)
dpick/rollout.py             closed-loop execution (receding horizon / temporal ensembling), GIF frames
configs/                     dp_reduced / act_reduced (CPU), dp_full / act_full (A10)
scripts/                     generate_demos, train, evaluate, export_onnx, make_report, make_lite_meshes
data/demos_64.zarr           the 200 demonstrations used here
runs/*/best.pt               selected checkpoints (+ train_log.jsonl)
results/                     results.json per policy, results.md, success_rates.png, GIFs
exports/                     ONNX models + metadata json
```
