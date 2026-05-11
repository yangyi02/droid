# DROID Extrinsic Calibration Pipeline

Multi-stage camera extrinsics calibration pipeline for the [DROID dataset](https://droid-dataset.github.io/),
combining VGGT visual anchoring, differentiable depth-based robot alignment, and global joint optimization
to produce high-quality 4×4 camera extrinsic matrices.

## Quickstart

```bash
# 1. Clone repo with all dependencies in one shot
git clone --recurse-submodules https://github.com/yangyi02/droid.git
cd droid

# 2. Install dependencies and download model weights
bash setup.sh

# 3. Run Stage 1 (stereo depth + gripper depth refinement)
bash run_parallel.sh

# 4. Run Stage 2 (camera extrinsics calibration)
bash run_parallel.sh --stage 2
```

> If you already cloned **without** `--recurse-submodules`, run `bash setup.sh` anyway —
> it will call `git submodule update --init --recursive` automatically.

## Pipeline Overview

### Stage 1 — `process_droid_stage1.py`
Decodes raw ZED SVO stereo video, extracts robot kinematics, and infers metric depth.

| Step | Description |
|------|-------------|
| SVO decode | Extract left/right video + calibration from ZED SVO files |
| Kinematics | Parse robot joint positions, EE poses, hand-eye matrix from H5 |
| Stereo depth | S2M2 stereo matching → metric depth |
| SAM mask | Extract static gripper mask from closed-gripper frames |
| Depth distillation | Temporal median filtering within gripper mask |
| Depth injection | Inject clean gripper depth into raw stereo stream |

**Output** (`~/droid_data/output/mv-tap/droid/stage1/<episode_id>/`):
```
robot.npz                      # joint_positions, T_ee_base_all, T_cam_ee_init, ...
<cam_serial>/
  calibration.npz              # K matrix, baseline
  video_left.mp4               # decoded left video
  raw_depth.npz                # refined depth (uint16 mm)
  original_raw_depth.npz       # pre-injection backup (wrist cam only)
  gripper_mask.npz             # SAM consensus mask (wrist cam only)
  gripper_depth.npz            # distilled gripper surface depth (wrist cam only)
```

### Stage 2 — `process_droid_stage2.py`
Multi-stage camera extrinsics calibration using differentiable rendering and point cloud alignment.

| Stage | Description |
|-------|-------------|
| Stage 0 | Read pre-calibrated dataset extrinsics (if available) |
| Stage 1 | VGGT visual anchoring from first frame |
| Stage 2a | Dual-base competition: external camera ↔ robot arm alignment |
| Stage 2b | Wrist camera ↔ gripper body alignment |
| Stage 3 | Global joint optimization — Chamfer + Robot + Wrist (lr=0.001) |
| Stage 4 | Fine-tuning refinement (lr=0.0001, lower robot weight) |

**Output** (`~/droid_data/output/mv-tap/droid/stage2/<episode_id>/`):
```
extrinsics.npz
  <cam_serial>_base_extrinsic  # (4, 4) static extrinsic matrix
  <cam_serial>_extrinsics      # (N, 4, 4) per-frame trajectory
  wrist_serial                 # wrist camera serial string
```

## Directory Structure

```
droid/
├── third_party/               # Dependencies (populated by git submodules + setup.sh)
│   ├── s2m2/                  # Stereo matching model
│   │   └── weights/           # Downloaded by setup.sh
│   ├── vggt/                  # Visual camera pose estimation
│   ├── co-tracker/            # Dense point tracking
│   │   └── weights/           # Downloaded by setup.sh
│   ├── PointWorld/            # Franka + Robotiq URDF assets (branch: data)
│   └── sam_weights/           # SAM ViT-H weights (downloaded by setup.sh)
├── process_droid_stage1.py    # Stage 1: depth extraction pipeline
├── process_droid_stage2.py    # Stage 2: extrinsics calibration pipeline
├── run_parallel.sh            # Multi-GPU parallel runner
├── setup.sh                   # One-shot dependency setup
├── pipeline.ipynb             # Reference notebook
└── .gitmodules                # Submodule declarations
```

## Dependencies

### Python Packages
```bash
pip install pybullet pybullet-data opencv-python scipy tqdm h5py
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Git Submodules (auto-cloned with `--recurse-submodules`)

| Submodule | Repo | Notes |
|-----------|------|-------|
| `third_party/s2m2` | [junhong-3dv/s2m2](https://github.com/junhong-3dv/s2m2) | Stereo depth |
| `third_party/vggt` | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | Camera pose |
| `third_party/co-tracker` | [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker) | Point tracking |
| `third_party/PointWorld` | [NVlabs/PointWorld @ data](https://github.com/NVlabs/PointWorld/tree/data) | Robot URDF assets |

### Model Weights (downloaded by `setup.sh`)

| Model | Source | Path |
|-------|--------|------|
| S2M2 XL | HuggingFace `minimok/s2m2` | `third_party/s2m2/weights/` |
| CoTracker3 | HuggingFace `facebook/cotracker3` | `third_party/co-tracker/weights/` |
| SAM ViT-H | `dl.fbaipublicfiles.com` | `third_party/sam_weights/` |
| VGGT-1B | HuggingFace `facebook/VGGT-1B` | Auto-downloaded at first run |

## Running Options

```bash
# Stage 1: all episodes (auto-discovers from metadata)
bash run_parallel.sh

# Stage 1: custom episode list
bash run_parallel.sh --file my_episodes.txt

# Stage 2: process episodes completed by stage 1
bash run_parallel.sh --stage 2

# Stage 2: custom episode list
bash run_parallel.sh --stage 2 --file episodes_stage1.txt
```

Parallel jobs are automatically scaled to the number of available GPUs detected by `nvidia-smi`.
Each GPU gets one job at a time: `CUDA_VISIBLE_DEVICES=<gpu_id>`.
