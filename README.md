# DROID Multi-View 3D Tracking Pipeline

Multi-stage pipeline for the [DROID dataset](https://droid-dataset.github.io/):
stereo depth extraction, camera-robot extrinsics calibration, and dense 3D point tracking.

## Quickstart

```bash
# 1. Clone with all dependencies
git clone --recurse-submodules https://github.com/yangyi02/droid.git
cd droid

# 2. Install dependencies + download model weights
bash setup.sh

# 3. Mount GCS input/output buckets
bash mount_gcs.sh

# 4. Run pipeline (3 stages)
bash run_parallel.sh                    # Stage 1: depth
bash run_parallel.sh --mode extrinsics  # Stage 2: extrinsics
bash run_parallel.sh --mode tracks      # Stage 3: tracks
```

> If you cloned **without** `--recurse-submodules`, run `bash setup.sh` —
> it calls `git submodule update --init --recursive` automatically.

## Pipeline Overview

| Stage | Script | Core Modules | Description |
|-------|--------|--------------|-------------|
| 1. Depth | `compute_depth.py` | `core.depth` | SVO decode → S2M2 stereo depth → SAM gripper mask → depth distillation |
| 2. Extrinsics | `compute_extrinsics.py` | `core.physics` | VGGT visual anchoring → differentiable robot alignment → global joint optimization |
| 3. Tracks | `compute_tracks2.py` | `core.tracking` | CoTracker 2D tracking → 3D lift → cross-view dedup → multi-view fusion |

### Stage 1 — `compute_depth.py`

Decodes raw ZED SVO stereo video, extracts robot kinematics, and infers metric depth.

| Step | Description |
|------|-------------|
| SVO decode | Extract left/right video + calibration from ZED SVO files |
| Kinematics | Parse robot joint positions, EE poses, hand-eye matrix from H5 |
| Stereo depth | S2M2 stereo matching → metric depth |
| SAM mask | Extract static gripper mask from closed-gripper frames |
| Depth distillation | Temporal median filtering within gripper mask |
| Depth injection | Inject clean gripper depth into raw stereo stream |

**Output** (`~/droid_data/output/mv-tap/droid/depth/<episode_id>/`):
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

### Stage 2 — `compute_extrinsics.py`

Multi-stage camera extrinsics calibration using differentiable rendering and point cloud alignment.

| Stage | Description |
|-------|-------------|
| Stage 0 | Read pre-calibrated dataset extrinsics (if available) |
| Stage 1 | VGGT visual anchoring from first frame |
| Stage 2a | External camera ↔ robot arm depth alignment |
| Stage 2b | Wrist camera ↔ gripper body alignment |
| Stage 3 | Global joint optimization (Chamfer + Robot + Wrist) |

**Output** (`~/droid_data/output/mv-tap/droid/extrinsics/<episode_id>/`):
```
<cam_serial>/
  extrinsics.json              # base_extrinsic (4x4), extrinsics (Nx4x4), is_wrist
```

### Stage 3 — `compute_tracks2.py`

Dense multi-view 3D point tracking via CoTracker + URDF kinematics.

| Phase | Description |
|-------|-------------|
| Phase 1 | Per-view 2D tracking (CoTracker3) |
| Phase 2 | Lift to 3D + robot/environment split (PyBullet masking) |
| Phase 3 | Cross-view 3D deduplication |
| Phase 4 | Cross-view completion (re-track unified points) |
| Phase 5 | Median 3D fusion across views |

**Output** (`~/droid_data/output/mv-tap/droid/tracks/<episode_id>/`):
```
tracks_3d.npz                  # final_traj_3d, final_vis_global
<cam_serial>/
  tracks_2d.npz                # per-camera 2D tracks + visibility
```

## Directory Structure

```
droid/
├── compute_depth.py           # Stage 1: SVO → stereo depth + gripper refinement
├── compute_extrinsics.py      # Stage 2: VGGT + camera-robot alignment
├── compute_tracks2.py         # Stage 3: CoTracker + multi-view 3D fusion
├── core/                      # Shared algorithmic modules
│   ├── geometry.py            #   3D math: unproject, project, make_4x4, rodrigues
│   ├── io.py                  #   Data loading: get_accelerator, load_depth/extrinsics
│   ├── depth.py               #   S2M2 stereo, SAM gripper mask, depth distillation
│   ├── physics.py             #   TensorRobotRenderer + PyBulletRenderer
│   └── tracking.py            #   URDFKinematicsTracker (FK propagation + visibility)
├── utils/
│   └── visualization.py       # Notebook visualization helpers (point clouds, videos)
├── pipeline.ipynb             # Interactive Colab notebook (thin orchestration layer)
├── run_parallel.sh            # Multi-GPU parallel runner
├── setup.sh                   # One-shot dependency + weights setup
├── mount_gcs.sh               # GCS bucket mount helper
├── episodes.txt               # Full episode ID list
├── verify_outputs.py          # Output verification script
├── assets/                    # Local assets (Franka + Robotiq URDF)
├── third_party/               # Dependencies (git submodules + downloaded weights)
│   ├── s2m2/                  #   Stereo matching model
│   ├── vggt/                  #   Visual camera pose estimation
│   ├── co-tracker/            #   Dense point tracking
│   └── sam_weights/           #   SAM ViT-H weights
└── .gitmodules                # Submodule declarations
```

## Data Setup

The pipeline reads raw DROID data from a GCS bucket and writes outputs to another.
Use `mount_gcs.sh` to mount both via [gcsfuse](https://cloud.google.com/storage/docs/gcsfuse-cli):

```bash
bash mount_gcs.sh
```

| Mount | GCS Bucket / Prefix | Local Path |
|-------|---------------------|------------|
| Input (DROID raw) | `gs://gresearch/robotics/droid_raw` | `~/droid_data/input/robotics/droid_raw` |
| Output | `gs://dm-tapnet/mv-tap` | `~/droid_data/output/mv-tap` |

> To manually unmount:
> ```bash
> fusermount -u ~/droid_data/input/robotics/droid_raw
> fusermount -u ~/droid_data/output/mv-tap
> ```

## Running Options

```bash
bash run_parallel.sh                          # depth, all episodes
bash run_parallel.sh --mode extrinsics        # extrinsics, all episodes
bash run_parallel.sh --mode tracks            # tracks, all episodes
bash run_parallel.sh --mode depth --limit 32  # depth, first 32 episodes
```

| Flag | Short | Values | Default | Description |
|------|-------|--------|---------|-------------|
| `--mode` | `-m` | `depth`, `extrinsics`, `tracks` | `depth` | Pipeline stage |
| `--limit` | `-l` | integer | all | Max episodes to process |

Jobs auto-scale to the number of GPUs detected by `nvidia-smi`.

## Interactive Notebook

Open [`pipeline.ipynb`](https://colab.research.google.com/github/yangyi02/droid/blob/main/pipeline.ipynb) in Colab for single-episode debugging.

Each stage has two paths:
- **🚧 Compute** — run from scratch
- **☁️ Load** — pull pre-computed results from GCS (skip earlier stages)

## Dependencies

### ZED SDK (required for Stage 1 SVO decoding)

```bash
wget https://download.stereolabs.com/zedsdk/5.2/cu12/ubuntu22 -O ZED_SDK_Linux_Ubuntu22.run
chmod +x ZED_SDK_Linux_Ubuntu22.run
./ZED_SDK_Linux_Ubuntu22.run silent runtime_only skip_tools
find /usr/local/zed/ -name "pyzed*.whl" -exec pip install {} \;
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
