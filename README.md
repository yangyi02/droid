# DROID Multi-View 3D Tracking Pipeline

Multi-stage pipeline for the [DROID dataset](https://droid-dataset.github.io/):
stereo depth extraction, camera-robot extrinsics calibration, and dense 3D point tracking.

## Quickstart

```bash
# 1. Clone with all dependencies
git clone --recurse-submodules https://github.com/yangyi02/droid.git
cd droid

# 2. Create the virtualenv -- setup.sh installs into whichever python is
#    active, so without this it goes into the system interpreter
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies + download model weights
bash setup.sh

# 4. Mount GCS input/output buckets
bash mount_gcs.sh

# 5. Run pipeline (3 stages)
bash run_parallel.sh --mode depth        # Stage 1: depth
bash run_parallel.sh --mode extrinsics  # Stage 2: extrinsics
bash run_parallel.sh --mode tracks      # Stage 3: tracks
```

> If you cloned **without** `--recurse-submodules`, run `bash setup.sh` —
> it calls `git submodule update --init --recursive` automatically.

## Pipeline Overview

| Stage | Script | Core Modules | Description |
|-------|--------|--------------|-------------|
| 1. Depth | `compute_depth.py` | `core.depth` | SVO decode → S2M2 stereo depth → SAM gripper mask → depth distillation |
| 2. Extrinsics | `compute_extrinsics.py` | `core.physics` | Dataset extrinsics → rendered robot alignment → global joint optimization |
| 3. Tracks | `compute_tracks.py` | `core.tracking` | Static background depth consensus + URDF FK robot tracks (model-free) |

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

**Output** (`data/output/droid/depth/<episode_id>/`):
```
robot.npz                      # joint_positions, T_ee_base_all, T_cam_ee_init, ...
<cam_serial>/
  calibration.npz              # K matrix, baseline
  video_left.mp4               # decoded left video (also _right, _left_raw, _right_raw)
  raw_depth.npz                # refined depth (uint16 mm)
  original_raw_depth.npz       # pre-injection backup (wrist cam only)
  gripper_mask.npz             # SAM consensus mask (wrist cam only)
  gripper_depth.npz            # distilled gripper surface depth (wrist cam only)
```

### Stage 2 — `compute_extrinsics.py`

Multi-stage camera extrinsics calibration: the robot is rasterised from each camera's current pose estimate, and the resulting point cloud is aligned against the observed depth.

| Step | Description |
|------|-------------|
| `init_camera_states` | Read pre-calibrated dataset extrinsics from metadata |
| `per_camera_alignment` | Per-camera independent depth & robot alignment |
| `global_joint_alignment` | Global joint optimization (Chamfer + Robot + Wrist) |

**Output** (`data/output/droid/extrinsics/<episode_id>/`):
```
<cam_serial>/
  extrinsics.json              # base_extrinsic (4x4), extrinsics (Nx4x4), is_wrist
```

### Stage 3 — `compute_tracks.py`

Dense multi-view 3D point tracking via static background prior + URDF forward kinematics (model-free).

| Step | Description |
|------|-------------|
| `find_static_candidates` | Multi-view depth consensus to sample static background points |
| `project_static_tracks` | Project static background 3D points into per-view 2D trajectories |
| `compute_robot_tracks` | Sample robot CAD surface points + URDF FK forward propagation |
| `merge_tracks` | Merge static background & robot tracks with global visibility masks |

**Output** (`data/output/droid/tracks/<episode_id>/`):
```
tracks_3d.npz                  # traj_3d, vis_global
track_metadata.npz             # n_static, n_robot
<cam_serial>/
  tracks_2d.npz                # per-camera 2D tracks (traj_2d) + visibility (vis_2d)
```

## Directory Structure

```
droid/
├── compute_depth.py           # Stage 1: SVO → stereo depth + gripper refinement
├── compute_extrinsics.py      # Stage 2: Dataset init + camera-robot alignment
├── compute_tracks.py          # Stage 3: Static prior + URDF FK dense 3D tracking
├── compute_metrics.py         # Batch quality metrics evaluation (GCP)
├── run_parallel.sh            # Multi-GPU parallel runner for the stages above
├── setup.sh                   # One-shot dependency + weights setup
├── mount_gcs.sh               # GCS bucket mount helper
├── core/                      # Shared algorithmic modules
│   ├── geometry.py            #   3D math: unproject, project, make_4x4, rodrigues
│   ├── io.py                  #   Data loading: get_accelerator, load_depth/extrinsics
│   ├── depth.py               #   S2M2 stereo, SAM gripper mask, depth distillation
│   ├── physics.py             #   PyBulletRenderer + robot point clouds and depth losses
│   ├── runner.py              #   Episode sharding + resume-aware batch loop
│   ├── tracking.py            #   URDFKinematicsTracker (FK propagation + visibility)
│   └── visualization.py       #   Visualization helpers (point clouds, tracking videos, 4D orbit)
├── tapvidmv/                  # Everything specific to the TAPVid-MV release
│   ├── export_tapvidmv.py     #   Pipeline outputs → TAPVid-MV release format
│   ├── run_export.sh          #   Parallel runner for the export
│   ├── select_episodes.py     #   Scene-stratified candidate pool from metrics CSV
│   ├── pick_episodes.ipynb    #   Visual picker: candidate pool → release set
│   ├── episodes_eval50.txt    #   The 50 selected evaluation episodes
│   ├── download_episodes.sh   #   Fetch the released episodes
│   ├── verify_downloads.sh    #   Size-check downloads, delete corrupt files
│   ├── visualize_groundtruth_colab.ipynb   # Self-contained ground-truth viewer
│   └── visualize_tracks_groundtruth.ipynb  # 3D/2D track inspection, all episodes
├── notebooks/                 # Interactive notebooks (run from anywhere in the checkout)
│   ├── pipeline.ipynb         #   Whole pipeline, one episode at a time (flag-based flow)
│   ├── filter_points.ipynb    #   Dropping background points carried away by the gripper
│   ├── pybullet_numpy_benchmark.ipynb  # Why PyBullet must be built with NumPy support
│   ├── pybullet_egl_mask_benchmark.ipynb  # Why the GPU rasteriser is off, and what it would take
│   └── pybullet_gpu_pipeline_validation.ipynb  # gpu=True on a real episode: renders, point clouds, extrinsics
├── reports/                   # Tech-report statistics and figures
│   ├── compute_stats.py       #   Dataset-level statistics
│   └── figures.ipynb          #   Qualitative figure generation
├── episodes_success.txt       # Successful DROID episode IDs (notebooks/pipeline.ipynb samples from these)
├── assets/                    # Local assets (Franka + Robotiq URDF)
└── third_party/               # Gitignored: submodule source + downloaded weights
    ├── s2m2/                  #   Stereo matching model (submodule)
    │   └── weights/           #     S2M2 XL weights (fetched by setup.sh)
    └── sam_weights/           #   SAM ViT-H weights (fetched by setup.sh)
```

## Data Setup

The pipeline reads raw DROID data from a GCS bucket and writes outputs to another.
Use `mount_gcs.sh` to mount both via [gcsfuse](https://cloud.google.com/storage/docs/gcsfuse-cli):

```bash
bash mount_gcs.sh
```

| Mount | GCS Bucket / Prefix | Local Path |
|-------|---------------------|------------|
| Input (DROID raw) | `gs://gresearch/robotics/droid_raw` | `data/input/robotics/droid_raw` |
| Output | `gs://dm-tapnet/tmp/droid` | `data/output/droid` |

> To manually unmount:
> ```bash
> fusermount -u data/input/robotics/droid_raw
> fusermount -u data/output/droid
> ```

## Running Options

```bash
bash run_parallel.sh --mode depth              # depth, all episodes
bash run_parallel.sh --mode extrinsics        # extrinsics, all episodes
bash run_parallel.sh --mode tracks            # tracks, all episodes
bash run_parallel.sh --mode metrics           # quality metrics, all episodes
bash run_parallel.sh --mode depth --limit 32  # depth, first 32 episodes
```

| Flag | Short | Values | Default | Description |
|------|-------|--------|---------|-------------|
| `--mode` | `-m` | `depth`, `extrinsics`, `tracks`, `metrics` | required | Pipeline stage |
| `--limit` | `-l` | integer | all | Max episodes to process |

Jobs auto-scale to the number of GPUs detected by `nvidia-smi`.

## Episode Evaluation & Selection

After running all 3 stages, compute quality metrics across episodes and select a diverse evaluation set:

### Step 1: Batch Metrics (on GCP)

```bash
bash run_parallel.sh --mode metrics
```

Auto-detects GPUs and runs `compute_metrics.py` in parallel across all of them.

Outputs `metrics.csv` (shared across all ranks via file locking) with 30+ quality columns per episode:

| Category | Metrics |
|---|---|
| Extrinsics | Chamfer distance, robot depth loss per camera |
| Track consistency | Depth residual (median/mean mm) for static/robot/overall |
| Motion | End-effector travel distance, joint range, gripper range |
| Coverage | Depth valid-pixel percentage, per-camera visibility |
| Metadata | Site, robot ID, frame count, resolution |

### Step 2: Select

```bash
# Select 50 episodes (stratified by site + motion diversity)
python tapvidmv/select_episodes.py --n 50
```

Selection applies quality filtering (chamfer, depth residual thresholds),
site-proportional quotas, and within-site motion diversity (evenly spaced by EE travel).

Selection is deterministic: quotas are equal per *scene* (the middle field of
the episode id, 62 of them against 13 sites) and filled round-robin, and
`--min_ee_travel` drops episodes where the arm barely moves. Those pass every
quality threshold — a frozen arm has nothing to blur and no FK error to
accumulate — while being worth nothing to a tracking benchmark.

Run it with a larger `--n` than the release needs: it produces a candidate
pool, not the final set.

### Step 3: Pick

Open [`tapvidmv/pick_episodes.ipynb`](tapvidmv/pick_episodes.ipynb) and work
through the pool by eye. Each candidate is shown as one row per camera and
eight frames across the episode, beside its metrics; **Keep** / **Skip** /
**Back** build the set, and the last cell writes `episodes_eval50.txt`.

What the metrics cannot see is whether the manipulation is interesting, or
whether two candidates from different scenes are doing the same thing anyway.

### Step 4: Export

```bash
bash tapvidmv/run_export.sh                             # episodes_eval50.txt
bash tapvidmv/run_export.sh --list episodes_eval150.txt # a different set
bash tapvidmv/run_export.sh --list all                  # everything with tracks
```

Converts the selected episodes into the TAPVid-MV release layout. This runs
*after* selection: it re-encodes every frame to JPEG and writes the depth
maps, so exporting first and selecting second meant paying that over thousands
of episodes to keep fifty. CPU-only, so it sizes itself to the core count
rather than the GPU count — which is why it is a separate runner from
`run_parallel.sh` rather than another `--mode`.

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--list` | `-f` | `episodes_eval50.txt` | Episode list to export, or `all` |
| `--limit` | `-l` | all | Max episodes to export |

## Interactive Notebook

`notebooks/pipeline.ipynb` runs one episode at a time, for debugging. It works
both ways round: open it from a local checkout and it uses that checkout as-is
(from any directory inside it), or open it
[in Colab](https://colab.research.google.com/github/yangyi02/droid/blob/main/notebooks/pipeline.ipynb)
and the first cell clones the repo. Nothing is pulled or cloned over a local
working tree.

The notebook uses **3 global boolean flags** at the top (`COMPUTE_DEPTH`, `COMPUTE_EXTRINSICS`, `COMPUTE_TRACKS`):
- `True` — Compute stage from scratch
- `False` — Load pre-computed results directly from GCS

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

### Model Weights (downloaded by `setup.sh`)

| Model | Source | Path |
|-------|--------|------|
| S2M2 XL | HuggingFace `minimok/s2m2` | `third_party/s2m2/weights/` |
| SAM ViT-H | `dl.fbaipublicfiles.com` | `third_party/sam_weights/` |
