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
bash setup.sh              # everything, including the Stage 1 depth toolchain
bash setup.sh --no-depth   # skip s2m2, Segment Anything, the ZED SDK and the weights

# 4. Mount GCS input/output buckets
bash mount_gcs.sh

# 5. Run pipeline (3 stages)
bash run_parallel.sh depth        # Stage 1: depth
bash run_parallel.sh extrinsics   # Stage 2: extrinsics
bash run_parallel.sh tracks       # Stage 3: tracks
```

Every path, threshold and optimizer setting lives in `config.py`. The stage
scripts read it through `ml_collections.config_flags`, so any field can be
overridden on the command line without editing the file:

```bash
python compute_tracks.py --config.tracks.depth_tolerance=0.03 --config.tracks.num_static_points_per_view=200
python compute_extrinsics.py --config.extrinsics.lr=0.005 --config.extrinsics.n_steps=800
python compute_depth.py --config.depth.max_frames=400 --config.runner.limit=20
python compute_tracks.py --config.render.gpu=False  # CPU rasteriser, for a box with no EGL
```

> If you cloned **without** `--recurse-submodules`, run `bash setup.sh` —
> it calls `git submodule update --init --recursive` automatically.

## Pipeline Overview

| Stage | Script | Core Modules | Description |
|-------|--------|--------------|-------------|
| 1. Depth | `compute_depth.py` | `core.depth` | SVO decode → S2M2 stereo depth → SAM gripper mask → depth distillation |
| 2. Extrinsics | `compute_extrinsics.py` | `core.physics` | Dataset extrinsics → rendered robot alignment → global joint optimization |
| 3. Tracks | `compute_tracks.py` | `core.geometry`, `core.physics` | Static background depth consensus + URDF FK robot tracks (model-free) |

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
Every view's first frame is a query frame: the final sample takes up to a fixed quota of points from
each view, drawn from what that view sees in frame 0.

| Step | Description |
|------|-------------|
| `find_static_candidates` | Multi-view depth consensus over each view's first frame, deduplicated by voxel |
| `project_static_tracks` | Project static points into every view; the sensor depth gap labels visibility |
| `filter_static_tracks` | Drop points that recede from the depth map or flicker |
| `find_robot_candidates` | Every robot mask pixel in each view's first frame, carried through time by URDF forward kinematics |
| `project_robot_tracks` | Project robot points into every view; URDF and sensor depth label visibility |
| `filter_robot_tracks` | Drop points never visible in any view |
| `sample_static_tracks`, `sample_robot_tracks` | Keep up to `num_*_points_per_view` points visible in each view's frame 0 |
| `merge_tracks` | Merge static background & robot tracks with global visibility masks |

**Output** (`data/output/droid/tracks/<episode_id>/`):
```
tracks_3d.npz                  # tracks_3d
track_metadata.npz             # n_static, n_robot
<cam_serial>/
  tracks_2d.npz                # per-camera 2D tracks (tracks_2d) + visibility (vis_2d)
```

## Naming Conventions

One concept, one spelling, repo-wide. The pipeline files and the notebooks all follow these.

| Rule | |
|---|---|
| Identifiers | `episode_id`, `cam_id`, `cam_ids`, `wrist_cam_id`, `cam_data`, `cam_dir` |
| Transforms | `T_<from>2<to>` — `T_cam2world`, `T_world2cam`, `T_ee2base`, `T_cam2ee`, `T_link2world`. The prefix keeps the family greppable; the direction is always in the name, so there is no bare `T_cam` or `T_init`. The exported `extrinsics_w2c.npy` uses the same idiom |
| Points vs tracks | `points_3d` is `(N, 3)`, positions with no time; `tracks_3d` is `(T, N, 3)`, a position per frame. Static candidates are points, robot candidates are already tracks |
| Frames on data | `points_cam`, `points_world` — suffix names the frame the coordinates are in |
| Image size | `height`, `width` — never `h`/`w` or `h_img`/`w_img` |
| Images | `img_rgb`, `img_left`, `img_right` — modifier last, matching `video_rgb`, `video_right` |
| Percentages | `_percent`, spelled out (`vis_percent_<cam>`, `robot_percent_<cam>`) |
| Counts | `n_` for things that exist (`n_frames`, `n_points`, `n_static`); `num_` only in `config.py`, where it is a cap being requested |
| Indices | `t` for a frame, `u`/`v` for a pixel |
| Per-camera dicts | `per_cam_tracks`, `per_cam_vis` keyed by `cam_id`; one camera's array drops the prefix |
| Math symbols | `K`, `T`, `R` stay symbols — everything else is complete words |
| Modules | don't repeat the module in its functions (`compute_metrics.motion_stats`, not `compute_motion_stats`) |

The two dicts threaded through every stage are `episode` (one episode's loaded data:
`meta`, `robot`, `camera`) and `poses` (per-camera extrinsics, the thing stage 2 estimates
and stage 4 measures).

**On-disk keys are frozen and may disagree with the code.** `robot.npz` still says
`wrist_serial`, `T_ee_base_all` and `T_cam_ee_init`, and `extrinsics.json` still says
`base_extrinsic` and `extrinsics`, because 255 episodes are already computed and stage 1 is
too expensive to re-run for a name. The loaders in `core/io.py` translate at the boundary.

## Directory Structure

```
droid/
├── pipeline.ipynb             # Whole pipeline, one episode at a time (flag-based flow)
├── compute_depth.py           # Stage 1: SVO → stereo depth + gripper refinement
├── compute_extrinsics.py      # Stage 2: Dataset init + camera-robot alignment
├── compute_tracks.py          # Stage 3: Static prior + URDF FK dense 3D tracking
├── compute_metrics.py         # Batch quality metrics evaluation (GCP)
├── run_parallel.sh            # Multi-GPU parallel runner for the stages above
├── config.py                  # Paths, GCS buckets and every hyperparameter (ConfigDict)
├── setup.sh                   # One-shot dependency + weights setup (--no-depth skips Stage 1)
├── mount_gcs.sh               # GCS bucket mount helper
├── core/                      # Shared algorithmic modules
│   ├── geometry.py            #   3D math: unproject, project, pose_from_euler, rodrigues
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
│   ├── filter_points.ipynb    #   Dropping background points carried away by the gripper
│   ├── pybullet_numpy_benchmark.ipynb  # Why PyBullet must be built with NumPy support
│   ├── pybullet_egl_mask_benchmark.ipynb  # Why the GPU rasteriser is off, and what it would take
│   └── pybullet_gpu_pipeline_validation.ipynb  # gpu=True on a real episode: renders, point clouds, extrinsics
├── reports/                   # Tech-report statistics and figures
│   ├── compute_stats.py       #   Dataset-level statistics
│   └── figures.ipynb          #   Qualitative figure generation
├── episodes_success.txt       # Successful DROID episode IDs (pipeline.ipynb samples from these)
├── assets/                    # Local assets (Franka + Robotiq URDF)
└── third_party/               # Gitignored: submodule source + downloaded weights
    ├── s2m2/                  #   Stereo matching model (submodule)
    │   └── weights/           #     S2M2 XL weights (fetched by setup.sh)
    └── segment_anything/      #   Segment Anything
        └── weights/           #     SAM ViT-H weights (fetched by setup.sh)
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
bash run_parallel.sh depth         # depth, all episodes
bash run_parallel.sh extrinsics    # extrinsics, all episodes
bash run_parallel.sh tracks        # tracks, all episodes
bash run_parallel.sh metrics       # quality metrics, all episodes
bash run_parallel.sh depth 32      # depth, first 32 episodes
```

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| stage | `depth`, `extrinsics`, `tracks`, `metrics` | required | Pipeline stage |
| limit | integer | all | Max episodes to process |

One worker per GPU detected by `nvidia-smi`, episodes sharded by rank.

## Episode Evaluation & Selection

After running all 3 stages, compute quality metrics across episodes and select a diverse evaluation set:

### Step 1: Batch Metrics (on GCP)

```bash
bash run_parallel.sh metrics
```

Auto-detects GPUs and runs `compute_metrics.py` in parallel across all of them.

Outputs `<metrics>/<episode_id>/metrics.json`, one file per episode like every other stage, so
ranks never share a file and a crash costs only its own episode. Columns are keyed by camera
serial, the same names the depth, extrinsics and tracks directories already use:

| Category | Metrics |
|---|---|
| Extrinsics | `chamfer_*`, `overlap_*` and `robot_loss_*` per camera pair and camera — stage 2's own objective, read at the pose it converged to |
| Track consistency | `depth_residual_{static,robot}_mm_<cam>` per camera, and `cross_view_px_<cam>_<cam>` — what two cameras disagree by, in the pixels the benchmark is scored in |
| Motion | End-effector travel distance, joint range, gripper range, `track_jitter_mm` |
| Coverage | `vis_percent_<cam>` per camera, and `robot_percent_<cam>` — the share of each view's first frame the arm covers, for picking eval episodes |
| Metadata | Site, scene, camera count, frame count |

Nothing is reduced to a single "worst camera" number here: the metrics store every view, and
`tapvidmv/select_episodes.py` is where the worst of them condemns an episode.

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
`run_parallel.sh` rather than another stage.

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--list` | `-f` | `episodes_eval50.txt` | Episode list to export, or `all` |
| `--limit` | `-l` | all | Max episodes to export |

## Interactive Notebook

`pipeline.ipynb` runs one episode at a time, for debugging. It sits at the repo
root so that its working directory is the checkout. It works both ways round:
open it from a local checkout and it uses that checkout as-is, or open it
[in Colab](https://colab.research.google.com/github/yangyi02/droid/blob/main/pipeline.ipynb)
and the first cell clones the repo. Nothing is pulled or cloned over a local
working tree.

The notebook uses **3 global boolean flags** at the top (`COMPUTE_DEPTH`, `COMPUTE_EXTRINSICS`, `COMPUTE_TRACKS`):
- `True` — Compute stage from scratch
- `False` — Load pre-computed results directly from GCS

## Dependencies

### ZED SDK (required for Stage 1 SVO decoding)

Installed by `bash setup.sh` (runtime only, from `download.stereolabs.com/zedsdk/5.2/cu12/ubuntu22`),
together with the `pyzed` wheel it ships. `bash setup.sh --no-depth` skips it, along with
everything else only Stage 1 needs — use that when depth is loaded from GCS rather than recomputed.

### Git Submodules (auto-cloned with `--recurse-submodules`)

| Submodule | Repo | Notes |
|-----------|------|-------|
| `third_party/s2m2` | [junhong-3dv/s2m2](https://github.com/junhong-3dv/s2m2) | Stereo depth |

### Model Weights (downloaded by `setup.sh`, skipped by `--no-depth`)

| Model | Source | Path |
|-------|--------|------|
| S2M2 XL | HuggingFace `minimok/s2m2` | `third_party/s2m2/weights/` |
| SAM ViT-H | `dl.fbaipublicfiles.com` | `third_party/segment_anything/weights/` |
