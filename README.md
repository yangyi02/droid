# DROID Multi-View 3D Tracking Pipeline

Multi-stage pipeline for the [DROID dataset](https://droid-dataset.github.io/):
stereo depth extraction, camera-robot extrinsics calibration, and dense 3D point tracking.

## Quickstart

```bash
# 1. Clone with all dependencies
git clone --recurse-submodules https://github.com/yangyi02/droid.git
cd droid

# 2. Create the virtualenv -- setup.sh installs into whichever python is
#    active, so without this it goes into the system interpreter. It also
#    appends a cuDNN library path to venv/bin/activate, so every later run
#    starts by activating this venv
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies + download model weights
bash setup.sh              # everything, including the Stage 1 depth toolchain
bash setup.sh --no-depth   # skip s2m2, Segment Anything, the ZED SDK and the weights

# 4. Mount GCS input/output buckets
bash mount_gcs.sh

# 5. Run pipeline (4 stages)
bash run_parallel.sh depth        # Stage 1: depth
bash run_parallel.sh extrinsics   # Stage 2: extrinsics
bash run_parallel.sh tracks       # Stage 3: tracks
bash run_parallel.sh metrics      # Stage 4: quality metrics (needs stage 2, not stage 3)

# 6. Look at what stage 3 produced, on a handful of episodes (optional)
python compute_review.py --config.runner.limit=5 && rerun data/output/droid/review/<episode_id>.rrd
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
| 4. Metrics | `compute_metrics.py` | `core.pointcloud` | Per-episode quality numbers: the extrinsics objective, read at the converged pose |
| 5. Review | `compute_review.py` | `core.geometry`, `rerun` | Stage 3's tracks as a Rerun recording: every point in 3D over the depth cloud, and reprojected into every camera |

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
| `find_static_candidates` | Multi-view depth consensus over each view's first frame, within `config.tracks.max_depth` of the camera, deduplicated by voxel |
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

### Stage 4 — `compute_metrics.py`

Quality numbers for one episode, written as `<metrics>/<episode_id>/metrics.json` — one
file per episode like every other stage, so ranks never share a file and a crash costs
only its own episode. It scores stage 2's poses and never opens a track file, so it runs
off the extrinsics directory and can go in parallel with stage 3. Columns are keyed by
camera serial, the same names the depth and extrinsics directories already use:

| Category | Metrics |
|---|---|
| Extrinsics | `chamfer_*`, `overlap_*` and `robot_loss_*` per camera pair and camera — stage 2's own objective, read at the pose it converged to |
| Metadata | Site, scene, camera count, frame count |

Nothing is reduced to a single "worst camera" number: the metrics keep every view, and
leave it to whatever reads them to decide which view condemns an episode.

### Stage 5 — `compute_review.py`

Stage 3's tracks, all of them at once, the way you would judge them by eye. One Rerun
recording per episode, built straight from stages 1–3: no TAPVid-3D export, nothing
frozen, so it runs on whatever `compute_tracks.py` just wrote and is thrown away when the
next setting is tried.

```bash
python compute_review.py --config.runner.limit=5    # ~1 min and ~370 MB per episode
rerun data/output/droid/review/<episode_id>.rrd
```

The 3D view holds the depth cloud of every camera, the camera frustums moving through it,
and every track — cyan for the robot's URDF tracks, amber for the static ones. Below it
sits one 2D view per camera: its RGB with every track reprojected onto it, green where
stage 3 annotated the point visible in that camera and red where it did not. Scrub the
timeline and a bad track shows up as a point that slides off its texture, or as a colour
that disagrees with what the image plainly shows.

A few tracks (`config.review.n_inspect`, spread over the scene, half robot and half
static) carry the whole single-point overlay on top of that: the point in magenta with the
trail of where it has just been, a line from every camera centre — green where that camera
annotated the point visible, red where it annotated it hidden, and blue where the point is
outside that camera's frustum altogether, which is a different thing from being occluded —
the marker where it lands in each image with a one-line verdict
(`VISIBLE`, `NOT VISIBLE | outside image`, `INCONSISTENT`, `QUERY FRAME`), and the yellow
cross at the query pixel it was born at — the gap between cross and marker on the query
frame is reprojection error. Each one is its own entity tree, `/inspect/<track>`, and only
the first one's rays start visible, because three rays read and a dozen do not. Ticking a
different track's rays on in the 3D view's entity tree is how you switch between them.

Sharding is the same shuffle every stage uses, so `--config.runner.limit=5` is the five
episodes stage 3 ran first. Each episode prints what it wrote, including how many
observations are annotated visible while projecting outside their own image — a
contradiction, and normally a handful at the border.

| Knob | |
|---|---|
| `config.review.depth_stride` | Every nth pixel of the depth map becomes a scene point. 4 is ~20 M points and ~370 MB for a 150-frame three-camera episode, which a browser tab opens without complaint; 2 is four times that and 1 is sixteen, and the whole recording has to reach the viewer before it is useful |
| `config.review.max_depth` | Metres. 2 m is the DROID tabletop — anything past it is the rest of the room |
| `config.review.n_inspect` | How many tracks carry the full overlay. Every one of them adds a verdict label to each camera view, so a handful stays readable |
| `config.review.fps` | Playback speed in the viewer, not a claim about the source |

**Output** (`data/output/droid/review/<episode_id>.rrd`): one recording per episode, and
the viewer streams a whole one into memory when it opens, so they are looked at one at a
time and thrown away when stage 3 changes.

To look at one from a laptop, serve it where it was written and open it in the laptop's
browser. Nothing is copied and nothing is installed on the laptop, which also puts it out
of reach of anything that vets executables:

```bash
bash serve_review.sh <episode_id>      # on the machine holding the recording
```

Then forward both ports — 9090 serves the viewer, 9876 serves the data — and open the
URL the script prints, which carries the data port as a query parameter:
`http://localhost:9090?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9876%2Fproxy`. Without it the
viewer loads with nothing in it, and the source can be added by hand from its Sources
panel instead. VS Code's Remote-SSH forwards the ports from its PORTS panel;
otherwise `ssh -L 9090:localhost:9090 -L 9876:localhost:9876 <host>`. A laptop that can
run the viewer natively can skip the browser and connect to the data port instead, with
`rerun rerun+http://127.0.0.1:9876/proxy`.

`serve_review.sh` raises the proxy's memory ceiling, which matters: the default is 1 GiB,
and a recording larger than that is served with its oldest messages quietly dropped.

Forward both ports by hand rather than letting the editor do it. A port the editor
forwarded on its own, or one whose forward outlived a restart of the server, goes on
accepting connections after the tunnel behind it has died: the viewer page still loads and
then sits on its welcome screen, because the data port is the one that is dead.
`curl -I --noproxy '*' http://localhost:9876` answers `400 Bad Request` when that tunnel
is alive and hangs when it is not.

The recording is written tracks first and cameras second, and the camera pass interleaves
the views frame by frame, so the tracks are there to look at while the cloud is still
arriving and the three camera views fill in together rather than one after another. All of
it still has to reach the viewer, and `rerun rrd filter` makes that smaller while the
recording keeps its shape: dropping two of the three `/scene/<view>` entities leaves one
camera's cloud at a third of the size, and dropping all three leaves the RGB and every
track at 6% of it.

```bash
rerun rrd filter --drop-entity /scene/1 --drop-entity /scene/2 \
    -o one-camera.rrd data/output/droid/review/<episode_id>.rrd
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
| Percentages | `_percent`, spelled out (`vis_percent_<cam>`) |
| Counts | `n_` for things that exist (`n_frames`, `n_points`, `n_static`); `num_` only in `config.py`, where it is a cap being requested |
| Indices | `t` for a frame, `u`/`v` for a pixel |
| Per-camera dicts | `per_cam_tracks`, `per_cam_vis` keyed by `cam_id`; one camera's array drops the prefix |
| Math symbols | `K`, `T`, `R` stay symbols — everything else is complete words |
| Modules | don't repeat the module in its functions (`compute_metrics.track_stats`, not `compute_track_stats`) |

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
├── compute_review.py          # Stage 5: Rerun recordings of stage 3's tracks
├── serve_review.sh            # Serve one recording to a browser on your laptop
├── run_parallel.sh            # Multi-GPU parallel runner for the stages above
├── config.py                  # Paths, GCS buckets and every hyperparameter (ConfigDict)
├── setup.sh                   # One-shot dependency + weights setup (--no-depth skips Stage 1)
├── mount_gcs.sh               # GCS bucket mount helper
├── core/                      # Shared algorithmic modules
│   ├── geometry.py            #   3D math: unproject, project, pose_from_euler, rodrigues
│   ├── io.py                  #   Data loading: get_accelerator, load_depth/extrinsics
│   ├── depth.py               #   S2M2 stereo, SAM gripper mask, depth distillation
│   ├── physics.py             #   PyBulletRenderer + robot point clouds and depth losses
│   ├── pointcloud.py          #   Robot/scene clouds, chamfer + overlap, robot depth loss
│   ├── runner.py              #   Episode sharding + resume-aware batch loop
│   └── visualization.py       #   Visualization helpers (point clouds, tracking videos, 4D orbit)
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

One worker per GPU detected by `nvidia-smi`, episodes sharded by rank. `compute_review.py`
is deliberately not in the list: it is a few episodes you then sit and watch, not a batch
stage, so it is run directly with `--config.runner.limit`.

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

Installed by `bash setup.sh` (runtime only, from `download.stereolabs.com/zedsdk/5.2/cu12/ubuntu22`).
The SDK is system-wide, but `pyzed` is a wheel per interpreter and the SDK does not carry one:
the installer's `/usr/local/zed/get_python_api.py` fetches the wheel matching the active python.
A new virtualenv therefore needs that step even when `/usr/local/zed` is already there, which is
why `setup.sh` guards on whether `pyzed` imports rather than on whether the SDK exists.
`bash setup.sh --no-depth` skips it, along with everything else only Stage 1 needs — use that when
depth is loaded from GCS rather than recomputed.

### PyBullet (built from source)

Deliberately absent from `requirements.txt`. PyPI ships no pybullet wheels, and a build that cannot
see numpy silently drops NumPy support, which `core.physics` depends on — see
`notebooks/pybullet_numpy_benchmark.ipynb`. `setup.sh` builds it after numpy is installed and with
build isolation off, and skips the build when `pybullet.isNumpyEnabled()` is already true.

### cuDNN (this machine, not the pipeline)

The VM image prepends `/usr/lib/x86_64-linux-gnu` to `LD_LIBRARY_PATH`, which outranks the
`DT_RUNPATH` inside pip's cuDNN: torch's convolutions bind the older system `libcudnn_graph` and
die with `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`. `setup.sh` appends the wheel's own library
directory to `venv/bin/activate`, so running a stage without activating the venv brings the failure
back.

### Git Submodules (auto-cloned with `--recurse-submodules`)

| Submodule | Repo | Notes |
|-----------|------|-------|
| `third_party/s2m2` | [junhong-3dv/s2m2](https://github.com/junhong-3dv/s2m2) | Stereo depth |

### Model Weights (downloaded by `setup.sh`, skipped by `--no-depth`)

| Model | Source | Path |
|-------|--------|------|
| S2M2 XL | HuggingFace `minimok/s2m2` | `third_party/s2m2/weights/` |
| SAM ViT-H | `dl.fbaipublicfiles.com` | `third_party/segment_anything/weights/` |
