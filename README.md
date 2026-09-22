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
python compute_review.py --config.runner.limit=5 && rerun tapvidmv/data/review/<episode_id>.rrd
```

Every path, threshold and optimizer setting lives in `config.py`. The stage
scripts read it through `ml_collections.config_flags`, so any field can be
overridden on the command line without editing the file:

```bash
python compute_tracks.py --config.tracks.sensor_tolerance_base=0.03 --config.tracks.points_per_class=30
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
The stage samples before it tracks: choosing a point needs only where it sits and which camera and
frame it was born on, so the tens of thousands of candidates never enter the expensive pass and only
the few hundred that are kept are carried through the episode and projected into every view.

| Step | Description |
|------|-------------|
| `query_frames` | One set for the episode: the first frame, which a forward-only model needs to track from and which is the only query covering the whole episode, and then in every equal stretch of the rest of it, the frame where the camera that sees the least of the arm sees the most of it. The stretches reach the end because ground truth does not run out — kinematics and a scene that stands still give a point born on the last frame its whole track before it. An evenly spaced frame takes whatever the arm happened to be doing — on one episode here the first frame left a camera 53 candidates to fill a quota of 20 from, while another frame in the same stretch offered 2243. The cameras share the frames because each camera's own best moment costs a tracker a separate pass: sharing leaves every camera within a few percent of its best and thirds the distinct query times |
| `find_static_candidates` | Background pixels within `config.tracks.max_depth` whose depth a second camera confirms, thinned to one candidate per `min_gap` cube |
| `find_robot_candidates` | Robot mask pixels, thinned the same way, kept in the frame of the link they sit on. No second camera is asked: the position comes from kinematics, not from depth. A candidate the depth map says is hidden on the frame it is born is dropped here — it could never be a query |
| `out_of_reach` | Drop background candidates the gripper ever closes on: those are the ones it carries away. It reads only the 3D positions and the robot's poses, so it runs before the sampling rather than after |
| `sample_tracks` | Each camera, on each of its query frames: `points_per_class` points on the arm and as many again on the background, every pick as far as it goes from every point taken before it |
| `carry_robot` | The chosen arm points on every frame, by forward kinematics |
| `project_tracks` | Every chosen point in every view at every frame. A point is visible where neither the rendered robot nor the depth map stands in front of it, judged per camera. A frame where this camera's stereo measured nothing is left unanswered, and the label holds through it |
| `latch` | Read the arm's depth margin with two lines rather than one; the background keeps the single cut. A point sitting on a single cut is labelled by measurement noise, so it takes `hysteresis` past the cut to call it hidden and the same the other way to call it visible again |
| `settle` | Drop labels that change for a single frame and change straight back: nothing on a rigid arm is revealed and hidden again in a thirtieth of a second |
| `never_seen_through` | Drop background points the depth map keeps looking straight through: they sat on something that moved. This one needs the whole episode, so it runs after the sampling and takes a few points back out of it |

A quota is handed to a camera because a query is a pixel in one camera's video. Pooling the arm's
quota across the cameras hands points out by surface area instead, and the surface the wrist camera
can see is a few percent of the arm, so that camera came away with eight annotated points for a whole
episode. The gripper holding most of the wrist camera's points is not a bias to correct here: it is
what that camera films. A 2D number pooled over three cameras this different is what makes it look
like one, and a per-camera number does not need saving from it.

Distances are measured at the first frame's pose, where the same spot on a link always lands in the
same place whatever the arm is doing. So a stretch of surface already covered on an earlier query
frame is the last place the next one looks, and the gripper — in view on every query frame — is
covered once rather than five times.

**A hole is not free space.** Stereo fails exactly where surfaces are dark, thin or textureless, and
those are also the things that occlude, so reading a missing depth as "nothing in the way" quietly
calls hidden points visible — on the wrist camera that was a fifth to a third of everything it called
visible. It cannot be read as "hidden" either: the surface is often simply untextured. So the question
goes to the other cameras, which are looking at the same scene on the same frame from somewhere else:
`core.scene.blocked` walks the ray from the camera to the point and asks whether any of them measured
a surface standing on it. Measured over two episodes, that answers about three quarters of the holes,
and the tenth or so of queries that reach it drops to under 2% with no answer at all, which default to
visible.

A background model fused over the whole episode was built and measured against this, and it answered
almost nothing the other cameras had not already answered — the cameras that can fill a hole are the
ones looking from elsewhere, and extra frames add nothing while the scene cameras stand still. It also
carried every surface that was ever moved as an occluder no longer there. It is not in the pipeline.

**Output** (`config.paths.tracks/<episode_id>/`, temporarily `tapvidmv/data/tracks/` rather than
`data/output/droid/tracks/`, so this run can be compared against the old one before either is thrown
away — `config.py` carries the note to put it back):
```
tracks_3d.npz                  # tracks_3d
track_metadata.npz             # n_static, n_robot, query_view, query_frame
<cam_serial>/
  tracks_2d.npz                # per-camera 2D tracks (tracks_2d) + visibility (vis_2d)
```

Stage 3 is the expensive stage and the evaluation set is a small slice of what stage 2 produced, so
point it at a selection rather than at everything:

```bash
bash run_parallel.sh tracks "" --config.paths.episode_list=tapvidmv/episodes_eval150.txt
```

| Knob | |
|---|---|
| `num_query_frames` | How many frames points are born on, and so how many passes an evaluation costs per video |
| `points_per_class` | Background points per camera per query frame, and the arm's whole quota for that frame is the same number times the cameras -- one object all of them are looking at, rather than one each. Handing the arm out per camera instead spent a third of it on the gripper, the only thing the wrist camera can see. An episode holds `num_query_frames × views × 2 × points_per_class` of them. Area is deliberately not part of this: it decides how many candidates there are, not how many points are wanted, and while it did decide the split the background — always the larger surface — spent the arm's budget on most frames |
| `min_gap` | Metres between two candidates on the surface. In metres and not in pixels because a grid on the image measures the camera rather than the scene: the wrist camera sits 15 cm from the gripper and the room cameras 65 cm from the arm, so a pixel grid hands the gripper six times the density of everything else, and sampling runs out of arm to pick from long before it runs out of budget |
| `urdf_tolerance` | How far behind the rendered robot a point may sit and still count as visible. Swept over ten episodes against what the depth map says where it is unambiguous, the two errors it trades — hiding a point the depth puts right at the surface, showing one the depth puts well behind a surface — exchange at about one for one anywhere between 0.5 cm and 1 cm, and turn sharply worse outside that. It matters more than it looks: the gripper's fingers are thinner than the tolerances, so a value that covers pose error also covers the whole finger, and every point on the far side of one comes back visible |
| `sensor_tolerance_base`, `sensor_tolerance_slope` | The same against the measured depth, as `base + slope × range`. Stereo error grows with distance, so one number cannot serve the whole scene: binned by range, the gap between the sensor noise and real occlusion sits near 2.5 cm at half a metre and near 4 cm at a metre and a half. Setting it by range also does away with naming the wrist camera — it is simply the close one |
| `hysteresis` | How far past the cut at -1 an arm point's depth margin has to go before its label changes; the background is not read this way. Below it lies the band where the reading cannot tell an occlusion from its own noise, so the label simply stays where it was. Without it a point resting on the cut flips on and off every few frames, and the points worst affected are the ones that really do pass behind something over and over — exactly the ones worth keeping |
| `max_seen_through` | The share of the frames with a clear line to a background point on which the depth map may look straight through it before the point is dropped |
| `gripper_clearance` | How close the gripper has to come to a background point for it to be treated as something that will be carried away. Measured from the joint centres, which sit a few centimetres inside the fingers |

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
rerun tapvidmv/data/review/<episode_id>.rrd
```

The 3D view holds the depth cloud of every camera, the camera frustums moving through it,
and every track — cyan for the robot's URDF tracks, amber for the static ones. Below it
sits one 2D view per camera. Scrub the timeline and a bad track shows up as a point that
slides off its texture, or as a colour that disagrees with what the image plainly shows.

`config.review.n_inspect` tracks are picked out to be judged one at a time, spread over the
scene, half on the arm and half on the background. Each camera view carries all of them at
once as numbered dots, green where stage 3 annotated the point visible in that camera and
red where it did not, plus a yellow cross at the query pixel of any that were born on this
frame in this camera — the gap between cross and dot is reprojection error. The numbers are
all the 2D views say, because a verdict per track per view would bury the image.

The 3D view carries one of them at a time: the point in magenta, a line from every camera
centre — green where that camera annotated it visible, red where it annotated it hidden, and
blue where the point is outside that camera's frustum altogether, which is a different thing
from being occluded — and a line of text with every camera's verdict at once
(`31 (robot) | cam0 hidden | cam1 visible | cam2 off-frame`, where `*` marks the query frame
and `!` marks a camera calling a point visible while it lands outside the image). Each track
is its own entity tree, `/inspect/<track>`, and only the first starts visible: read a number
off a camera view, tick that tree on and the previous one off, and you have switched.

Nothing pins the orbit, so dragging rotates about whatever you last centred on. Double-click
a point in the 3D view to centre on it.

Sharding is the same shuffle every stage uses, so `--config.runner.limit=5` is the five
episodes stage 3 ran first. Each episode prints what it wrote, including how many
observations are annotated visible while projecting outside their own image — a
contradiction, and normally a handful at the border.

| Knob | |
|---|---|
| `config.review.scene_radius` | Metres. How big a scene point is drawn. It has to be about half the spacing the stride leaves on the surface — `stride × range / focal length`, so ~2.7 mm at a metre with stride 4 — or the cloud is full of gaps and an occluder cannot be told from empty air |
| `config.review.depth_stride` | Every nth pixel of the depth map becomes a scene point. 4 is ~20 M points and ~370 MB for a 150-frame three-camera episode, which a browser tab opens without complaint; 2 is four times that and 1 is sixteen, and the whole recording has to reach the viewer before it is useful |
| `config.review.max_depth` | Metres. 2 m is the DROID tabletop — anything past it is the rest of the room |
| `config.review.inspect_track` | Which track the overlay starts on. -1 picks a background one, so the eye pivots on a point that holds still rather than swinging the scene around with the arm |
| `config.review.live` | Stream to a viewer instead of writing a file, and switch tracks by typing their number |
| `config.review.fps` | Playback speed in the viewer, not a claim about the source |

**Output** (`tapvidmv/data/review/<episode_id>.rrd`, on local disk — the bucket writes at a twentieth of the speed and charges for the rename twice): one recording per episode, and
the viewer streams a whole one into memory when it opens, so they are looked at one at a
time and thrown away when stage 3 changes.

### Switching tracks without rebuilding

The overlay lives at one set of entity paths — `/selected/**` and `/views/<n>/selected/**` —
so logging it again is what switches tracks. `--config.review.live=True` opens a viewer with
nothing in it, streams one episode into it, and then reads track numbers from stdin:

```bash
python compute_review.py --config.review.live=True \
    --config.paths.episode_list=tapvidmv/episodes_eval150.txt
```

It prints the viewer URL, which track it is showing, and what each track it switches to is —
robot or background, and the frame and camera it was born in. Type a number to switch, blank
to quit. Nothing is written to disk, so the ports are the same two and the recording never
goes stale.

The wait before anything appears is the depth cloud: 33 M points take a few minutes through
the proxy. `--config.review.depth_stride=8` cuts that to a quarter when the tracks, not the
scene, are what is being judged.

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
    -o one-camera.rrd tapvidmv/data/review/<episode_id>.rrd
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
