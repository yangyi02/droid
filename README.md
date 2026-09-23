# DROID Multi-View 3D Tracking Pipeline

Multi-stage pipeline for the [DROID dataset](https://droid-dataset.github.io/):
stereo depth extraction, camera-robot extrinsics calibration, and dense 3D point tracking.

## Quickstart

```bash
git clone --recurse-submodules https://github.com/yangyi02/droid.git
cd droid

# setup.sh installs into whichever python is active, and appends a cuDNN library
# path to venv/bin/activate -- so make the venv first, and activate it every time.
python3 -m venv venv && source venv/bin/activate

bash setup.sh              # everything, including the Stage 1 depth toolchain
bash setup.sh --no-depth   # skip s2m2, Segment Anything, the ZED SDK and the weights

bash mount_gcs.sh          # mount the input and output buckets

bash run_parallel.sh depth        # Stage 1
bash run_parallel.sh extrinsics   # Stage 2
bash run_parallel.sh tracks       # Stage 3
bash run_parallel.sh metrics      # Stage 4 (needs stage 2, not stage 3)

python compute_review.py --config.runner.limit=5   # Stage 5, then open the .rrd
```

Every path, threshold and optimizer setting lives in `config.py`, read through
`ml_collections.config_flags`, so any field can be overridden on the command line:

```bash
python compute_tracks.py --config.tracks.sensor_tolerance_base=0.03 --config.tracks.points_per_class=30
python compute_extrinsics.py --config.extrinsics.lr=0.005 --config.extrinsics.n_steps=800
python compute_depth.py --config.depth.max_frames=400 --config.runner.limit=20
python compute_tracks.py --config.render.gpu=False   # CPU rasteriser, for a box with no EGL
```

Cloned without `--recurse-submodules`? `bash setup.sh` runs
`git submodule update --init --recursive` for you.

## Pipeline

| Stage | Script | Reads | Writes |
|---|---|---|---|
| 1. Depth | `compute_depth.py` | raw SVO + `trajectory.h5` | `depth/` |
| 2. Extrinsics | `compute_extrinsics.py` | `depth/` | `extrinsics/` |
| 3. Tracks | `compute_tracks.py` | `depth/`, `extrinsics/` | `tracks/` |
| 4. Metrics | `compute_metrics.py` | `depth/`, `extrinsics/` | `metrics/` |
| 5. Review | `compute_review.py` | all of the above | `review/*.rrd` |

Stage 4 scores stage 2's poses and never opens a track file, so it can run beside stage 3.
Stage 5 is a few episodes you sit and watch, not a batch stage, so it is run directly rather
than through `run_parallel.sh`.

### Stage 1 — `compute_depth.py`

Decode ZED SVO stereo video and robot kinematics, then infer metric depth: S2M2 stereo
matching, a SAM gripper mask taken from the closed-gripper frames, a temporal median of the
depth inside that mask, and that distilled surface injected back over the stereo's guess at
the gripper.

**Output** — `depth/<episode_id>/`

```
robot.npz                      # joint_positions, T_ee_base_all, T_cam_ee_init, wrist_serial
<cam_serial>/
  calibration.npz              # K, baseline, distortion (rectified and raw)
  video_left.mp4               # also _right, _left_raw, _right_raw
  raw_depth.npz                # refined depth, uint16 mm
  original_raw_depth.npz       # pre-injection backup   ] wrist
  gripper_mask.npz             # SAM consensus mask     ] camera
  gripper_depth.npz            # distilled surface      ] only
```

### Stage 2 — `compute_extrinsics.py`

The robot is rasterised from each camera's current pose estimate and the resulting cloud is
aligned against the observed depth: `init_camera_states` reads the dataset's pre-calibrated
extrinsics from metadata, `per_camera_alignment` refines each camera on its own, and
`global_joint_alignment` optimises all of them together against a Chamfer term between camera
pairs plus a depth term per camera.

**Output** — `extrinsics/<episode_id>/<cam_serial>/extrinsics.json`, holding `base_extrinsic`
(4×4) and `extrinsics` (N×4×4). Both are **cam2world**; the export inverts them.

### Stage 3 — `compute_tracks.py`

Dense multi-view 3D tracks from a static background prior plus URDF forward kinematics — no
tracking model. It samples *before* it tracks: picking a point needs only where it sits and
which camera and frame it was born on, so tens of thousands of candidates never enter the
expensive pass, and only the few hundred that survive are carried through the episode and
projected into every view.

| Step | |
|---|---|
| `query_frames` | Frame 0, plus the best frame in each equal stretch of the rest: the one where the camera that sees the least of the arm sees the most of it. Frame 0 is there because a forward-only model has nothing to track from without it, and it takes the first stretch's place rather than being added, so no two queries sit a tenth of a second apart. Choosing beats spacing — on one episode an evenly spaced frame left a camera 53 candidates for a quota of 20 while another frame in the same stretch offered 2243. The cameras share the frames: each lands within a few percent of its own best, and it thirds the distinct query times a tracker must run a pass for |
| `find_static_candidates` | Background pixels that no camera puts on a depth edge, and that every camera able to see the place agrees about within `match_radius`. A camera abstains where it measured nothing, where the arm is in the way, or where the place is off its image; one that reads a surface *elsewhere* is reporting the point is not where we think |
| `find_robot_candidates` | Robot-mask pixels, kept in the frame of the link they sit on. No second camera is asked — the position comes from kinematics, not depth. A candidate the depth map calls hidden on its own birth frame is dropped: it could never be a query |
| `out_of_reach` | Drop background candidates the gripper ever closes within `gripper_clearance` of: those are the ones it carries away. It reads only 3D positions and robot poses, so it runs before the sampling |
| `sample_tracks` | Per camera per query frame, `points_per_class` on the arm and as many on the background, every pick as far as it goes from everything taken before it |
| `carry_robot` | The chosen arm points on every frame, by forward kinematics |
| `project_tracks` | Every point in every view at every frame: where it lands, and how far behind it the nearest surface sits — the rendered robot in units of `urdf_tolerance`, the depth map in units of its own noise at that range |
| `latch` | Turn that margin into a label. The arm gets two lines `hysteresis` either side of the cut so a point resting on it stops flickering; the background keeps the single cut |
| `settle` | Drop labels that flip for one frame and flip straight back — nothing on a rigid arm is revealed and hidden again in a thirtieth of a second |
| `never_seen_through` | Drop background points the depth map keeps looking through on more than `max_seen_through` of their clear-line frames: they sat on something that has since moved |

**A depth hole is the sensor failing, not the point hiding.** Stereo finds no match on dark,
thin or textureless surfaces. An earlier rule dropped any point sitting on a hole in any
camera and cost 17.6% of candidates — a wrist camera that could not measure a wall for
fourteen frames killed tracks the other two saw perfectly. So a hole is read as no evidence
either way, and the pipeline works around it in three places instead:

- Stage 1 fills the wrist camera's worst holes — the metallic gripper — with a surface
  distilled from the frames where it is closed.
- Candidates are not seeded where depth is unreliable. A background pixel needs a second
  camera to agree within `match_radius` and must not sit on a depth edge; a robot candidate
  the sensor calls occluded on its own birth frame is dropped.
- Where the sensor is silent the rendered robot answers alone: `sensor_slack` returns
  infinity, `project_tracks` turns it into a NaN, and `fmin` ignores it. The URDF is exact,
  and for a background point it is a foreign occluder that can only add occlusion.

The arm is the thing that moves in front of points, and its geometry is known exactly. What
is left over is a point occluded by something that is neither the robot nor measurable by
stereo; it reads visible.

The one margin that really is NaN — both readings missing — is a point that landed off the
image or behind the camera, and `latch` holds its previous label there.

A quota goes to a *camera* because a query is a pixel in one camera's video. Pooling the arm's
quota across cameras hands points out by surface area instead, and the surface the wrist camera
sees is a few percent of the arm — that camera came away with eight annotated points for a
whole episode. The gripper holding most of the wrist camera's points is not a bias to correct:
it is what that camera films.

Distances are measured at the first frame's pose, where the same spot on a link always lands in
the same place whatever the arm is doing. A stretch of surface covered on an earlier query
frame is the last place the next one looks, so the gripper — in view on every query frame — is
covered once rather than five times.

| Knob | |
|---|---|
| `num_query_frames` | How many frames points are born on, and so how many passes an evaluation costs per video |
| `points_per_class` | Points per class per camera per query frame. An episode holds `num_query_frames × views × 2 × points_per_class`. Area is deliberately not part of this: it decides how many candidates exist, not how many are wanted |
| `min_gap` | Metres between two candidates on the surface. Metres, not pixels: the wrist camera sits 15 cm from the gripper and the room cameras 65 cm from the arm, so a pixel grid hands the gripper six times the density of everything else |
| `urdf_tolerance` | How far behind the rendered robot a point may sit and still count as visible. Swept over ten episodes, the two errors it trades exchange about one for one anywhere between 0.5 cm and 1 cm and turn sharply worse outside that. The gripper's fingers are thinner than the tolerance, so a value covering pose error also covers a whole finger |
| `sensor_tolerance_base`, `sensor_tolerance_slope` | The same against measured depth, as `base + slope × range`. Stereo error grows with distance — binned by range, the gap between sensor noise and real occlusion sits near 2.5 cm at half a metre and near 4 cm at a metre and a half. Setting it by range also does away with naming the wrist camera; it is simply the close one |
| `hysteresis` | How far past the cut an arm point's margin must go before its label changes. Without it a point resting on the cut flips every few frames — and the worst affected are the ones that really do pass behind things, exactly the ones worth keeping |
| `max_seen_through`, `gripper_clearance`, `match_radius`, `max_edge_step`, `mask_margin` | Thresholds for the steps above |

**Output** — `config.paths.tracks/<episode_id>/`

```
tracks_3d.npz                  # tracks_3d
track_metadata.npz             # n_robot, n_static, query_view, query_frame
<cam_serial>/
  tracks_2d.npz                # tracks_2d + vis_2d
```

> `config.paths.tracks` currently points at `tapvidmv/data/tracks/` rather than
> `data/output/droid/tracks/`, so this run can be compared against the old one before either
> is thrown away. `config.py` carries the note to put it back.

Stage 3 is the expensive stage and the evaluation set is a small slice of stage 2's output, so
point it at a selection:

```bash
bash run_parallel.sh tracks "" --config.paths.episode_list=tapvidmv/episodes_eval150.txt
```

### Stage 4 — `compute_metrics.py`

One `metrics/<episode_id>/metrics.json` per episode, so ranks never share a file and a crash
costs only its own episode. It holds stage 2's own objective read at the pose it converged to
— `chamfer_*` and `overlap_*` per camera pair, `robot_loss_*` per camera — plus site, scene,
camera count and frame count. Nothing is reduced to a single "worst camera" number; whatever
reads them decides which view condemns an episode.

### Stage 5 — `compute_review.py`

Stage 3's tracks, all of them at once, the way you would judge them by eye. One Rerun recording
per episode built straight from stages 1–3 — no export, nothing frozen — so it runs on whatever
`compute_tracks.py` just wrote and is thrown away when the next setting is tried.

```bash
python compute_review.py --config.runner.limit=5    # ~1 min and ~370 MB per episode
rerun tapvidmv/data/review/<episode_id>.rrd
```

The 3D view holds every camera's depth cloud, the moving frustums, and every track — cyan for
the arm, amber for the background. Below it sits one 2D view per camera. Scrub the timeline and
a bad track shows up as a point sliding off its texture, or as a colour that disagrees with
what the image plainly shows.

`config.review.n_inspect` tracks are sampled at random (seeded, so an episode always shows the
same ones) and singled out for closer reading:

- **Every camera view** carries all of them as numbered dots — green where stage 3 called the
  point visible in that camera, red where it did not — plus a yellow cross at the query pixel
  of any born on this frame in this camera. The gap between cross and dot is reprojection error.
- **The 3D view** carries one at a time at `/inspect/<track>`: the point in magenta, a ray to
  every camera centre coloured green/red/blue (visible, hidden, outside that camera's frustum —
  a different thing from occluded), and one line of text with every verdict at once, e.g.
  `31 (robot) | cam0 hidden | cam1 visible | cam2 off-frame`. `*` marks the query frame, `!` a
  camera calling a point visible while it lands outside the image. Only the first tree starts
  visible: read a number off a camera view, tick that tree on and the previous one off.

Nothing pins the orbit — double-click a point to centre on it. Sharding is the same shuffle
every stage uses, so `--config.runner.limit=5` is the five episodes stage 3 ran first.

| Knob | |
|---|---|
| `depth_stride` | Every nth pixel of the depth map becomes a scene point. 4 is ~20 M points and ~370 MB for a 150-frame three-camera episode; 2 is four times that, and the whole recording must reach the viewer before it is useful |
| `scene_radius` | Metres. About half the spacing the stride leaves on the surface (`stride × range / focal`, ~2.7 mm at a metre with stride 4), or the cloud is full of gaps and an occluder cannot be told from empty air |
| `max_depth` | Metres. 2 m is the DROID tabletop; past it is the rest of the room |
| `n_inspect` | How many tracks get the closer reading above |
| `fps` | Playback speed in the viewer, not a claim about the source |

Recordings go to `tapvidmv/data/review/` on local disk — the bucket writes at a twentieth of
the speed and charges for the rename twice.

#### Looking at one from a laptop

Serve it where it was written and open it in the laptop's browser. Nothing is copied and
nothing is installed on the laptop, which also puts it out of reach of anything vetting
executables:

```bash
bash serve_review.sh <episode_id>      # on the machine holding the recording
```

Forward both ports — 9090 serves the viewer, 9876 the data — and open the URL the script
prints, which carries the data port as a query parameter. VS Code's Remote-SSH forwards from
its PORTS panel; otherwise `ssh -L 9090:localhost:9090 -L 9876:localhost:9876 <host>`.

Forward them **by hand**. A port the editor forwarded on its own, or one whose forward outlived
a server restart, goes on accepting connections after the tunnel behind it has died: the viewer
page loads and then sits on its welcome screen, because the data port is the dead one.
`curl -I --noproxy '*' http://localhost:9876` answers `400 Bad Request` when that tunnel is
alive and hangs when it is not.

`serve_review.sh` also raises the proxy's memory ceiling — the default is 1 GiB, and a larger
recording is served with its oldest messages quietly dropped.

To make one smaller while keeping its shape, drop scene entities:

```bash
rerun rrd filter --drop-entity /scene/1 --drop-entity /scene/2 \
    -o one-camera.rrd tapvidmv/data/review/<episode_id>.rrd
```

Dropping two of the three leaves one camera's cloud at a third of the size; dropping all three
leaves the RGB and every track at 6% of it.

## TAPVid-MV release

`tapvidmv/` turns the pipeline's output into the released evaluation set — which fifty episodes
are the benchmark, written in the release layout and checked.
See [`tapvidmv/README.md`](tapvidmv/README.md).

## Naming conventions

| Rule | |
|---|---|
| Identifiers | `episode_id`, `cam_id`, `cam_ids`, `wrist_cam_id`, `cam_data`, `cam_dir` |
| Transforms | `T_<from>2<to>` — `T_cam2world`, `T_world2cam`, `T_ee2base`, `T_link2world`. The direction is always in the name, so there is no bare `T_cam` or `T_init`. The exported `extrinsics_w2c.npy` uses the same idiom |
| Points vs tracks | `points_3d` is `(N, 3)`, positions with no time; `tracks_3d` is `(T, N, 3)`, a position per frame. Static candidates are points, robot candidates are already tracks |
| Frames on data | `points_cam`, `points_world` — the suffix names the frame the coordinates are in |
| Image size | `height`, `width` — never `h`/`w` |
| Images | `img_rgb`, `img_left`, `img_right` — modifier last, matching `video_rgb`, `video_right` |
| Counts | `n_` for things that exist (`n_frames`, `n_points`, `n_static`); `num_` only in `config.py`, where it is a cap being requested |
| Indices | `t` for a frame, `u`/`v` for a pixel |
| Math symbols | `K`, `T`, `R` stay symbols — everything else is complete words |
| Modules | don't repeat the module in its functions |

The two dicts threaded through every stage are `episode` (one episode's loaded data: `meta`,
`robot`, `camera`) and `poses` (per-camera extrinsics — what stage 2 estimates and stage 4
measures).

**On-disk keys are frozen and may disagree with the code.** `robot.npz` still says
`wrist_serial`, `T_ee_base_all` and `T_cam_ee_init`; `extrinsics.json` still says
`base_extrinsic` and `extrinsics`. 255 episodes are already computed and stage 1 is too
expensive to re-run for a name. `core/io.py` translates at the boundary.

## Layout

```
droid/
├── pipeline.ipynb             # Whole pipeline, one episode at a time
├── compute_depth.py           # Stage 1
├── compute_extrinsics.py      # Stage 2
├── compute_tracks.py          # Stage 3
├── compute_metrics.py         # Stage 4
├── compute_review.py          # Stage 5
├── config.py                  # Paths, buckets and every hyperparameter (ConfigDict)
├── setup.sh                   # Dependencies + weights (--no-depth skips Stage 1's toolchain)
├── mount_gcs.sh               # Mount the input and output buckets
├── run_parallel.sh            # One worker per GPU, episodes sharded by rank
├── serve_review.sh            # Serve one recording to a browser on your laptop
├── episodes_success.txt       # Episode ids pipeline.ipynb samples from
├── core/
│   ├── depth.py               #   S2M2 stereo, SAM gripper mask, depth distillation
│   ├── geometry.py            #   project, unproject, poses, farthest-point sampling
│   ├── io.py                  #   Metadata, depth/extrinsics/track loading
│   ├── physics.py             #   PyBulletRenderer: depth, mask and segmentation renders
│   ├── pointcloud.py          #   Robot/scene clouds, chamfer + overlap, depth loss
│   ├── runner.py              #   Episode sharding + resume-aware batch loop
│   └── visualization.py       #   Point clouds, tracking videos, 4D orbit
├── tapvidmv/                  # The released evaluation set -- see its own README
├── assets/                    # Franka + Robotiq URDF and meshes
└── third_party/               # Gitignored: submodule source + downloaded weights
```

## Data

The pipeline reads raw DROID data from one GCS bucket and writes to another; `mount_gcs.sh`
mounts both via [gcsfuse](https://cloud.google.com/storage/docs/gcsfuse-cli).

| Mount | Bucket / prefix | Local path |
|---|---|---|
| Input | `gs://gresearch/robotics/droid_raw` | `data/input/robotics/droid_raw` |
| Output | `gs://dm-tapnet/tmp/droid` | `data/output/droid` |

Unmount by hand with `fusermount -u <local path>`.

## Running

```bash
bash run_parallel.sh <stage> [limit] [--config.x=y ...]

bash run_parallel.sh depth          # all episodes
bash run_parallel.sh depth 32       # first 32
bash run_parallel.sh tracks "" --config.paths.episode_list=tapvidmv/episodes_eval150.txt
```

One worker per GPU detected by `nvidia-smi`, episodes sharded by rank, per-rank logs under
`logs/`. Every stage skips episodes it has already written, so a rerun resumes.

`pipeline.ipynb` runs one episode at a time for debugging. It sits at the repo root so its
working directory is the checkout, and works either way round: open it from a local checkout
and it uses that checkout as-is, or open it
[in Colab](https://colab.research.google.com/github/yangyi02/droid/blob/main/pipeline.ipynb)
and the first cell clones the repo. Three flags at the top — `COMPUTE_DEPTH`,
`COMPUTE_EXTRINSICS`, `COMPUTE_TRACKS` — choose between computing a stage and loading it from
GCS.

## Dependencies

**ZED SDK** (Stage 1 SVO decoding). Installed by `setup.sh`, runtime only. The SDK is
system-wide but `pyzed` is a wheel per interpreter and the SDK does not carry one:
`/usr/local/zed/get_python_api.py` fetches the one matching the active python. A new virtualenv
therefore needs that step even when `/usr/local/zed` is already there, which is why `setup.sh`
guards on whether `pyzed` imports rather than on whether the SDK exists.

**PyBullet** is deliberately absent from `requirements.txt`. PyPI ships no wheels, and a build
that cannot see numpy silently drops NumPy support, which `core.physics` requires and checks
for at startup. `setup.sh` builds it after numpy, with build isolation off, and skips the build
when `pybullet.isNumpyEnabled()` is already true.

**cuDNN** — this machine, not the pipeline. The VM image prepends `/usr/lib/x86_64-linux-gnu`
to `LD_LIBRARY_PATH`, which outranks the `DT_RUNPATH` inside pip's cuDNN: torch's convolutions
bind the older system `libcudnn_graph` and die with `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`.
`setup.sh` appends the wheel's own library directory to `venv/bin/activate`, so running a stage
without activating the venv brings the failure back.

| Fetched by `setup.sh` | Source | Path |
|---|---|---|
| s2m2 (submodule) | [junhong-3dv/s2m2](https://github.com/junhong-3dv/s2m2) | `third_party/s2m2/` |
| S2M2 XL weights | HuggingFace `minimok/s2m2` | `third_party/s2m2/weights/` |
| SAM ViT-H weights | `dl.fbaipublicfiles.com` | `third_party/segment_anything/weights/` |
