# TAPVid-MV

Everything that turns the pipeline's output into the released evaluation set.

The pipeline produces depth, extrinsics, tracks and metrics for every episode it
can. This directory decides **which fifty of them are the benchmark**, writes
those fifty in the release layout, and checks that what it wrote is right.

Nothing in the repository root knows about this directory. It reads the
pipeline's outputs through `config.paths` and owns everything downstream —
including where the release lands (`RELEASE_ROOT` in `export.py`, not
`config.py`) and everything it generates (`data/`, `previews/`, `logs/`, all
gitignored).

## Prerequisites

All four pipeline stages, over every episode:

```bash
bash run_parallel.sh depth
bash run_parallel.sh extrinsics
bash run_parallel.sh tracks
bash run_parallel.sh metrics
```

## The four steps

```
metrics ──▶ 1. calibrate the cuts ─┐
                                   ├─ shortlist.ipynb ─▶ episodes_eval100.txt
            2. draw the pool ──────┘
                                       │
                                       ▼
            3. pick by eye ─── review.ipynb ──────────▶ episodes_eval50.txt
                                       │
                                       ▼
            4. export ─────── export.py ────────────▶ tapvidmv/data/
                                       │
                                       ▼
                            verify.ipynb
```

---

### Steps 1 & 2 — [`shortlist.ipynb`](shortlist.ipynb)

Two decisions off one table of numbers, which is why they share a notebook: the
sampling in step 2 runs on whatever step 1 leaves behind, so moving a threshold
redraws the pool.

**1. Which episodes are broken enough to drop.** The notebook plots every cut's
distribution with its threshold drawn on top, and counts what each one rejects —
both in total and *alone*. The second number is the one that matters: it is
exactly how many episodes relaxing that threshold buys back. A cut that rejects
nothing, or nothing another cut has already caught, is doing no work.

The thresholds in `CUTS` were set from eight episodes that looked fine, with
headroom, and have never been calibrated against a full metrics run. Across the
first 40 real episodes, seven of the eight rejected nothing at all. Calibrating
them is what the notebook is for.

**2. Which survivors go in the pool.** Quotas are equal per *scene* — the middle
field of the episode id, 62 of them against 13 sites — filled round-robin. Scene rather
than site, so the pool spreads over camera placements and tabletops rather than
over labs. The notebook then shows the coverage that came out: sites and scenes.

Writes `episodes_eval100.txt`. The command line reproduces whatever you settle on:

```bash
python tapvidmv/shortlist.py --n 100 --cut chamfer=0.040
```

| Flag | Default | Description |
|---|---|---|
| `--n` | 150 | Size of the candidate pool |
| `--cut COLUMN=VALUE` | — | Move one threshold; repeatable |
| `--input` | `config.paths.metrics` | Directory of per-episode metrics |
| `--output_dir` | this directory | Where the list and CSV are written |

### Step 3 — [`review.ipynb`](review.ipynb)

Work through the pool by eye. Each candidate plays as a three-view clip with the
**ground-truth tracks drawn on it** — filled where that view calls a point
visible, hollow where it is occluded, with a short trail behind each dot — beside
the metrics that let it through. **Keep** / **Skip** / **Back** build the set,
and the last cell writes `episodes_eval50.txt`.

This is the only step that can catch tracks that are confidently wrong. Every cut
in step 1 is computed from the pipeline's own self-consistency, so an episode
whose ground truth is wrong in a self-consistent way passes all of them. Nor can the
metrics see whether the manipulation is interesting, or whether two candidates
from different scenes are doing the same thing anyway.

Clips are cached under `previews/` and rendered a few ahead of the one on screen,
so the picker does not wait on video decoding.

### Step 4 — [`export.py`](export.py)

```bash
python tapvidmv/export.py                                     # episodes_eval50.txt
python tapvidmv/export.py --episode_list episodes_eval100.txt
python tapvidmv/export.py --episode_id AUTOLab+5d05c5aa+2023-10-14-21h-59m-22s
```

Writes the release layout into **`tapvidmv/data/`** — local disk, not the gcsfuse
mount the pipeline writes to. The export re-encodes every frame to JPEG and
writes the depth maps, which is far too many bytes to push through fuse, and
publishing is a separate step done by hand.

It runs after selection, not before: exporting first would mean paying that cost
over thousands of episodes to keep fifty. Fifty run fine serially; it skips
episodes already present, so an interrupted run resumes.

**Budget the disk.** `depth.npy` is float32 metres, twice the size of the uint16
millimetres on disk, so one episode is ~1.9 GB at the median 170 frames and
**fifty come to roughly 95 GB**. Check `df -h` first, or export in batches and
upload as you go.

| Flag | Default | Description |
|---|---|---|
| `--episode_list` | `episodes_eval50.txt` | List to export, or `all` for everything with tracks |
| `--episode_id` | — | A single episode, overriding the list |
| `--limit` | all | Max episodes |
| `--output_root` | `tapvidmv/data` | Where the release is written |
| `--no_depth` | off | Skip `depth.npy` — much smaller, for checking the flow |
| `--jpeg_quality` | 95 | |

Per episode the layout is:

```
<episode_id>/
├── tracks_xyz.npy          (frames, points, 3)   world-space 3D tracks
├── queries_xytv.npy        (points, 4)           x, y, t, query view
└── <view>/                 one directory per camera, 0-indexed
    ├── images_jpeg_bytes.npy   (frames,)  object array of JPEG buffers
    ├── intrinsics.npy          (4,)       fx, fy, cx, cy
    ├── extrinsics_w2c.npy      (frames, 4, 4)
    ├── visibility.npy          (frames, points)
    ├── depth.npy               (frames, height, width)
    └── foreground_mask.npy     (frames, height, width)   wrist view only
```

### Verify — [`verify.ipynb`](verify.ipynb)

Reads the **exported** files, so it checks the thing that ships rather than the
pipeline's intermediate state. It finds `tapvidmv/data/` on its own.

The questions sharpen as you go down: is the rig what we think it is (one camera
on the wrist, two fixed) → do the 2D tracks stay glued to their surfaces → do the
3D tracks lie *on* the depth cloud or float above it → do the views agree about
what is visible and where the queries live → is the depth sane inside the 2 m
workspace and does the wrist mask fit → read a point's depth in one view and
reproject it into another, how far off → all fifty at once, which need opening.

That last reprojection check is the sharpest one.

## Publishing

Uploading is done by hand, outside this repository. The public layout is:

```
gs://dm-tapnet/mv-tap/droid/tapvidmv/<episode_id>/...
```

## Files

| File | Role |
|---|---|
| `shortlist.py` | Quality cuts (`CUTS`, `judge`), scene-stratified sampling, list + CSV writing |
| `shortlist.ipynb` | Steps 1 & 2 — calibrate the cuts, draw the pool |
| `review.ipynb` | Step 3 — the human pass, ground truth drawn on every clip |
| `export.py` | Step 4 — pipeline outputs → release layout; owns `RELEASE_ROOT` |
| `verify.ipynb` | Verification — reads the export, seven ways of asking whether it is right |
| `release.py` | Reads the release layout: `View`/`Episode`, projection, unprojection |
| `viz.py` | Drawing primitives: points, trails, montage, frame reading |
| `data/` | The export (gitignored) |
| `previews/` | Cached picker clips (gitignored) |
