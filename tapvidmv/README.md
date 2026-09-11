# TAPVid-MV

Everything that turns the pipeline's output into the released evaluation set.

The pipeline produces depth, extrinsics, tracks and metrics for every episode it
can. This directory decides **which fifty of them are the benchmark**, and writes
those fifty in the release layout.

Nothing in the repository root knows about this directory. It reads the pipeline's
outputs through `config.paths`, and owns everything downstream of them — including
where the release lands (`RELEASE_ROOT` in `export_tapvidmv.py`, not `config.py`).

## Prerequisites

All four pipeline stages, over every episode:

```bash
bash run_parallel.sh depth
bash run_parallel.sh extrinsics
bash run_parallel.sh tracks
bash run_parallel.sh metrics
```

Step 1 below reads `<metrics>/<episode_id>/metrics.json`; steps 3 and 4 read the
depth videos and the tracks.

## The four steps

```
metrics ──▶ 1. calibrate the cuts  ─┐
                                    ├─ select_episodes.ipynb ──▶ episodes_eval100.txt
            2. draw the pool  ──────┘
                                        │
                                        ▼
            3. pick by eye ──── pick_episodes.ipynb ─────────▶ episodes_eval50.txt
                                        │
                                        ▼
            4. export ────────── run_export.sh ──────────────▶ data/release/tapvidmv/
                                        │
                                        ▼
                          visualize_tracks_groundtruth.ipynb
```

---

### Steps 1 & 2 — [`select_episodes.ipynb`](select_episodes.ipynb)

Two decisions off one table of numbers, which is why they share a notebook: the
sampling in step 2 runs on whatever step 1 leaves behind, so moving a threshold
redraws the pool.

**1. Which episodes are broken enough to drop.** The notebook plots every cut's
distribution with its threshold drawn on top, and counts what each one rejects —
both in total and *alone*. The second number is the one that matters: it is
exactly how many episodes relaxing that threshold buys back. A cut that rejects
nothing, or nothing another cut has already caught, is doing no work.

The thresholds in `CUTS` have never been calibrated against a full metrics run —
they were set from eight episodes that looked fine, with headroom. Calibrating
them is what the notebook is for.

**2. Which survivors go in the pool.** Quotas are equal per *scene* — the middle
field of the episode id, 62 of them against 13 sites — filled round-robin, taking
each scene's episodes in an order spread over end-effector travel. Scene rather
than site, so the pool spreads over camera placements and tabletops rather than
over labs. The notebook then shows the coverage that came out: sites, scenes, and
the motion spread the diversity is sampled along.

Writes `episodes_eval100.txt` and `episodes_eval100_details.csv`. The command
line reproduces whatever you settle on:

```bash
python tapvidmv/select_episodes.py --n 100 --cut cross_view_px=8.0
```

| Flag | Default | Description |
|---|---|---|
| `--n` | 150 | Size of the candidate pool |
| `--cut COLUMN=VALUE` | — | Move one threshold; repeatable |
| `--input` | `config.paths.metrics` | Directory of per-episode metrics |
| `--output_dir` | this directory | Where the list and CSV are written |

### Step 3 — [`pick_episodes.ipynb`](pick_episodes.ipynb)

Work through the pool by eye. Each candidate plays as a three-view clip with the
**ground-truth tracks drawn on it** — filled where that view calls a point
visible, hollow where it is occluded, with a short trail behind each dot — beside
the metrics that let it through. **Keep** / **Skip** / **Back** build the set,
and the last cell writes `episodes_eval50.txt`.

This is the only step that can catch tracks that are confidently wrong. Every cut
in step 1 is computed from the tracker's own residuals, so an episode whose
ground truth is wrong in a self-consistent way passes all of them. Nor can the
metrics see whether the manipulation is interesting, or whether two candidates
from different scenes are doing the same thing anyway.

Clips are cached under `previews/` (gitignored) and rendered a few ahead of the
one on screen, so the picker does not wait on video decoding.

### Step 4 — [`run_export.sh`](run_export.sh)

```bash
bash tapvidmv/run_export.sh                              # episodes_eval50.txt
bash tapvidmv/run_export.sh --list episodes_eval100.txt  # a different set
bash tapvidmv/run_export.sh --list all                   # everything with tracks
```

Converts the selected episodes into the release layout under
**`data/release/tapvidmv/`** — local disk, not the gcsfuse mount the pipeline
writes to. The export re-encodes every frame to JPEG and writes the depth maps,
which is far too many bytes to push through fuse, and publishing is a separate,
deliberate step.

This runs *after* selection: exporting first would mean paying that cost over
thousands of episodes to keep fifty. It is CPU-only, so it sizes itself to the
core count rather than the GPU count — which is why it is a separate runner from
`run_parallel.sh` rather than another pipeline stage.

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--list` | `-f` | `episodes_eval50.txt` | Episode list to export, or `all` |
| `--limit` | `-l` | all | Max episodes to export |

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

### Verify the export

Open [`visualize_tracks_groundtruth.ipynb`](visualize_tracks_groundtruth.ipynb).
It finds `data/release/tapvidmv/` on its own and draws the 3D tracks and the 2D
tracks they project to in each view — one episode closely, then a sweep over all
fifty. This reads the *released* files, so it checks the export itself, not the
pipeline's intermediate state.

## Publishing

Uploading the export is done by hand, outside this repository. The public layout
that [`download_episodes.sh`](download_episodes.sh) and
[`visualize_groundtruth_colab.ipynb`](visualize_groundtruth_colab.ipynb) expect
is:

```
gs://dm-tapnet/mv-tap/droid/tapvidmv/<episode_id>/...
```

Those two, plus [`verify_downloads.sh`](verify_downloads.sh), are tools for
*consumers* of the released dataset rather than steps in the pipeline — they
fetch from that public path and check what arrived. `visualize_groundtruth_colab.ipynb`
carries its own hard-coded list of the released episodes, so it needs updating
whenever the release set changes.

## Files

| File | Role |
|---|---|
| `select_episodes.py` | Quality cuts (`CUTS`, `judge`), scene-stratified sampling, list + CSV writing |
| `select_episodes.ipynb` | Steps 1 & 2 — calibrate the cuts, draw the pool |
| `pick_episodes.ipynb` | Step 3 — the human pass, ground truth drawn on every clip |
| `viz.py` | Track-drawing primitives shared by the notebooks: points, trails, montage, frame reading |
| `export_tapvidmv.py` | Step 4 — pipeline outputs → release layout; owns `RELEASE_ROOT` |
| `run_export.sh` | Parallel runner for the export, sized to the core count |
| `episodes_eval100.txt` | The candidate pool (step 2) |
| `episodes_eval50.txt` | The release set (step 3) |
| `visualize_tracks_groundtruth.ipynb` | 3D/2D ground-truth inspection of the export |
| `visualize_groundtruth_colab.ipynb` | Self-contained viewer for consumers, reads the public bucket |
| `download_episodes.sh` | Fetch the released episodes from the public bucket |
| `verify_downloads.sh` | Size-check those downloads, delete corrupt files |
| `archive/` | Superseded episode lists, kept for provenance — see `archive/README.md` |
