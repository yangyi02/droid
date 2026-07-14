#!/usr/bin/env python3
"""Ablation experiment runner for MV-TAP extrinsics calibration.

Iterates over experiment configs × episodes, runs the extrinsics calibration
pipeline with different settings, and saves structured metric results.

Usage (from repo root on Colab or any machine with GPU):
  python run_extrinsics_ablation.py --episodes 10 --configs E0,E4
  python run_extrinsics_ablation.py --episode_id "ILIAD+5e938e3b+2023-07-20-11h-50m-51s" --configs E0
  python run_extrinsics_ablation.py --all  # run everything

Results are saved to:
  ~/droid_data/output/ablation/<config_id>/<episode_id>/metrics.json
  ~/droid_data/output/ablation/summary.csv
"""

import argparse
import copy
import csv
import gc
import json
import os
import random
import sys
import time
import traceback

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Episode selection
# ---------------------------------------------------------------------------

# Diverse set of 15 episodes spanning different labs and tasks.
# Hand-picked for variety: ILIAD, AUTOLab, TRI, REAL, IRIS, RAIL, IPRL, etc.
DEFAULT_EPISODES = [
    "ILIAD+5e938e3b+2023-07-20-11h-50m-51s",
    "AUTOLab+0d4edc83+2023-10-27-20h-11m-25s",
    "TRI+52ca9b6a+2023-11-27-15h-54m-57s",
    "AUTOLab+0d4edc83+2023-12-02-13h-31m-12s",
    "TRI+52ca9b6a+2023-12-05-15h-19m-45s",
    "REAL+75b7b0f9+2023-06-23-16h-46m-08s",
    "IRIS+7dfa2da3+2023-05-11-14h-12m-57s",
    "RAIL+d027f2ae+2023-11-04-14h-49m-32s",
    "IPRL+edf28ef3+2024-01-01-10h-36m-18s",
    "AUTOLab+84bd5053+2023-09-04-19h-24m-59s",
    "GuptaLab+553d1bd5+2023-05-28-16h-48m-13s",
    "PennPAL+c5f808b7+2023-06-15-17h-12m-39s",
    "TRI+52ca9b6a+2024-01-03-15h-28m-44s",
    "REAL+4f8ca688+2023-08-29-15h-01m-27s",
    "AUTOLab+84bd5053+2023-07-21-13h-35m-45s",
    "TRI+52ca9b6a+2023-12-13-11h-20m-33s",
    "AUTOLab+5d05c5aa+2023-10-15-17h-56m-37s",
    "TRI+52ca9b6a+2023-11-07-16h-59m-24s",
    "AUTOLab+84bd5053+2023-08-18-12h-24m-35s",
    "REAL+4f8ca688+2023-07-20-14h-52m-36s",
    "RPL+32cba90c+2023-11-30-23h-02m-46s",
    "RAIL+d027f2ae+2023-09-20-18h-09m-55s",
    "ILIAD+5e938e3b+2023-07-19-20h-07m-31s",
    "AUTOLab+9aed6d7a+2023-12-02-17h-37m-39s",
    "IPRL+7790ec0a+2023-06-30-18h-55m-22s",
    "tri+7dfa2da3+2023-10-12-13h-24m-22s",
    "TRI+52ca9b6a+2024-01-25-14h-13m-07s",
    "TRI+938130c4+2023-08-10-15h-54m-54s",
    "GuptaLab+553d1bd5+2023-05-28-16h-54m-48s",
    "TRI+52ca9b6a+2024-01-09-14h-49m-56s",
]

# ---------------------------------------------------------------------------
# Module-level model cache (avoid reloading VGGT between episodes)
# ---------------------------------------------------------------------------
_MODEL_CACHE = {}


# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------

# Default values for optional config fields (merged into each config)
_DEFAULTS = {
    "backend": "yourdfpy",
    "chamfer": True,
    "robot_weight": 1.0,
    "track_weight": 0.0,
    "tracker": None,
    "grid_size": 30,
    "stage2_restarts": False,
    "lr": 0.001,
    "n_steps": 500,
    "chamfer_n_points": 2000,
    "stage4": False,
    "stage4_lr": 0.0001,
    "stage4_steps": 500,
    "stage4_robot_weight": 0.1,
}


def _cfg(desc, **overrides):
  """Build a config dict from defaults + overrides."""
  c = dict(_DEFAULTS)
  c["desc"] = desc
  c.update(overrides)
  return c


CONFIGS = {
    # ── Core ablations (Phase 1) ──
    "E0":  _cfg("Baseline: yourdfpy, Chamfer+Robot, no tracks"),
    "E1":  _cfg("PyBullet backend",
               backend="pybullet", stage2_restarts=True),
    "E2":  _cfg("No Chamfer (robot-only)", chamfer=False),
    "E3":  _cfg("No Robot depth (Chamfer-only)", robot_weight=0.0),
    "E4":  _cfg("Chamfer+Robot+Tracks (CoTracker, w=0.001)",
               track_weight=0.001, tracker="cotracker"),
    "E5":  _cfg("Chamfer+Robot+Tracks (TAPNext, w=0.001)",
               track_weight=0.001, tracker="tapnext"),

    # ── Track weight sweep (Phase 2) ──
    "E6":  _cfg("Tracks (CoTracker, w=0.0001)",
               track_weight=0.0001, tracker="cotracker"),
    "E7":  _cfg("Tracks (CoTracker, w=0.01)",
               track_weight=0.01, tracker="cotracker"),

    # ── Secondary ablations (Phase 3) ──
    "E8":  _cfg("Tracks (CoTracker, grid=15)",
               track_weight=0.001, tracker="cotracker", grid_size=15),
    "E9":  _cfg("Tracks (CoTracker, grid=50)",
               track_weight=0.001, tracker="cotracker", grid_size=50),
    "E10": _cfg("Stage 2 multi-restart", stage2_restarts=True),
    "E12": _cfg("PyBullet + Tracks (CoTracker, w=0.001)",
               backend="pybullet", stage2_restarts=True,
               track_weight=0.001, tracker="cotracker"),

    # ── Learning rate sweep (Phase 4) ──
    "E13": _cfg("lr=0.0003 (3x lower)", lr=0.0003),
    "E14": _cfg("lr=0.003  (3x higher)", lr=0.003),

    # ── Optimization steps sweep (Phase 5) ──
    "E15": _cfg("n_steps=200 (short)", n_steps=200),
    "E16": _cfg("n_steps=1000 (long)", n_steps=1000),

    # ── Stage 4 fine-tuning (Phase 6) ──
    "E17": _cfg("Baseline + Stage 4 fine-tune",
               stage4=True),
    "E18": _cfg("Tracks (CoTracker) + Stage 4 fine-tune",
               track_weight=0.001, tracker="cotracker",
               stage4=True),

    # ── Robot weight sweep (Phase 7) ──
    "E19": _cfg("robot_weight=0.1 (weak robot)", robot_weight=0.1),
    "E20": _cfg("robot_weight=10.0 (strong robot)", robot_weight=10.0),

    # ── Chamfer point count sweep (Phase 8) ──
    "E21": _cfg("chamfer_n_points=1000 (sparse)", chamfer_n_points=1000),
    "E22": _cfg("chamfer_n_points=4000 (dense)", chamfer_n_points=4000),

    # ── Eval grid_size sweep (Phase 9) ──
    "E23": _cfg("grid_size=15 (sparse eval)", grid_size=15),
    "E24": _cfg("grid_size=50 (dense eval)", grid_size=50),
}


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(episode_id, cfg, device, metadata, output_root,
               tracker_cache=None):
  """Run one experiment config on one episode. Returns metrics dict."""
  from compute_extrinsics import (
      init_extrinsics,
      run_stage2_alignment, run_global_joint_alignment, export_extrinsics,
      prepare_track_anchors, evaluate_extrinsics,
  )
  from core.io import load_depth_data
  from core.physics import PyBulletRenderer, TensorRobotRenderer

  extrinsics_db, depth_root = metadata
  config_id = cfg["_id"]

  print(f"\n{'='*70}")
  print(f"▶ {config_id} | {episode_id}")
  print(f"  {cfg['desc']}")
  print(f"{'='*70}")

  t0 = time.time()

  # ── Load depth data (from GCS mount or local disk) ──
  scene_constants = load_depth_data(
      episode_id, depth_root=depth_root, load_video="full")

  # ── Init extrinsics (Stage 0 + 1) ──
  scene_state, _MODEL_CACHE["vggt"] = init_extrinsics(
      scene_constants, extrinsics_db, device,
      vggt_models=_MODEL_CACHE.get("vggt"))

  # ── Stage 2: Per-camera alignment ──
  use_pybullet = (cfg["backend"] == "pybullet")
  pb_renderer = PyBulletRenderer()
  tensor_renderer = TensorRobotRenderer(device=device)

  if cfg["stage2_restarts"]:
    # Multi-restart Stage 2 uses PyBullet renderer (5 outer × 100 inner)
    from core.pybullet_extrinsics import run_stage2_alignment_pybullet
    scene_state = run_stage2_alignment_pybullet(
        scene_constants, pb_renderer, scene_state, device)
  else:
    # Single-sweep Stage 2 uses yourdfpy TensorRobotRenderer
    scene_state = run_stage2_alignment(
        scene_constants, tensor_renderer, scene_state)

  # ── 2D Tracking (if needed for optimization) ──
  track_anchors = None
  if cfg["track_weight"] > 0 and cfg["tracker"] is not None:
    from compute_2d_tracks import init_tracker, run_2d_tracking
    tracker_key = cfg["tracker"]
    if tracker_cache is None:
      tracker_cache = {}
    if tracker_key not in tracker_cache:
      tracker_cache[tracker_key] = init_tracker(tracker_key, device)
    tracker = tracker_cache[tracker_key]
    scene_constants = run_2d_tracking(
        tracker, scene_constants, device, grid_size=cfg["grid_size"])
    track_anchors = prepare_track_anchors(
        scene_constants, scene_state, pb_renderer, device)

  # ── Stage 3: Global joint optimization ──
  chamfer_w = 1.0 if cfg["chamfer"] else 0.0
  s3_lr = cfg.get("lr", 0.001)
  s3_steps = cfg.get("n_steps", 500)

  s3_cnp = cfg.get("chamfer_n_points", 2000)

  if use_pybullet:
    from core.pybullet_extrinsics import run_global_joint_alignment_pybullet
    scene_state = run_global_joint_alignment_pybullet(
        scene_constants, scene_state, pb_renderer, device,
        robot_weight=cfg["robot_weight"],
        lr=s3_lr, n_steps=s3_steps)
    if track_anchors and cfg["track_weight"] > 0:
      scene_state = run_global_joint_alignment(
          scene_constants, scene_state, tensor_renderer,
          chamfer_weight=chamfer_w,
          robot_weight=cfg["robot_weight"],
          track_anchors=track_anchors,
          track_weight=cfg["track_weight"],
          chamfer_n_points=s3_cnp,
          stage_name="Stage 3b (Track refine)")
  else:
    scene_state = run_global_joint_alignment(
        scene_constants, scene_state, tensor_renderer,
        lr=s3_lr, n_steps=s3_steps,
        chamfer_weight=chamfer_w,
        robot_weight=cfg["robot_weight"],
        track_anchors=track_anchors,
        track_weight=cfg["track_weight"],
        chamfer_n_points=s3_cnp)

  # ── Stage 4: Optional fine-tuning pass ──
  if cfg.get("stage4", False):
    s4_lr = cfg.get("stage4_lr", 0.0001)
    s4_steps = cfg.get("stage4_steps", 500)
    s4_rw = cfg.get("stage4_robot_weight", 0.1)
    scene_state = run_global_joint_alignment(
        scene_constants, scene_state, tensor_renderer,
        lr=s4_lr, n_steps=s4_steps,
        chamfer_weight=chamfer_w,
        robot_weight=s4_rw,
        track_anchors=track_anchors,
        track_weight=cfg["track_weight"],
        chamfer_n_points=s3_cnp,
        stage_name="Stage 4 (Fine-Tune)")

  # ── Evaluate: base metrics ──
  metrics = evaluate_extrinsics(
      scene_constants, scene_state, device,
      tensor_renderer=tensor_renderer)
  metrics["config_id"] = config_id
  metrics["episode_id"] = episode_id
  metrics["desc"] = cfg["desc"]
  metrics["elapsed_s"] = time.time() - t0

  # ── Evaluate: dual-tracker track reproj (always both) ──
  from compute_2d_tracks import init_tracker, run_2d_tracking
  if tracker_cache is None:
    tracker_cache = {}
  eval_grid = cfg.get("grid_size", 30)
  for trk_method in ("cotracker", "tapnext"):
    key = f"track_reproj_{trk_method}_px"
    try:
      if trk_method not in tracker_cache:
        tracker_cache[trk_method] = init_tracker(trk_method, device)
      sc_tmp = copy.deepcopy(scene_constants)
      sc_tmp = run_2d_tracking(
          tracker_cache[trk_method], sc_tmp, device, grid_size=eval_grid)
      anchors_tmp = prepare_track_anchors(
          sc_tmp, scene_state, pb_renderer, device)
      m_tmp = evaluate_extrinsics(
          sc_tmp, scene_state, device,
          tensor_renderer=tensor_renderer,
          track_anchors=anchors_tmp)
      metrics[key] = m_tmp.get("track_reproj_mean_px", float("nan"))
    except Exception as e:
      print(f"  ⚠️ {trk_method} eval skipped: {e}")
      metrics[key] = float("nan")

  # ── Save extrinsics + metrics ──
  exp_dir = os.path.join(output_root, config_id, episode_id)
  os.makedirs(exp_dir, exist_ok=True)

  export_extrinsics(scene_constants, scene_state,
                    export_root=os.path.join(exp_dir, "extrinsics"),
                    stage_suffix=config_id)

  with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2, default=str)

  def _fmt(v, fmt):
    return format(v, fmt) if isinstance(v, float) and not np.isnan(v) else "—"

  print(f"\n📊 [{config_id}] {episode_id}:")
  print(f"   Chamfer:  {_fmt(metrics.get('chamfer_total'), '.5f')}")
  print(f"   Robot:    C1={_fmt(metrics.get('robot_loss_cam1'), '.5f')}"
        f"  C2={_fmt(metrics.get('robot_loss_cam2'), '.5f')}"
        f"  W={_fmt(metrics.get('robot_loss_wrist'), '.5f')}")
  print(f"   BG Ovlp:  {_fmt(metrics.get('bg_overlap_pct'), '.1f')}%")
  print(f"   Track px: cotracker={_fmt(metrics.get('track_reproj_cotracker_px'), '.2f')}"
        f"  tapnext={_fmt(metrics.get('track_reproj_tapnext_px'), '.2f')}")
  print(f"   Time:     {metrics['elapsed_s']:.0f}s")

  # Cleanup GPU
  del pb_renderer
  del tensor_renderer
  gc.collect()
  torch.cuda.empty_cache()

  return metrics


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(all_results, output_root):
  """Write summary CSV with one row per (config, episode)."""
  csv_path = os.path.join(output_root, "summary.csv")
  if not all_results:
    return

  fieldnames = [
      "config_id", "episode_id", "desc",
      "chamfer_total", "chamfer_12", "chamfer_1w", "chamfer_2w",
      "robot_loss_cam1", "robot_loss_cam2", "robot_loss_wrist",
      "bg_overlap_pct",
      "track_reproj_cotracker_px", "track_reproj_tapnext_px",
      "track_reproj_mean_px",  # legacy compat
      "elapsed_s",
  ]

  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in all_results:
      writer.writerow(row)

  # ── Aggregate summary per config ──
  configs_seen = {}
  for r in all_results:
    cid = r["config_id"]
    configs_seen.setdefault(cid, []).append(r)

  def _agg(rows, key):
    vals = [r[key] for r in rows
            if not np.isnan(r.get(key, float("nan")))]
    return f"{np.mean(vals):.4f}" if vals else "—"

  def _agg1(rows, key):
    vals = [r[key] for r in rows
            if not np.isnan(r.get(key, float("nan")))]
    return f"{np.mean(vals):.1f}" if vals else "—"

  print(f"\n{'='*110}")
  print("📊 ABLATION SUMMARY (mean over episodes)")
  print(f"{'='*110}")
  header = (f"{'ID':<5} {'Description':<40} {'Chamfer':>8} "
            f"{'Robot':>8} {'BG%':>6} "
            f"{'CoTrk px':>9} {'TAP px':>9} {'N':>3} {'Time':>5}")
  print(header)
  print("─" * len(header))

  for cid, rows in configs_seen.items():
    desc = rows[0]["desc"][:38]
    # Combined robot = cam1 + cam2 (exclude wrist for comparability)
    rob_vals = [r.get("robot_loss_cam1", 0) + r.get("robot_loss_cam2", 0)
                for r in rows
                if not np.isnan(r.get("robot_loss_cam1", float("nan")))]
    rob_s = f"{np.mean(rob_vals):.4f}" if rob_vals else "—"
    time_vals = [r.get("elapsed_s", 0) for r in rows]
    time_s = f"{np.mean(time_vals):.0f}s" if time_vals else "—"

    print(f"{cid:<5} {desc:<40} "
          f"{_agg(rows, 'chamfer_total'):>8} "
          f"{rob_s:>8} "
          f"{_agg1(rows, 'bg_overlap_pct') + '%':>6} "
          f"{_agg1(rows, 'track_reproj_cotracker_px'):>9} "
          f"{_agg1(rows, 'track_reproj_tapnext_px'):>9} "
          f"{len(rows):>3} {time_s:>5}")

  print(f"\n💾 Full results: {csv_path}")


def collect_results_from_disk(output_root, config_ids, episodes):
  """Scan output_root for completed metrics.json files and aggregate."""
  all_results = []
  for cid in config_ids:
    for eid in episodes:
      result_path = os.path.join(output_root, cid, eid, "metrics.json")
      if os.path.exists(result_path):
        try:
          with open(result_path) as f:
            metrics = json.load(f)
          all_results.append(metrics)
        except json.JSONDecodeError:
          pass  # skip corrupt files
  return all_results


def main():
  parser = argparse.ArgumentParser(description="MV-TAP extrinsics ablation runner")
  parser.add_argument("--configs", type=str, default="E0,E4",
                      help="Comma-separated config IDs to run (e.g. E0,E1,E4). "
                           "'all' runs everything.")
  parser.add_argument("--episodes", type=int, default=10,
                      help="Number of episodes to run (sampled from default set)")
  parser.add_argument("--episode_id", type=str, default=None,
                      help="Single episode to run (overrides --episodes)")
  parser.add_argument("--output_root", type=str,
                      default=os.path.expanduser(
                          "~/droid_data/output/ablation"),
                      help="Output directory")
  parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for episode sampling")
  parser.add_argument("--meta_root", type=str, default=None,
                      help="Path to directory containing camera_serials.json, "
                           "episode_id_to_path.json, etc. "
                           "Auto-detected or downloaded if not specified.")

  # ── Distributed / Multi-GPU ──
  parser.add_argument("--rank", type=int, default=0,
                      help="Worker rank for distributed execution (0-indexed)")
  parser.add_argument("--world_size", type=int, default=1,
                      help="Total number of parallel workers")
  parser.add_argument("--summarize", action="store_true",
                      help="Only aggregate existing results from disk "
                           "(no computation, run after all workers finish)")
  parser.add_argument("--depth_root", type=str, default=None,
                      help="Root dir with Stage 1 depth output "
                           "(default: ~/droid_data/output/mv-tap/droid/depth)")
  args = parser.parse_args()

  # Parse config list
  if args.configs.lower() == "all":
    config_ids = list(CONFIGS.keys())
  else:
    config_ids = [c.strip() for c in args.configs.split(",")]
  for cid in config_ids:
    if cid not in CONFIGS:
      print(f"❌ Unknown config: {cid}. Available: {list(CONFIGS.keys())}")
      sys.exit(1)

  # Select episodes
  if args.episode_id:
    episodes = [args.episode_id]
  else:
    rng = random.Random(args.seed)
    episodes = rng.sample(DEFAULT_EPISODES,
                          min(args.episodes, len(DEFAULT_EPISODES)))

  os.makedirs(args.output_root, exist_ok=True)

  # ── Summarize-only mode ──
  if args.summarize:
    print(f"📊 Collecting results from {args.output_root}...")
    all_results = collect_results_from_disk(
        args.output_root, config_ids, episodes)
    write_summary(all_results, args.output_root)
    print(f"✅ Summary complete ({len(all_results)} results)")
    return

  # Load metadata — only extrinsics_db is needed (for init_extrinsics);
  # depth data is loaded directly from disk via load_depth_data().
  meta_root = args.meta_root
  if meta_root is None:
    candidates = [
        "/content/droid_raw/1.0.1",
        os.path.expanduser("~/droid_workspace/droid/metadata"),
        os.path.expanduser("~/droid_data/input/robotics/droid_raw/1.0.1"),
        os.path.expanduser("~/droid_data/meta/1.0.1"),
    ]
    for cand in candidates:
      if os.path.exists(os.path.join(cand, "cam2base_extrinsic_superset.json")):
        meta_root = cand
        break

  if meta_root is None:
    meta_root = os.path.expanduser("~/droid_workspace/droid/metadata")
    os.makedirs(meta_root, exist_ok=True)
    fname = "cam2base_extrinsic_superset.json"
    dest = os.path.join(meta_root, fname)
    if not os.path.exists(dest):
      HF_BASE = "https://huggingface.co/KarlP/droid/resolve/main"
      print(f"📥 Downloading {fname} from HuggingFace → {meta_root}")
      ret = os.system(f"wget -q -O '{dest}' '{HF_BASE}/{fname}'")
      if ret != 0:
        print(f"  ❌ Failed to download {fname}")
        sys.exit(1)

  print(f"📂 Metadata root: {meta_root}")

  with open(os.path.join(meta_root, "cam2base_extrinsic_superset.json")) as f:
    extrinsics_db = json.load(f)

  # Depth root (Stage 1 output, typically GCS-mounted)
  depth_root = args.depth_root
  if depth_root is None:
    depth_root_candidates = [
        os.path.expanduser("~/droid_data/output/mv-tap/droid/depth"),
        "/content/droid_depth_cache",
        os.path.expanduser("~/droid_workspace/droid/depth_cache"),
    ]
    depth_root = next(
        (p for p in depth_root_candidates if os.path.isdir(p)),
        depth_root_candidates[0])

  print(f"   Depth root: {depth_root}")

  metadata = (extrinsics_db, depth_root)

  # GPU: always use cuda:0 — GPU isolation via CUDA_VISIBLE_DEVICES (same
  # pattern as compute_extrinsics.py / run_parallel.sh)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  os.environ["PYOPENGL_PLATFORM"] = "egl"

  # ── Build work matrix and shard ──
  all_jobs = [(c, e) for c in config_ids for e in episodes]
  my_jobs = all_jobs[args.rank::args.world_size]

  print(f"🔬 Ablation Experiment Runner")
  print(f"   Configs: {config_ids}")
  print(f"   Episodes: {len(episodes)}")
  print(f"   Output: {args.output_root}")
  print(f"   Device: {device}")
  print(f"   Worker: {args.rank}/{args.world_size} "
        f"({len(my_jobs)}/{len(all_jobs)} jobs)")

  # Run assigned jobs
  my_results = []
  tracker_cache = {}

  for i, (cid, eid) in enumerate(my_jobs):
    cfg = copy.deepcopy(CONFIGS[cid])
    cfg["_id"] = cid

    # Skip if already completed successfully (re-run errors)
    result_path = os.path.join(args.output_root, cid, eid, "metrics.json")
    if os.path.exists(result_path):
      with open(result_path) as f:
        prev_metrics = json.load(f)
      if "error" not in prev_metrics:
        print(f"\n⏭️  [{i+1}/{len(my_jobs)}] {cid} | {eid} — already done, "
              f"loading")
        my_results.append(prev_metrics)
        continue
      else:
        print(f"\n🔄 [{i+1}/{len(my_jobs)}] {cid} | {eid} — "
              f"previous run failed, retrying...")

    print(f"\n🚀 [{i+1}/{len(my_jobs)}] {cid} | {eid} "
          f"(worker {args.rank})")
    try:
      metrics = run_single(
          eid, cfg, device, metadata, args.output_root,
          tracker_cache=tracker_cache)
      my_results.append(metrics)
    except Exception as e:
      print(f"\n❌ FAILED: {cid} | {eid}")
      traceback.print_exc()
      err_metrics = {
          "config_id": cid, "episode_id": eid,
          "desc": cfg["desc"], "error": str(e),
          "chamfer_total": float("nan"),
          "robot_loss_cam1": float("nan"),
          "robot_loss_cam2": float("nan"),
          "robot_loss_wrist": float("nan"),
          "bg_overlap_pct": float("nan"),
          "track_reproj_mean_px": float("nan"),
          "track_reproj_cotracker_px": float("nan"),
          "track_reproj_tapnext_px": float("nan"),
          "elapsed_s": 0,
      }
      # Write error metrics to disk so summarize can pick it up
      err_dir = os.path.join(args.output_root, cid, eid)
      os.makedirs(err_dir, exist_ok=True)
      with open(os.path.join(err_dir, "metrics.json"), "w") as f:
        json.dump(err_metrics, f, indent=2, default=str)
      my_results.append(err_metrics)

  # Worker-local summary
  print(f"\n✅ Worker {args.rank} complete: "
        f"{len(my_results)}/{len(my_jobs)} jobs")

  # If single-worker mode, write full summary immediately
  if args.world_size == 1:
    all_results = collect_results_from_disk(
        args.output_root, config_ids, episodes)
    write_summary(all_results, args.output_root)
  else:
    print(f"💡 Run with --summarize after all workers finish to aggregate:")
    print(f"   python run_extrinsics_ablation.py --configs {args.configs} "
          f"--episodes {args.episodes} --output_root {args.output_root} "
          f"--summarize")


if __name__ == "__main__":
  main()
