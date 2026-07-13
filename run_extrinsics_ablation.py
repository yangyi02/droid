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
]

# ---------------------------------------------------------------------------
# Module-level model cache (avoid reloading VGGT between episodes)
# ---------------------------------------------------------------------------
_MODEL_CACHE = {}


# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------

CONFIGS = {
    # ── Core ablations (Phase 1) ──
    "E0": {
        "desc": "Baseline: yourdfpy, Chamfer+Robot, no tracks",
        "backend": "yourdfpy",
        "chamfer": True,
        "robot_weight": 1.0,
        "track_weight": 0.0,
        "tracker": None,
        "grid_size": 30,
        "stage2_restarts": False,
    },
    "E1": {
        "desc": "PyBullet backend (instead of yourdfpy)",
        "backend": "pybullet",
        "chamfer": True,
        "robot_weight": 1.0,
        "track_weight": 0.0,
        "tracker": None,
        "grid_size": 30,
        "stage2_restarts": True,
    },
    "E2": {
        "desc": "No Chamfer loss (robot-only)",
        "backend": "yourdfpy",
        "chamfer": False,
        "robot_weight": 1.0,
        "track_weight": 0.0,
        "tracker": None,
        "grid_size": 30,
        "stage2_restarts": False,
    },
    "E3": {
        "desc": "No Robot depth loss (Chamfer-only)",
        "backend": "yourdfpy",
        "chamfer": True,
        "robot_weight": 0.0,
        "track_weight": 0.0,
        "tracker": None,
        "grid_size": 30,
        "stage2_restarts": False,
    },
    "E4": {
        "desc": "Chamfer+Robot+Tracks (CoTracker, w=0.001)",
        "backend": "yourdfpy",
        "chamfer": True,
        "robot_weight": 1.0,
        "track_weight": 0.001,
        "tracker": "cotracker",
        "grid_size": 30,
        "stage2_restarts": False,
    },
    "E5": {
        "desc": "Chamfer+Robot+Tracks (TAPNext++, w=0.001)",
        "backend": "yourdfpy",
        "chamfer": True,
        "robot_weight": 1.0,
        "track_weight": 0.001,
        "tracker": "tapnext",
        "grid_size": 30,
        "stage2_restarts": False,
    },

    # ── Track weight sweep (Phase 2) ──
    "E6": {
        "desc": "Tracks (CoTracker, low weight=0.0001)",
        "backend": "yourdfpy",
        "chamfer": True,
        "robot_weight": 1.0,
        "track_weight": 0.0001,
        "tracker": "cotracker",
        "grid_size": 30,
        "stage2_restarts": False,
    },
    "E7": {
        "desc": "Tracks (CoTracker, high weight=0.01)",
        "backend": "yourdfpy",
        "chamfer": True,
        "robot_weight": 1.0,
        "track_weight": 0.01,
        "tracker": "cotracker",
        "grid_size": 30,
        "stage2_restarts": False,
    },

    # ── Secondary ablations (Phase 3) ──
    "E8": {
        "desc": "Tracks (CoTracker, sparse grid_size=15)",
        "backend": "yourdfpy",
        "chamfer": True,
        "robot_weight": 1.0,
        "track_weight": 0.001,
        "tracker": "cotracker",
        "grid_size": 15,
        "stage2_restarts": False,
    },
    "E9": {
        "desc": "Tracks (CoTracker, dense grid_size=50)",
        "backend": "yourdfpy",
        "chamfer": True,
        "robot_weight": 1.0,
        "track_weight": 0.001,
        "tracker": "cotracker",
        "grid_size": 50,
        "stage2_restarts": False,
    },
    "E10": {
        "desc": "Stage 2 multi-restart (yourdfpy, no tracks)",
        "backend": "yourdfpy",
        "chamfer": True,
        "robot_weight": 1.0,
        "track_weight": 0.0,
        "tracker": None,
        "grid_size": 30,
        "stage2_restarts": True,
    },
    "E12": {
        "desc": "PyBullet + Tracks (CoTracker, w=0.001)",
        "backend": "pybullet",
        "chamfer": True,
        "robot_weight": 1.0,
        "track_weight": 0.001,
        "tracker": "cotracker",
        "grid_size": 30,
        "stage2_restarts": True,
    },
}


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(episode_id, cfg, device, metadata, output_root,
               tracker_cache=None):
  """Run one experiment config on one episode. Returns metrics dict."""
  import mediapy as media
  from compute_depth import init_episode
  from compute_extrinsics import (
      init_extrinsics,
      run_stage2_alignment, run_global_joint_alignment, export_extrinsics,
      prepare_track_anchors, evaluate_extrinsics,
  )
  from core.physics import PyBulletRenderer, TensorRobotRenderer

  id_to_path, serials_db, keep_ranges, extrinsics_db, raw_data_root, depth_cache_root = metadata
  config_id = cfg["_id"]

  print(f"\n{'='*70}")
  print(f"▶ {config_id} | {episode_id}")
  print(f"  {cfg['desc']}")
  print(f"{'='*70}")

  t0 = time.time()

  # ── Stage 0-1: Init ──
  scene_constants = init_episode(
      episode_id, raw_data_root,
      id_to_path, serials_db, keep_ranges)

  # Load depth from GCS cache
  local_cache = os.path.join(depth_cache_root, episode_id)
  if not os.path.exists(local_cache):
    gcs_depth = "gs://dm-tapnet/mv-tap/droid/depth"
    os.makedirs(local_cache, exist_ok=True)
    os.system(f"gsutil -m rsync -r '{gcs_depth}/{episode_id}' '{local_cache}/' "
              "> /dev/null 2>&1")

  # Load robot data from cache
  robot_path = os.path.join(local_cache, "robot.npz")
  if os.path.exists(robot_path):
    robot_data = np.load(robot_path, allow_pickle=True)
    for k in ["joint_positions", "gripper_positions",
              "T_cam_ee_init", "T_ee_base_all"]:
      if k in robot_data:
        scene_constants["robot"][k] = robot_data[k]
    if "wrist_serial" in robot_data:
      scene_constants["meta"]["wrist_serial"] = str(
          robot_data["wrist_serial"].item())

  # Fallback: if robot.npz is missing or incomplete (old cache format),
  # compute kinematics from the raw H5 trajectory file
  if "T_ee_base_all" not in scene_constants["robot"]:
    ep_path = scene_constants["meta"]["episode_path"]
    h5_path = os.path.join(ep_path, "trajectory.h5")
    if os.path.exists(h5_path):
      from compute_depth import parse_robot_kinematics
      print(f"  ⚠️ T_ee_base_all missing from cache, "
            f"computing from H5...")
      parse_robot_kinematics(scene_constants)
    else:
      raise RuntimeError(
          f"Robot kinematics unavailable: no T_ee_base_all in "
          f"robot.npz and no trajectory.h5 at {h5_path}")

  # Load per-camera depth + video + calibration
  for cam_id in scene_constants["camera"]:
    cam_dir = os.path.join(local_cache, cam_id)
    vid_path = os.path.join(cam_dir, "video_left.mp4")
    if os.path.exists(vid_path):
      scene_constants["camera"][cam_id]["video_rgb"] = media.read_video(vid_path)
    depth_path = os.path.join(cam_dir, "raw_depth.npz")
    if os.path.exists(depth_path):
      scene_constants["camera"][cam_id]["raw_depth"] = (
          np.load(depth_path)["depth"].astype(np.float32) / 1000.0)
    calib_path = os.path.join(cam_dir, "calibration.npz")
    if os.path.exists(calib_path):
      c = np.load(calib_path)
      scene_constants["camera"][cam_id]["K_mat"] = c["K_calib_left"]
      scene_constants["camera"][cam_id]["baseline"] = float(c["baseline"])

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

  # ── 2D Tracking (if needed) ──
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

  if use_pybullet:
    from core.pybullet_extrinsics import run_global_joint_alignment_pybullet
    # PyBullet Stage 3 (Chamfer + Robot); then yourdfpy pass adds tracks
    scene_state = run_global_joint_alignment_pybullet(
        scene_constants, scene_state, pb_renderer, device,
        robot_weight=cfg["robot_weight"])
    # Second pass with tracks if requested (pybullet doesn't support tracks)
    if track_anchors and cfg["track_weight"] > 0:
      scene_state = run_global_joint_alignment(
          scene_constants, scene_state, tensor_renderer,
          chamfer_weight=chamfer_w,
          robot_weight=cfg["robot_weight"],
          track_anchors=track_anchors,
          track_weight=cfg["track_weight"],
          stage_name="Stage 3b (Track refine)")
  else:
    # yourdfpy path — supports all options natively
    scene_state = run_global_joint_alignment(
        scene_constants, scene_state, tensor_renderer,
        chamfer_weight=chamfer_w,
        robot_weight=cfg["robot_weight"],
        track_anchors=track_anchors,
        track_weight=cfg["track_weight"])

  # ── Evaluate ──
  metrics = evaluate_extrinsics(
      scene_constants, scene_state, device,
      tensor_renderer=tensor_renderer,
      track_anchors=track_anchors)
  metrics["config_id"] = config_id
  metrics["episode_id"] = episode_id
  metrics["desc"] = cfg["desc"]
  metrics["elapsed_s"] = time.time() - t0

  # ── Save extrinsics + metrics ──
  exp_dir = os.path.join(output_root, config_id, episode_id)
  os.makedirs(exp_dir, exist_ok=True)

  export_extrinsics(scene_constants, scene_state,
                    export_root=os.path.join(exp_dir, "extrinsics"),
                    stage_suffix=config_id)

  with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2, default=str)

  def _fmt(v, fmt):
    return format(v, fmt) if isinstance(v, float) and not np.isnan(v) else str(v)

  print(f"\n📊 [{config_id}] {episode_id}:")
  print(f"   Chamfer: {_fmt(metrics.get('chamfer_total', float('nan')), '.5f')}")
  print(f"   Robot C1: {_fmt(metrics.get('robot_loss_cam1', float('nan')), '.5f')} | "
        f"C2: {_fmt(metrics.get('robot_loss_cam2', float('nan')), '.5f')} | "
        f"W: {_fmt(metrics.get('robot_loss_wrist', float('nan')), '.5f')}")
  print(f"   BG Overlap: {_fmt(metrics.get('bg_overlap_pct', float('nan')), '.1f')}%")
  trk = metrics.get("track_reproj_mean_px", float("nan"))
  if isinstance(trk, float) and not np.isnan(trk):
    print(f"   Track Reproj: {trk:.2f}px")
  print(f"   Time: {metrics['elapsed_s']:.0f}s")

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
      "bg_overlap_pct", "track_reproj_mean_px", "elapsed_s",
  ]

  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in all_results:
      writer.writerow(row)

  # Print aggregate summary per config
  print(f"\n{'='*80}")
  print("📊 ABLATION SUMMARY")
  print(f"{'='*80}")

  configs_seen = {}
  for r in all_results:
    cid = r["config_id"]
    if cid not in configs_seen:
      configs_seen[cid] = []
    configs_seen[cid].append(r)

  header = (f"{'Config':<8} {'Desc':<45} {'Chamfer':>10} {'Robot':>10} "
            f"{'BG Ovlp':>8} {'Track':>8} {'N':>3}")
  print(header)
  print("-" * len(header))

  for cid, rows in configs_seen.items():
    desc = rows[0]["desc"][:43]
    chamf_vals = [r["chamfer_total"] for r in rows
                  if not np.isnan(r.get("chamfer_total", float("nan")))]
    rob_vals = [r["robot_loss_cam1"] + r["robot_loss_cam2"]
                for r in rows
                if not np.isnan(r.get("robot_loss_cam1", float("nan")))]
    bg_vals = [r["bg_overlap_pct"] for r in rows
               if not np.isnan(r.get("bg_overlap_pct", float("nan")))]
    trk_vals = [r["track_reproj_mean_px"] for r in rows
                if not np.isnan(r.get("track_reproj_mean_px", float("nan")))]

    chamf_s = f"{np.mean(chamf_vals):.5f}" if chamf_vals else "N/A"
    rob_s = f"{np.mean(rob_vals):.5f}" if rob_vals else "N/A"
    bg_s = f"{np.mean(bg_vals):.1f}%" if bg_vals else "N/A"
    trk_s = f"{np.mean(trk_vals):.1f}px" if trk_vals else "—"

    print(f"{cid:<8} {desc:<45} {chamf_s:>10} {rob_s:>10} "
          f"{bg_s:>8} {trk_s:>8} {len(rows):>3}")

  print(f"\n💾 Results saved to: {csv_path}")


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

  # Load metadata JSONs (downloaded from HuggingFace on first use)
  META_FILES = [
      "camera_serials.json",
      "episode_id_to_path.json",
      "keep_ranges_1_0_1.json",
      "cam2base_extrinsic_superset.json",
  ]
  HF_BASE = "https://huggingface.co/KarlP/droid/resolve/main"

  # Search candidate paths; use the first one that has the metadata files
  meta_root = args.meta_root
  if meta_root is None:
    candidates = [
        "/content/droid_raw/1.0.1",          # Colab default
        os.path.expanduser("~/droid_workspace/droid/metadata"),  # GPU server
        os.path.expanduser("~/droid_data/input/robotics/droid_raw/1.0.1"),
    ]
    for cand in candidates:
      if os.path.exists(os.path.join(cand, "episode_id_to_path.json")):
        meta_root = cand
        break

  if meta_root is None:
    # Auto-download from HuggingFace into ~/droid_workspace/droid/metadata
    meta_root = os.path.expanduser("~/droid_workspace/droid/metadata")
    os.makedirs(meta_root, exist_ok=True)
    print(f"📥 Downloading metadata JSONs from HuggingFace → {meta_root}")
    for fname in META_FILES:
      dest = os.path.join(meta_root, fname)
      if not os.path.exists(dest):
        ret = os.system(f"wget -q -O '{dest}' '{HF_BASE}/{fname}'")
        if ret != 0:
          print(f"  ❌ Failed to download {fname}")
          sys.exit(1)
        print(f"  ✅ {fname}")
      else:
        print(f"  ⏭️  {fname} (cached)")

  print(f"📂 Metadata root: {meta_root}")

  def load_json(name):
    with open(os.path.join(meta_root, name)) as f:
      return json.load(f)

  id_to_path = load_json("episode_id_to_path.json")
  serials_db = load_json("camera_serials.json")
  keep_ranges = load_json("keep_ranges_1_0_1.json")
  extrinsics_db = load_json("cam2base_extrinsic_superset.json")

  # Raw episode data root (SVO files, etc.)
  raw_data_root_candidates = [
      os.path.expanduser("~/droid_data/input/robotics/droid_raw/1.0.1"),
      "/content/droid_raw/1.0.1",
      os.path.expanduser("~/droid_workspace/data/droid_raw/1.0.1"),
  ]
  raw_data_root = next(
      (p for p in raw_data_root_candidates if os.path.isdir(p)),
      raw_data_root_candidates[0])  # fallback (may not exist — OK for depth-only)

  # Depth cache root (pre-computed depth NPZs downloaded from GCS)
  depth_cache_candidates = [
      os.path.expanduser("~/droid_data/depth_cache"),
      "/content/droid_depth_cache",
      os.path.expanduser("~/droid_workspace/droid/depth_cache"),
  ]
  depth_cache_root = next(
      (p for p in depth_cache_candidates if os.path.isdir(p)),
      depth_cache_candidates[0])  # fallback (will be created on first use)

  print(f"   Raw data: {raw_data_root}")
  print(f"   Depth cache: {depth_cache_root}")

  metadata = (id_to_path, serials_db, keep_ranges, extrinsics_db,
              raw_data_root, depth_cache_root)

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

    # Skip if already completed (enables restarts and deduplication)
    result_path = os.path.join(args.output_root, cid, eid, "metrics.json")
    if os.path.exists(result_path):
      print(f"\n⏭️  [{i+1}/{len(my_jobs)}] {cid} | {eid} — already done, "
            f"loading")
      with open(result_path) as f:
        metrics = json.load(f)
      my_results.append(metrics)
      continue

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
