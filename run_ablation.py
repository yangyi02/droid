#!/usr/bin/env python3
"""Ablation experiment runner for MV-TAP extrinsics calibration.

Iterates over experiment configs × episodes, runs the extrinsics calibration
pipeline with different settings, and saves structured metric results.

Usage (from repo root on Colab or any machine with GPU):
  python run_ablation.py --episodes 10 --configs E0,E4
  python run_ablation.py --episode_id "ILIAD+5e938e3b+2023-07-20-11h-50m-51s" --configs E0
  python run_ablation.py --all  # run everything

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
# Metric collection
# ---------------------------------------------------------------------------

def evaluate_extrinsics(scene_constants, scene_state, device,
                        track_anchors=None):
  """Compute extrinsics quality metrics without re-running optimization.

  Returns a dict with:
    chamfer_12, chamfer_1w, chamfer_2w: pairwise Chamfer distances
    robot_loss_cam1, robot_loss_cam2, robot_loss_wrist
    bg_overlap_pct: average BG overlap %
    per_cam_shift_mm, per_cam_rot_deg: extrinsic magnitude from origin
    track_model_err_px: mean FK-based reprojection error (if anchors given)
  """
  from compute_extrinsics import (
      extract_robot_physical_tensors, get_cam_points_local_t,
      batched_chamfer_distance, compute_robot_loss,
  )


  wrist_cam = scene_constants["meta"]["wrist_serial"]
  ext_cams = [c for c in scene_constants["camera"].keys() if c != wrist_cam]
  cam1, cam2 = ext_cams[0], ext_cams[1]
  n_frames = len(scene_constants["robot"]["joint_positions"])
  T_ee_all = scene_constants["robot"]["T_ee_base_all"]

  # Lazily create tensor_renderer for evaluation only
  from core.physics import TensorRobotRenderer
  eval_renderer = TensorRobotRenderer(device=device)

  metrics = {}

  # --- Robot depth losses ---
  for cam_id, key_prefix in [(cam1, "cam1"), (cam2, "cam2"), (wrist_cam, "wrist")]:
    try:
      batch_X, batch_obs = extract_robot_physical_tensors(
          cam_id, scene_constants, eval_renderer)
      T_opt = torch.tensor(
          scene_state[cam_id]["base_extrinsic"],
          dtype=torch.float32, device=device)
      K_t = torch.tensor(
          scene_constants["camera"][cam_id]["K_mat"],
          dtype=torch.float32, device=device)
      tol = float("inf") if cam_id == wrist_cam else 0.15
      loss = compute_robot_loss(batch_X, T_opt, K_t, batch_obs,
                                depth_tolerance=tol)
      metrics[f"robot_loss_{key_prefix}"] = loss.item()
    except Exception as e:
      metrics[f"robot_loss_{key_prefix}"] = float("nan")
      print(f"    ⚠️ Robot loss failed for {cam_id}: {e}")

  # --- Chamfer losses ---
  try:
    cache_Pc1, cache_Pc2, cache_Pcw, cache_Tee = [], [], [], []
    for t in range(n_frames):
      pc1 = get_cam_points_local_t(t, scene_constants["camera"][cam1], device)
      pc2 = get_cam_points_local_t(t, scene_constants["camera"][cam2], device)
      pcw = get_cam_points_local_t(t, scene_constants["camera"][wrist_cam], device)
      if pc1 is not None and pc2 is not None and pcw is not None:
        cache_Pc1.append(pc1)
        cache_Pc2.append(pc2)
        cache_Pcw.append(pcw)
        cache_Tee.append(torch.tensor(T_ee_all[t], dtype=torch.float32, device=device))

    batch_Pc1 = torch.stack(cache_Pc1)
    batch_Pc2 = torch.stack(cache_Pc2)
    batch_Pcw = torch.stack(cache_Pcw)
    batch_Tee = torch.stack(cache_Tee)

    T1 = torch.tensor(scene_state[cam1]["base_extrinsic"], dtype=torch.float32, device=device)
    T2 = torch.tensor(scene_state[cam2]["base_extrinsic"], dtype=torch.float32, device=device)
    Tw = torch.tensor(scene_state[wrist_cam]["base_extrinsic"], dtype=torch.float32, device=device)

    bc1 = (T1 @ batch_Pc1)[:, :3, :].transpose(1, 2)
    bc2 = (T2 @ batch_Pc2)[:, :3, :].transpose(1, 2)
    bcw = torch.bmm(batch_Tee @ Tw, batch_Pcw)[:, :3, :].transpose(1, 2)

    l12, o12 = batched_chamfer_distance(bc1, bc2, device)
    l1w, o1w = batched_chamfer_distance(bc1, bcw, device)
    l2w, o2w = batched_chamfer_distance(bc2, bcw, device)

    metrics["chamfer_12"] = l12.item()
    metrics["chamfer_1w"] = l1w.item()
    metrics["chamfer_2w"] = l2w.item()
    metrics["chamfer_total"] = (l12 + l1w + l2w).item()
    metrics["bg_overlap_pct"] = (o12 + o1w + o2w).item() / 3.0 * 100
  except Exception as e:
    metrics["chamfer_total"] = float("nan")
    metrics["bg_overlap_pct"] = float("nan")
    print(f"    ⚠️ Chamfer eval failed: {e}")

  # --- Track model error (if anchors provided) ---
  if track_anchors:
    from compute_extrinsics import compute_track_reproj_loss
    T_ee_t = torch.tensor(T_ee_all, dtype=torch.float32, device=device)
    total_err = 0.0
    n_cam = 0
    for cam_id in track_anchors:
      T_opt = torch.tensor(
          scene_state[cam_id]["base_extrinsic"],
          dtype=torch.float32, device=device)
      K_t = torch.tensor(
          scene_constants["camera"][cam_id]["K_mat"],
          dtype=torch.float32, device=device)
      scheme = track_anchors[cam_id]["scheme"]
      l = compute_track_reproj_loss(
          track_anchors[cam_id], T_opt, K_t, T_ee_t, scheme, device)
      total_err += l.item()
      n_cam += 1
    metrics["track_reproj_mean_px"] = total_err / max(n_cam, 1)
  else:
    metrics["track_reproj_mean_px"] = float("nan")

  # --- Extrinsic magnitude from identity ---
  for cam_id in scene_constants["camera"]:
    T = scene_state[cam_id]["base_extrinsic"]
    metrics[f"shift_mm_{cam_id}"] = float(np.linalg.norm(T[:3, 3]) * 1000)

  # Cleanup
  del eval_renderer
  torch.cuda.empty_cache()

  return metrics


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(episode_id, cfg, device, metadata, output_root,
               tracker_cache=None):
  """Run one experiment config on one episode. Returns metrics dict."""
  import mediapy as media
  from compute_depth import init_episode
  from compute_extrinsics import (
      init_calibration_models, init_camera_states, vggt_warmup_extrinsics,
      run_stage2_alignment, run_global_joint_alignment, export_extrinsics,
      prepare_track_anchors,
  )
  from core.physics import PyBulletRenderer, TensorRobotRenderer

  id_to_path, serials_db, keep_ranges, extrinsics_db = metadata
  config_id = cfg["_id"]

  print(f"\n{'='*70}")
  print(f"▶ {config_id} | {episode_id}")
  print(f"  {cfg['desc']}")
  print(f"{'='*70}")

  t0 = time.time()

  # ── Stage 0-1: Init ──
  scene_constants = init_episode(
      episode_id,
      os.path.expanduser("~/droid_data/input/robotics/droid_raw/1.0.1"),
      id_to_path, serials_db, keep_ranges)

  # Load depth from GCS cache
  local_cache = f"/content/droid_depth_cache/{episode_id}"
  if not os.path.exists(local_cache):
    gcs_depth = "gs://dm-tapnet/mv-tap/droid/depth"
    os.makedirs(local_cache, exist_ok=True)
    os.system(f"gsutil -m rsync -r '{gcs_depth}/{episode_id}' '{local_cache}/' "
              "> /dev/null 2>&1")

  # Load robot data
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

  # Load per-camera depth + video + calibration
  wrist_serial = scene_constants["meta"].get("wrist_serial")
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

  # ── Init extrinsics (Stage 0) ──
  scene_state = init_camera_states(scene_constants, extrinsics_db)
  all_ext = all(s["extrinsics"] is not None for s in scene_state.values())

  if not all_ext:
    # Need VGGT — lazy-load
    if not hasattr(run_single, "_vggt"):
      vggt_model, load_fn, pose_fn = init_calibration_models()
      run_single._vggt = (vggt_model, load_fn, pose_fn)
    vggt_model, load_fn, pose_fn = run_single._vggt
    scene_state = vggt_warmup_extrinsics(
        scene_constants, vggt_model, load_fn, pose_fn, device)

  # ── Stage 2: Per-camera alignment ──
  use_pybullet = (cfg["backend"] == "pybullet")
  pb_renderer = PyBulletRenderer()
  tensor_renderer = TensorRobotRenderer(device=device)

  if use_pybullet:
    from core.pybullet_extrinsics import run_stage2_alignment_pybullet
    scene_state = run_stage2_alignment_pybullet(
        scene_constants, pb_renderer, scene_state, device)
  else:
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

  print(f"\n📊 [{config_id}] {episode_id}:")
  print(f"   Chamfer: {metrics.get('chamfer_total', 'N/A'):.5f}")
  print(f"   Robot C1: {metrics.get('robot_loss_cam1', 'N/A'):.5f} | "
        f"C2: {metrics.get('robot_loss_cam2', 'N/A'):.5f} | "
        f"W: {metrics.get('robot_loss_wrist', 'N/A'):.5f}")
  print(f"   BG Overlap: {metrics.get('bg_overlap_pct', 'N/A'):.1f}%")
  if not np.isnan(metrics.get("track_reproj_mean_px", float("nan"))):
    print(f"   Track Reproj: {metrics['track_reproj_mean_px']:.2f}px")
  print(f"   Time: {metrics['elapsed_s']:.0f}s")

  # Cleanup GPU
  del pb_renderer
  if "tensor_renderer" in dir():
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

  # Load metadata
  root_path = os.path.expanduser(
      "~/droid_data/input/robotics/droid_raw/1.0.1")
  # Fallback to Colab paths
  for alt in ["/content/droid_raw/1.0.1", root_path]:
    if os.path.exists(os.path.join(alt, "episode_id_to_path.json")):
      root_path = alt
      break

  def load_json(name):
    with open(os.path.join(root_path, name)) as f:
      return json.load(f)

  id_to_path = load_json("episode_id_to_path.json")
  serials_db = load_json("camera_serials.json")
  keep_ranges = load_json("keep_ranges_1_0_1.json")
  extrinsics_db = load_json("cam2base_extrinsic_superset.json")
  metadata = (id_to_path, serials_db, keep_ranges, extrinsics_db)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  os.makedirs(args.output_root, exist_ok=True)
  os.environ["PYOPENGL_PLATFORM"] = "egl"

  print(f"🔬 Ablation Experiment Runner")
  print(f"   Configs: {config_ids}")
  print(f"   Episodes: {len(episodes)}")
  print(f"   Output: {args.output_root}")
  print(f"   Device: {device}")
  print(f"   Total runs: {len(config_ids) * len(episodes)}")

  # Run matrix
  all_results = []
  tracker_cache = {}
  n_total = len(config_ids) * len(episodes)

  for i, (cid, eid) in enumerate(
      [(c, e) for c in config_ids for e in episodes]):
    cfg = copy.deepcopy(CONFIGS[cid])
    cfg["_id"] = cid

    # Skip if already completed
    result_path = os.path.join(args.output_root, cid, eid, "metrics.json")
    if os.path.exists(result_path):
      print(f"\n⏭️  [{i+1}/{n_total}] {cid} | {eid} — already done, loading")
      with open(result_path) as f:
        metrics = json.load(f)
      all_results.append(metrics)
      continue

    print(f"\n🚀 [{i+1}/{n_total}] Starting...")
    try:
      metrics = run_single(
          eid, cfg, device, metadata, args.output_root,
          tracker_cache=tracker_cache)
      all_results.append(metrics)
    except Exception as e:
      print(f"\n❌ FAILED: {cid} | {eid}")
      traceback.print_exc()
      all_results.append({
          "config_id": cid, "episode_id": eid,
          "desc": cfg["desc"], "error": str(e),
          "chamfer_total": float("nan"),
          "robot_loss_cam1": float("nan"),
          "robot_loss_cam2": float("nan"),
          "robot_loss_wrist": float("nan"),
          "bg_overlap_pct": float("nan"),
          "track_reproj_mean_px": float("nan"),
          "elapsed_s": 0,
      })

    # Write incremental summary after each run
    write_summary(all_results, args.output_root)

  # Final summary
  write_summary(all_results, args.output_root)
  print(f"\n✅ All {n_total} runs complete!")


if __name__ == "__main__":
  main()
