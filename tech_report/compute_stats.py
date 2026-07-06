#!/usr/bin/env python3
"""Compute dataset statistics and evaluation metrics for the DROID tech report.

Usage (from the droid/ directory):
  python tech_report/compute_stats.py                          # default paths
  python tech_report/compute_stats.py --tracks_root /path/to   # custom paths
  python tech_report/compute_stats.py --max_episodes 200       # quick subset
  python tech_report/compute_stats.py --workers 16             # parallelism

Outputs:
  tech_report/stats_output/
    dataset_summary.json        — aggregate statistics
    per_episode_stats.csv       — per-episode breakdown
    failure_analysis.json       — failure analysis
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
from tqdm import tqdm


# ============================================================================
# 1. Pipeline Coverage
# ============================================================================

def count_episodes_per_stage(depth_root, extrinsics_root, tracks_root):
  """Count how many episodes completed each stage."""
  def list_episode_dirs(root):
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.exists(root):
      print(f"  ⚠️ Directory not found: {root}")
      return []
    return sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    ])

  depth_eps = set(list_episode_dirs(depth_root))
  ext_eps = set(list_episode_dirs(extrinsics_root))
  track_eps = set(list_episode_dirs(tracks_root))

  all_eps = depth_eps | ext_eps | track_eps

  coverage = {
      "total_unique_episodes": len(all_eps),
      "stage1_depth_completed": len(depth_eps),
      "stage2_extrinsics_completed": len(ext_eps),
      "stage3_tracks_completed": len(track_eps),
      "all_3_stages_completed": len(depth_eps & ext_eps & track_eps),
      "depth_only": len(depth_eps - ext_eps),
      "depth_and_extrinsics_only": len((depth_eps & ext_eps) - track_eps),
      "failed_at_extrinsics": len(depth_eps - ext_eps),
      "failed_at_tracks": len((depth_eps & ext_eps) - track_eps),
  }

  failed_ext = sorted(depth_eps - ext_eps)
  failed_tracks = sorted((depth_eps & ext_eps) - track_eps)

  return coverage, {
      "failed_at_extrinsics": failed_ext,
      "failed_at_tracks": failed_tracks,
      "all_completed": sorted(depth_eps & ext_eps & track_eps),
  }


# ============================================================================
# 2. Per-Episode: All Stats in One Pass (merged I/O + vectorized reproj)
# ============================================================================

def _vectorized_reprojection_error(traj_3d, traj_2d, vis_2d, intrinsics,
                                   extrinsics_w2c):
  """Compute reprojection error with fully vectorized batch matmul.

  Instead of looping over T frames, we broadcast:
    pts_cam = extrinsics_w2c @ pts_homo  for all frames at once.

  Args:
    traj_3d: (T, N, 3)
    traj_2d: (T, N, 2)
    vis_2d: (T, N) bool
    intrinsics: (4,) [fx, fy, cx, cy]
    extrinsics_w2c: (T, 4, 4)

  Returns:
    1-D array of per-measurement reprojection errors, or empty array.
  """
  T, N, _ = traj_3d.shape
  fx, fy, cx, cy = intrinsics

  # Build homogeneous coords: (T, N, 4)
  ones = np.ones((T, N, 1), dtype=traj_3d.dtype)
  pts_homo = np.concatenate([traj_3d, ones], axis=2)  # (T, N, 4)

  # Batch transform: (T, 4, 4) @ (T, 4, N) -> (T, 4, N) -> (T, N, 4)
  pts_cam = np.einsum('tij,tnj->tni', extrinsics_w2c, pts_homo)  # (T, N, 4)

  z = pts_cam[:, :, 2]  # (T, N)

  # Valid: visible AND in front of camera
  valid = vis_2d & (z > 0.01)  # (T, N)

  if not valid.any():
    return np.array([], dtype=np.float32)

  # Project to 2D (only compute where valid, but vectorized)
  # We compute for everything then mask — faster than fancy indexing
  with np.errstate(divide='ignore', invalid='ignore'):
    u_proj = fx * pts_cam[:, :, 0] / z + cx  # (T, N)
    v_proj = fy * pts_cam[:, :, 1] / z + cy  # (T, N)

  # Compute errors
  du = u_proj - traj_2d[:, :, 0]  # (T, N)
  dv = v_proj - traj_2d[:, :, 1]  # (T, N)
  errors = np.sqrt(du * du + dv * dv)  # (T, N)

  return errors[valid]


def compute_all_episode_stats(episode_id, tracks_root, depth_root):
  """Compute ALL statistics for a single episode in one pass.

  Merges track stats, reprojection error, and multi-view consistency
  into a single function to avoid redundant file I/O.
  """
  tracks_dir = os.path.abspath(
      os.path.expanduser(os.path.join(tracks_root, episode_id)))
  if not os.path.exists(tracks_dir):
    return None

  # ---- Load 3D tracks (once) ----
  tracks_3d_path = os.path.join(tracks_dir, "tracks_3d.npz")
  if not os.path.exists(tracks_3d_path):
    return None

  data_3d = np.load(tracks_3d_path)
  traj_3d = data_3d["traj_3d"]      # (T, N, 3)
  vis_global = data_3d["vis_global"]  # (T, N)
  T, N, _ = traj_3d.shape

  stats = {"episode_id": episode_id}
  stats["n_frames"] = int(T)
  stats["n_points_total"] = int(N)

  # ---- Track metadata ----
  meta_path = os.path.join(tracks_dir, "track_metadata.npz")
  if os.path.exists(meta_path):
    meta = np.load(meta_path)
    stats["n_env_points"] = int(meta["n_env"])
    stats["n_robot_points"] = int(meta["n_robot"])
  else:
    stats["n_env_points"] = N
    stats["n_robot_points"] = 0

  # ---- Visibility stats ----
  vis_per_point = vis_global.sum(axis=0)  # (N,)
  vis_per_frame = vis_global.sum(axis=1)  # (T,)

  stats["avg_track_length"] = float(np.mean(vis_per_point))
  stats["median_track_length"] = float(np.median(vis_per_point))
  stats["min_track_length"] = int(np.min(vis_per_point))
  stats["max_track_length"] = int(np.max(vis_per_point))
  stats["pct_always_visible"] = float(np.mean(vis_per_point == T) * 100)
  stats["avg_points_per_frame"] = float(np.mean(vis_per_frame))
  stats["min_points_per_frame"] = int(np.min(vis_per_frame))
  stats["max_points_per_frame"] = int(np.max(vis_per_frame))

  # ---- 3D extent ----
  if vis_global.any():
    visible_pts = traj_3d[vis_global]  # (M, 3)
    bbox_min = visible_pts.min(axis=0)
    bbox_max = visible_pts.max(axis=0)
    bbox_size = bbox_max - bbox_min
    stats["scene_extent_m"] = float(np.linalg.norm(bbox_size))
    stats["scene_bbox_x_m"] = float(bbox_size[0])
    stats["scene_bbox_y_m"] = float(bbox_size[1])
    stats["scene_bbox_z_m"] = float(bbox_size[2])

  # ---- 3D motion ----
  deltas = np.diff(traj_3d, axis=0)
  delta_norms = np.linalg.norm(deltas, axis=2)
  vis_transitions = vis_global[:-1] & vis_global[1:]
  if vis_transitions.any():
    avg_disp = float(np.nanmean(
        np.where(vis_transitions, delta_norms, np.nan)))
    total_disp = np.nansum(
        np.where(vis_transitions, delta_norms, 0.0), axis=0)
    stats["avg_per_frame_displacement_mm"] = avg_disp * 1000
    stats["avg_total_displacement_mm"] = float(np.mean(total_disp)) * 1000

  # ---- Per-camera data (single pass for all metrics) ----
  cam_dirs = sorted([
      d for d in os.listdir(tracks_dir)
      if os.path.isdir(os.path.join(tracks_dir, d))
  ])
  stats["n_cameras"] = len(cam_dirs)

  cam_vis_list = []       # for multi-view consistency
  all_reproj_errors = []  # for reprojection error

  for cam_dir_name in cam_dirs:
    cam_dir = os.path.join(tracks_dir, cam_dir_name)

    tracks_2d_path = os.path.join(cam_dir, "tracks_2d.npz")
    intrinsics_path = os.path.join(cam_dir, "intrinsics.npy")
    extrinsics_path = os.path.join(cam_dir, "extrinsics_w2c.npy")

    if not os.path.exists(tracks_2d_path):
      continue

    cam_data = np.load(tracks_2d_path)
    vis_2d = cam_data["vis_2d"]  # (T, N)
    cam_vis_list.append(vis_2d)

    # Reprojection error (vectorized)
    if os.path.exists(intrinsics_path) and os.path.exists(extrinsics_path):
      traj_2d = cam_data["traj_2d"]
      intrinsics = np.load(intrinsics_path)
      extrinsics_w2c = np.load(extrinsics_path)

      errs = _vectorized_reprojection_error(
          traj_3d, traj_2d, vis_2d, intrinsics, extrinsics_w2c)
      if len(errs) > 0:
        all_reproj_errors.append(errs)

  # ---- Multi-view consistency ----
  if cam_vis_list:
    vis_stack = np.stack(cam_vis_list, axis=0)  # (C, T, N)
    cams_per_obs = vis_stack.sum(axis=0)  # (T, N)

    # Per-point: how many cameras ever see it
    cam_ever_sees = (vis_stack.sum(axis=1) > 0)  # (C, N)
    n_cameras_per_point = cam_ever_sees.sum(axis=0)  # (N,)
    stats["avg_cameras_per_point"] = float(np.mean(n_cameras_per_point))

    # Per-(frame, point): multi-view rate
    if vis_global.any():
      cams_at_visible = cams_per_obs[vis_global]
      stats["avg_cameras_per_visible_obs"] = float(cams_at_visible.mean())
      stats["pct_multi_view"] = float(
          (cams_at_visible >= 2).sum() / vis_global.sum() * 100)
    else:
      stats["avg_cameras_per_visible_obs"] = 0.0
      stats["pct_multi_view"] = 0.0

  # ---- Reprojection error ----
  if all_reproj_errors:
    all_errs = np.concatenate(all_reproj_errors)
    stats["reproj_mean_px"] = float(np.mean(all_errs))
    stats["reproj_median_px"] = float(np.median(all_errs))
    stats["reproj_p95_px"] = float(np.percentile(all_errs, 95))
    stats["reproj_p99_px"] = float(np.percentile(all_errs, 99))
    stats["reproj_n_measurements"] = int(len(all_errs))

  # ---- Disk usage ----
  total_bytes = 0
  n_files = 0
  for dirpath, _, filenames in os.walk(tracks_dir):
    for f in filenames:
      fp = os.path.join(dirpath, f)
      if os.path.isfile(fp):
        total_bytes += os.path.getsize(fp)
        n_files += 1
  stats["tracks_size_mb"] = round(total_bytes / 1e6, 1)

  return stats


def _worker(episode_id, tracks_root, depth_root):
  """Subprocess-safe wrapper."""
  try:
    return compute_all_episode_stats(episode_id, tracks_root, depth_root)
  except Exception as e:
    return {"episode_id": episode_id, "_error": str(e)}


# ============================================================================
# Main
# ============================================================================

def main():
  parser = argparse.ArgumentParser(
      description="Compute DROID dataset statistics for tech report")
  parser.add_argument("--depth_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/depth")
  parser.add_argument("--extrinsics_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/extrinsics")
  parser.add_argument("--tracks_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/tracks")
  parser.add_argument("--output_dir", type=str,
                      default="tech_report/stats_output")
  parser.add_argument("--max_episodes", type=int, default=-1,
                      help="Max episodes to analyze (-1 = all)")
  parser.add_argument("--workers", type=int, default=0,
                      help="Parallel workers (0 = auto = num CPUs)")
  args = parser.parse_args()

  output_dir = os.path.abspath(args.output_dir)
  os.makedirs(output_dir, exist_ok=True)

  if args.workers <= 0:
    args.workers = min(os.cpu_count() or 4, 32)

  print("=" * 60)
  print("DROID Pipeline: Dataset Statistics")
  print("=" * 60)

  # ------------------------------------------------------------------
  # 1. Pipeline Coverage
  # ------------------------------------------------------------------
  print("\n📊 Stage 1: Pipeline coverage...")
  coverage, episode_lists = count_episodes_per_stage(
      args.depth_root, args.extrinsics_root, args.tracks_root)
  print(json.dumps(coverage, indent=2))

  # ------------------------------------------------------------------
  # 2. Per-Episode Stats (parallel, single-pass)
  # ------------------------------------------------------------------
  completed_eps = episode_lists["all_completed"]
  if args.max_episodes > 0:
    completed_eps = completed_eps[:args.max_episodes]

  print(f"\n📊 Stage 2: Per-episode stats "
        f"({len(completed_eps)} episodes, {args.workers} workers)...")

  all_stats = []
  errors_list = []

  worker_fn = partial(
      _worker,
      tracks_root=args.tracks_root,
      depth_root=args.depth_root)

  with ProcessPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(worker_fn, ep_id): ep_id
               for ep_id in completed_eps}

    with tqdm(total=len(futures), desc="Computing stats") as pbar:
      for future in as_completed(futures):
        result = future.result()
        if result is not None:
          if "_error" in result:
            errors_list.append(result)
          else:
            all_stats.append(result)
        pbar.update(1)

  n_complete = len(all_stats)
  n_errors = len(errors_list)
  print(f"\n✅ {n_complete} episodes processed, {n_errors} errors")

  if n_complete == 0:
    print("❌ No episodes with valid tracks found.")
    return

  # ------------------------------------------------------------------
  # 3. Aggregate Summary
  # ------------------------------------------------------------------
  summary = {"coverage": coverage}

  # Track stats aggregation
  numeric_keys = [k for k in all_stats[0]
                  if isinstance(all_stats[0][k], (int, float))
                  and k != "episode_id"]
  agg = {}
  for k in numeric_keys:
    vals = [s[k] for s in all_stats if k in s and s[k] is not None]
    if vals:
      agg[k] = {
          "mean": round(float(np.mean(vals)), 2),
          "median": round(float(np.median(vals)), 2),
          "min": round(float(np.min(vals)), 2),
          "max": round(float(np.max(vals)), 2),
          "std": round(float(np.std(vals)), 2),
      }
  summary["track_stats"] = agg

  # Reprojection error summary
  reproj_means = [s["reproj_mean_px"] for s in all_stats
                   if "reproj_mean_px" in s]
  reproj_medians = [s["reproj_median_px"] for s in all_stats
                     if "reproj_median_px" in s]
  reproj_p95s = [s["reproj_p95_px"] for s in all_stats
                  if "reproj_p95_px" in s]
  if reproj_means:
    summary["reprojection_error"] = {
        "mean_of_means_px": round(float(np.mean(reproj_means)), 2),
        "mean_of_medians_px": round(float(np.mean(reproj_medians)), 2),
        "mean_of_p95_px": round(float(np.mean(reproj_p95s)), 2),
        "worst_episode_mean_px": round(float(np.max(reproj_means)), 2),
        "best_episode_mean_px": round(float(np.min(reproj_means)), 2),
        "n_episodes": len(reproj_means),
    }

  # Multi-view consistency summary
  pcts = [s["pct_multi_view"] for s in all_stats
          if "pct_multi_view" in s]
  avg_cams = [s["avg_cameras_per_visible_obs"] for s in all_stats
              if "avg_cameras_per_visible_obs" in s]
  if pcts:
    summary["multi_view_consistency"] = {
        "avg_pct_multi_view": round(float(np.mean(pcts)), 1),
        "avg_cameras_per_visible_point": round(float(np.mean(avg_cams)), 2),
        "n_episodes": len(pcts),
    }

  # Disk usage summary
  tracks_mbs = [s["tracks_size_mb"] for s in all_stats
                if "tracks_size_mb" in s]
  if tracks_mbs:
    summary["disk_usage"] = {
        "avg_tracks_mb": round(float(np.mean(tracks_mbs)), 1),
        "total_tracks_gb": round(float(np.sum(tracks_mbs)) / 1000, 1),
    }

  # ------------------------------------------------------------------
  # 4. Save Results
  # ------------------------------------------------------------------
  summary_path = os.path.join(output_dir, "dataset_summary.json")
  with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
  print(f"\n💾 Summary → {summary_path}")

  # Per-episode CSV
  if all_stats:
    # Use union of all keys for CSV header
    all_keys = set()
    for s in all_stats:
      all_keys.update(s.keys())
    fieldnames = sorted(all_keys)

    csv_path = os.path.join(output_dir, "per_episode_stats.csv")
    with open(csv_path, "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
      writer.writeheader()
      writer.writerows(all_stats)
    print(f"💾 Per-episode stats → {csv_path}")

  # Failure analysis
  failure_path = os.path.join(output_dir, "failure_analysis.json")
  with open(failure_path, "w") as f:
    json.dump({
        "coverage": coverage,
        "failed_episodes": {
            "at_extrinsics": episode_lists["failed_at_extrinsics"],
            "at_tracks": episode_lists["failed_at_tracks"],
        },
        "compute_errors": errors_list,
    }, f, indent=2)
  print(f"💾 Failure analysis → {failure_path}")

  # ------------------------------------------------------------------
  # 5. Print Summary Table
  # ------------------------------------------------------------------
  print("\n" + "=" * 60)
  print("SUMMARY")
  print("=" * 60)
  print(json.dumps(summary, indent=2))

  return summary


if __name__ == "__main__":
  main()
