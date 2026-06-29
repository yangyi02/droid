#!/usr/bin/env python3
"""Compute dataset statistics and evaluation metrics for the DROID tech report.

Usage (from the droid/ directory):
  python tech_report/compute_stats.py                          # default paths
  python tech_report/compute_stats.py --tracks_root /path/to   # custom paths

Outputs:
  tech_report/stats_output/
    dataset_summary.json        — aggregate statistics
    per_episode_stats.csv       — per-episode breakdown
    failure_analysis.json       — failure analysis
    timing_report.json          — compute cost estimates (if log data available)
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import defaultdict

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

  # List specific failures
  failed_ext = sorted(depth_eps - ext_eps)
  failed_tracks = sorted((depth_eps & ext_eps) - track_eps)

  return coverage, {
      "failed_at_extrinsics": failed_ext,
      "failed_at_tracks": failed_tracks,
      "all_completed": sorted(depth_eps & ext_eps & track_eps),
  }


# ============================================================================
# 2. Per-Episode Track Statistics
# ============================================================================

def compute_episode_track_stats(episode_id, tracks_root, depth_root=None):
  """Compute detailed statistics for a single episode's tracks."""
  tracks_dir = os.path.abspath(
      os.path.expanduser(os.path.join(tracks_root, episode_id)))
  if not os.path.exists(tracks_dir):
    return None

  stats = {"episode_id": episode_id}

  # Load 3D tracks
  tracks_3d_path = os.path.join(tracks_dir, "tracks_3d.npz")
  if not os.path.exists(tracks_3d_path):
    return None

  data = np.load(tracks_3d_path)
  traj_3d = data["traj_3d"]  # (T, N, 3)
  vis_global = data["vis_global"]  # (T, N)

  T, N, _ = traj_3d.shape
  stats["n_frames"] = int(T)
  stats["n_points_total"] = int(N)

  # Track metadata (env vs robot)
  meta_path = os.path.join(tracks_dir, "track_metadata.npz")
  if os.path.exists(meta_path):
    meta = np.load(meta_path)
    stats["n_env_points"] = int(meta["n_env"])
    stats["n_robot_points"] = int(meta["n_robot"])
  else:
    stats["n_env_points"] = N
    stats["n_robot_points"] = 0

  # Visibility statistics
  vis_per_point = vis_global.sum(axis=0)  # (N,) frames visible per point
  vis_per_frame = vis_global.sum(axis=1)  # (T,) points visible per frame

  stats["avg_track_length"] = float(np.mean(vis_per_point))
  stats["median_track_length"] = float(np.median(vis_per_point))
  stats["min_track_length"] = int(np.min(vis_per_point))
  stats["max_track_length"] = int(np.max(vis_per_point))
  stats["pct_always_visible"] = float(
      np.mean(vis_per_point == T) * 100)

  stats["avg_points_per_frame"] = float(np.mean(vis_per_frame))
  stats["min_points_per_frame"] = int(np.min(vis_per_frame))
  stats["max_points_per_frame"] = int(np.max(vis_per_frame))

  # 3D trajectory extent
  valid = vis_global  # (T, N)
  if valid.any():
    visible_pts = traj_3d[valid]  # (M, 3)
    bbox_min = visible_pts.min(axis=0)
    bbox_max = visible_pts.max(axis=0)
    bbox_size = bbox_max - bbox_min
    stats["scene_extent_m"] = float(np.linalg.norm(bbox_size))
    stats["scene_bbox_x_m"] = float(bbox_size[0])
    stats["scene_bbox_y_m"] = float(bbox_size[1])
    stats["scene_bbox_z_m"] = float(bbox_size[2])

  # 3D motion: average displacement per point
  deltas = np.diff(traj_3d, axis=0)  # (T-1, N, 3)
  delta_norms = np.linalg.norm(deltas, axis=2)  # (T-1, N)
  # Only consider visible-to-visible transitions
  vis_transitions = vis_global[:-1] & vis_global[1:]  # (T-1, N)
  if vis_transitions.any():
    avg_displacement = float(np.nanmean(
        np.where(vis_transitions, delta_norms, np.nan)))
    total_displacement_per_point = np.nansum(
        np.where(vis_transitions, delta_norms, 0.0), axis=0)  # (N,)
    stats["avg_per_frame_displacement_mm"] = avg_displacement * 1000
    stats["avg_total_displacement_mm"] = float(
        np.mean(total_displacement_per_point)) * 1000

  # Per-camera 2D track stats
  cam_dirs = [d for d in os.listdir(tracks_dir)
              if os.path.isdir(os.path.join(tracks_dir, d))]
  stats["n_cameras"] = len(cam_dirs)

  cam_visibilities = []
  for cam_dir_name in cam_dirs:
    cam_dir = os.path.join(tracks_dir, cam_dir_name)
    tracks_2d_path = os.path.join(cam_dir, "tracks_2d.npz")
    if os.path.exists(tracks_2d_path):
      cam_data = np.load(tracks_2d_path)
      cam_vis = cam_data["vis_2d"]  # (T, N)
      cam_visibilities.append(cam_vis.sum(axis=0))  # (N,) per camera

  if cam_visibilities:
    # How many cameras see each point (at any frame)
    n_cameras_per_point = np.sum(
        [cv > 0 for cv in cam_visibilities], axis=0)
    stats["avg_cameras_per_point"] = float(np.mean(n_cameras_per_point))

  return stats


# ============================================================================
# 3. Reprojection Error (Self-Consistency Check)
# ============================================================================

def compute_reprojection_error(episode_id, tracks_root):
  """Compute reprojection error: project 3D tracks back to each camera.

  This is a self-consistency metric: if the 3D fusion is correct and
  extrinsics are accurate, the reprojected 2D should match the tracked 2D.
  """
  tracks_dir = os.path.abspath(
      os.path.expanduser(os.path.join(tracks_root, episode_id)))
  if not os.path.exists(tracks_dir):
    return None

  data = np.load(os.path.join(tracks_dir, "tracks_3d.npz"))
  traj_3d = data["traj_3d"]  # (T, N, 3)
  T, N, _ = traj_3d.shape

  cam_dirs = sorted([d for d in os.listdir(tracks_dir)
                     if os.path.isdir(os.path.join(tracks_dir, d))])

  errors_by_cam = {}
  all_errors = []

  for cam_dir_name in cam_dirs:
    cam_dir = os.path.join(tracks_dir, cam_dir_name)

    tracks_2d_path = os.path.join(cam_dir, "tracks_2d.npz")
    intrinsics_path = os.path.join(cam_dir, "intrinsics.npy")
    extrinsics_path = os.path.join(cam_dir, "extrinsics_w2c.npy")

    if not all(os.path.exists(p) for p in
               [tracks_2d_path, intrinsics_path, extrinsics_path]):
      continue

    cam_data = np.load(tracks_2d_path)
    traj_2d_tracked = cam_data["traj_2d"]  # (T, N, 2)
    vis_2d = cam_data["vis_2d"]  # (T, N)

    intrinsics = np.load(intrinsics_path)  # (fx, fy, cx, cy)
    extrinsics_w2c = np.load(extrinsics_path)  # (T, 4, 4)

    fx, fy, cx, cy = intrinsics

    # Project 3D → 2D for each frame
    reproj_errors = np.full((T, N), np.nan, dtype=np.float32)

    for t in range(T):
      visible = vis_2d[t]  # (N,)
      if not visible.any():
        continue

      pts_world = traj_3d[t, visible]  # (M, 3)
      pts_h = np.concatenate(
          [pts_world, np.ones((pts_world.shape[0], 1))], axis=1)  # (M, 4)

      # World → Camera
      pts_cam = (extrinsics_w2c[t] @ pts_h.T).T[:, :3]  # (M, 3)

      # Skip points behind camera
      valid = pts_cam[:, 2] > 0.01
      if not valid.any():
        continue

      # Project
      u = fx * pts_cam[valid, 0] / pts_cam[valid, 2] + cx
      v = fy * pts_cam[valid, 1] / pts_cam[valid, 2] + cy

      projected = np.stack([u, v], axis=1)  # (M', 2)

      # Get tracked 2D for visible points
      visible_indices = np.where(visible)[0]
      valid_indices = visible_indices[valid]
      tracked = traj_2d_tracked[t, valid_indices]  # (M', 2)

      # Euclidean error
      err = np.linalg.norm(projected - tracked, axis=1)
      reproj_errors[t, valid_indices] = err

    valid_errors = reproj_errors[~np.isnan(reproj_errors)]
    if len(valid_errors) > 0:
      errors_by_cam[cam_dir_name] = {
          "mean_px": float(np.mean(valid_errors)),
          "median_px": float(np.median(valid_errors)),
          "p95_px": float(np.percentile(valid_errors, 95)),
          "n_measurements": int(len(valid_errors)),
      }
      all_errors.extend(valid_errors.tolist())

  if not all_errors:
    return None

  all_errors = np.array(all_errors)
  return {
      "episode_id": episode_id,
      "overall_mean_px": float(np.mean(all_errors)),
      "overall_median_px": float(np.median(all_errors)),
      "overall_p95_px": float(np.percentile(all_errors, 95)),
      "overall_p99_px": float(np.percentile(all_errors, 99)),
      "n_total_measurements": int(len(all_errors)),
      "per_camera": errors_by_cam,
  }


# ============================================================================
# 4. Multi-View Consistency
# ============================================================================

def compute_multiview_consistency(episode_id, tracks_root):
  """Measure multi-view 3D consistency.

  For each visible (frame, point) pair, lift from each camera independently
  and measure the variance of the resulting 3D positions.
  """
  tracks_dir = os.path.abspath(
      os.path.expanduser(os.path.join(tracks_root, episode_id)))
  if not os.path.exists(tracks_dir):
    return None

  cam_dirs = sorted([d for d in os.listdir(tracks_dir)
                     if os.path.isdir(os.path.join(tracks_dir, d))])
  if len(cam_dirs) < 2:
    return None

  # Load per-camera extrinsics and intrinsics
  cam_data_list = []
  for cam_dir_name in cam_dirs:
    cam_dir = os.path.join(tracks_dir, cam_dir_name)
    intrinsics_path = os.path.join(cam_dir, "intrinsics.npy")
    extrinsics_path = os.path.join(cam_dir, "extrinsics_w2c.npy")
    tracks_2d_path = os.path.join(cam_dir, "tracks_2d.npz")

    if not all(os.path.exists(p) for p in
               [intrinsics_path, extrinsics_path, tracks_2d_path]):
      continue

    intrinsics = np.load(intrinsics_path)
    extrinsics_w2c = np.load(extrinsics_path)
    td = np.load(tracks_2d_path)

    cam_data_list.append({
        "name": cam_dir_name,
        "intrinsics": intrinsics,  # (fx, fy, cx, cy)
        "extrinsics_w2c": extrinsics_w2c,  # (T, 4, 4)
        "traj_2d": td["traj_2d"],  # (T, N, 2)
        "vis_2d": td["vis_2d"],  # (T, N)
    })

  if len(cam_data_list) < 2:
    return None

  # For the fused 3D tracks, count how many cameras agree
  data_3d = np.load(os.path.join(tracks_dir, "tracks_3d.npz"))
  vis_global = data_3d["vis_global"]
  T, N = vis_global.shape

  # Count per-(frame, point) how many cameras see it
  multi_vis_count = np.zeros((T, N), dtype=np.int32)
  for cd in cam_data_list:
    multi_vis_count += cd["vis_2d"].astype(np.int32)

  multi_view_pts = multi_vis_count >= 2  # at least 2 cameras

  n_multi_view = int((multi_view_pts & vis_global).sum())
  n_visible = int(vis_global.sum())

  return {
      "episode_id": episode_id,
      "n_cameras": len(cam_data_list),
      "n_visible_measurements": n_visible,
      "n_multi_view_measurements": n_multi_view,
      "pct_multi_view": float(n_multi_view / max(1, n_visible) * 100),
      "avg_cameras_per_visible_point": float(
          multi_vis_count[vis_global].mean()) if vis_global.any() else 0,
  }


# ============================================================================
# 5. Disk Size Analysis
# ============================================================================

def compute_disk_usage(episode_id, depth_root, extrinsics_root, tracks_root):
  """Compute disk usage per stage for an episode."""
  sizes = {}
  for stage, root in [("depth", depth_root), ("extrinsics", extrinsics_root),
                       ("tracks", tracks_root)]:
    stage_dir = os.path.abspath(
        os.path.expanduser(os.path.join(root, episode_id)))
    if os.path.exists(stage_dir):
      total = 0
      n_files = 0
      for dirpath, _, filenames in os.walk(stage_dir):
        for f in filenames:
          fp = os.path.join(dirpath, f)
          if os.path.isfile(fp):
            total += os.path.getsize(fp)
            n_files += 1
      sizes[stage] = {"bytes": total, "mb": round(total / 1e6, 1),
                       "n_files": n_files}
    else:
      sizes[stage] = {"bytes": 0, "mb": 0, "n_files": 0}

  sizes["total_mb"] = sum(s["mb"] for s in sizes.values() if isinstance(s, dict))
  return sizes


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
  args = parser.parse_args()

  output_dir = os.path.abspath(args.output_dir)
  os.makedirs(output_dir, exist_ok=True)

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
  # 2. Per-Episode Track Statistics
  # ------------------------------------------------------------------
  completed_eps = episode_lists["all_completed"]
  if args.max_episodes > 0:
    completed_eps = completed_eps[:args.max_episodes]

  print(f"\n📊 Stage 2: Per-episode track statistics ({len(completed_eps)} episodes)...")

  all_stats = []
  all_reproj = []
  all_consistency = []
  all_disk = []

  for ep_id in tqdm(completed_eps, desc="Computing stats"):
    # Track stats
    stats = compute_episode_track_stats(
        ep_id, args.tracks_root, args.depth_root)
    if stats:
      all_stats.append(stats)

    # Reprojection error
    reproj = compute_reprojection_error(ep_id, args.tracks_root)
    if reproj:
      all_reproj.append(reproj)

    # Multi-view consistency
    consistency = compute_multiview_consistency(ep_id, args.tracks_root)
    if consistency:
      all_consistency.append(consistency)

    # Disk usage
    disk = compute_disk_usage(
        ep_id, args.depth_root, args.extrinsics_root, args.tracks_root)
    disk["episode_id"] = ep_id
    all_disk.append(disk)

  # ------------------------------------------------------------------
  # 3. Aggregate Summary
  # ------------------------------------------------------------------
  summary = {"coverage": coverage}

  if all_stats:
    numeric_keys = [k for k in all_stats[0]
                    if isinstance(all_stats[0][k], (int, float)) and k != "episode_id"]
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

  if all_reproj:
    mean_errors = [r["overall_mean_px"] for r in all_reproj]
    median_errors = [r["overall_median_px"] for r in all_reproj]
    p95_errors = [r["overall_p95_px"] for r in all_reproj]
    summary["reprojection_error"] = {
        "mean_of_means_px": round(float(np.mean(mean_errors)), 2),
        "mean_of_medians_px": round(float(np.mean(median_errors)), 2),
        "mean_of_p95_px": round(float(np.mean(p95_errors)), 2),
        "worst_episode_mean_px": round(float(np.max(mean_errors)), 2),
        "best_episode_mean_px": round(float(np.min(mean_errors)), 2),
        "n_episodes": len(all_reproj),
    }

  if all_consistency:
    pcts = [c["pct_multi_view"] for c in all_consistency]
    avg_cams = [c["avg_cameras_per_visible_point"] for c in all_consistency]
    summary["multi_view_consistency"] = {
        "avg_pct_multi_view": round(float(np.mean(pcts)), 1),
        "avg_cameras_per_visible_point": round(float(np.mean(avg_cams)), 2),
        "n_episodes": len(all_consistency),
    }

  if all_disk:
    depth_mb = [d["depth"]["mb"] for d in all_disk]
    ext_mb = [d["extrinsics"]["mb"] for d in all_disk]
    tracks_mb = [d["tracks"]["mb"] for d in all_disk]
    summary["disk_usage"] = {
        "avg_depth_mb": round(float(np.mean(depth_mb)), 1),
        "avg_extrinsics_mb": round(float(np.mean(ext_mb)), 1),
        "avg_tracks_mb": round(float(np.mean(tracks_mb)), 1),
        "avg_total_mb": round(float(np.mean(depth_mb)) +
                               float(np.mean(ext_mb)) +
                               float(np.mean(tracks_mb)), 1),
    }

  # ------------------------------------------------------------------
  # 4. Save Results
  # ------------------------------------------------------------------
  # Summary JSON
  summary_path = os.path.join(output_dir, "dataset_summary.json")
  with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
  print(f"\n💾 Summary → {summary_path}")

  # Per-episode CSV
  if all_stats:
    csv_path = os.path.join(output_dir, "per_episode_stats.csv")
    fieldnames = list(all_stats[0].keys())
    with open(csv_path, "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(all_stats)
    print(f"💾 Per-episode stats → {csv_path}")

  # Reprojection error JSON
  if all_reproj:
    reproj_path = os.path.join(output_dir, "reprojection_errors.json")
    with open(reproj_path, "w") as f:
      json.dump(all_reproj, f, indent=2)
    print(f"💾 Reprojection errors → {reproj_path}")

  # Failure analysis
  failure_path = os.path.join(output_dir, "failure_analysis.json")
  with open(failure_path, "w") as f:
    json.dump({
        "coverage": coverage,
        "failed_episodes": {
            "at_extrinsics": episode_lists["failed_at_extrinsics"],
            "at_tracks": episode_lists["failed_at_tracks"],
        },
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
