#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config

config = get_config()


def count_episodes_per_stage(depth_root, extrinsics_root, tracks_root):
  def list_episode_dirs(root):
    root = os.path.abspath(os.path.expanduser(root))
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")])

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


def compute_all_episode_stats(episode_id, tracks_root, depth_root):
  tracks_dir = os.path.abspath(os.path.expanduser(os.path.join(tracks_root, episode_id)))
  if not os.path.exists(tracks_dir):
    return None

  tracks_3d_path = os.path.join(tracks_dir, "tracks_3d.npz")
  if not os.path.exists(tracks_3d_path):
    return None

  data_3d = np.load(tracks_3d_path)
  tracks_3d = data_3d["tracks_3d"]
  T, N, _ = tracks_3d.shape

  stats = {"episode_id": episode_id}
  stats["n_frames"] = int(T)
  stats["n_points_total"] = int(N)

  meta_path = os.path.join(tracks_dir, "track_metadata.npz")
  if os.path.exists(meta_path):
    meta = np.load(meta_path)
    stats["n_robot_points"] = int(meta["n_robot"])
    stats["n_env_points"] = int(meta["n_static"])
  else:
    stats["n_env_points"] = N
    stats["n_robot_points"] = 0

  cam_dirs = sorted([d for d in os.listdir(tracks_dir) if os.path.isdir(os.path.join(tracks_dir, d))])
  stats["n_cameras"] = len(cam_dirs)

  cam_vis_list = []
  for cam_dir_name in cam_dirs:
    tracks_2d_path = os.path.join(tracks_dir, cam_dir_name, "tracks_2d.npz")
    if os.path.exists(tracks_2d_path):
      cam_vis_list.append(np.load(tracks_2d_path)["vis_2d"])

  vis_stack = np.stack(cam_vis_list, axis=0) if cam_vis_list else np.zeros((0, T, N), dtype=bool)
  vis_global = vis_stack.any(axis=0)

  vis_per_point = vis_global.sum(axis=0)
  vis_per_frame = vis_global.sum(axis=1)

  stats["avg_track_length"] = float(np.mean(vis_per_point))
  stats["median_track_length"] = float(np.median(vis_per_point))
  stats["min_track_length"] = int(np.min(vis_per_point))
  stats["max_track_length"] = int(np.max(vis_per_point))
  stats["pct_always_visible"] = float(np.mean(vis_per_point == T) * 100)
  stats["avg_points_per_frame"] = float(np.mean(vis_per_frame))
  stats["min_points_per_frame"] = int(np.min(vis_per_frame))
  stats["max_points_per_frame"] = int(np.max(vis_per_frame))

  if vis_global.any():
    visible_pts = tracks_3d[vis_global]
    bbox_min = visible_pts.min(axis=0)
    bbox_max = visible_pts.max(axis=0)
    bbox_size = bbox_max - bbox_min
    stats["scene_extent_m"] = float(np.linalg.norm(bbox_size))
    stats["scene_bbox_x_m"] = float(bbox_size[0])
    stats["scene_bbox_y_m"] = float(bbox_size[1])
    stats["scene_bbox_z_m"] = float(bbox_size[2])

  deltas = np.diff(tracks_3d, axis=0)
  delta_norms = np.linalg.norm(deltas, axis=2)
  vis_transitions = vis_global[:-1] & vis_global[1:]
  if vis_transitions.any():
    avg_disp = float(np.nanmean(np.where(vis_transitions, delta_norms, np.nan)))
    total_disp = np.nansum(np.where(vis_transitions, delta_norms, 0.0), axis=0)
    stats["avg_per_frame_displacement_mm"] = avg_disp * 1000
    stats["avg_total_displacement_mm"] = float(np.mean(total_disp)) * 1000

  if cam_vis_list:
    cams_per_obs = vis_stack.sum(axis=0)

    cam_ever_sees = vis_stack.sum(axis=1) > 0
    n_cameras_per_point = cam_ever_sees.sum(axis=0)
    stats["avg_cameras_per_point"] = float(np.mean(n_cameras_per_point))

    if vis_global.any():
      cams_at_visible = cams_per_obs[vis_global]
      stats["avg_cameras_per_visible_obs"] = float(cams_at_visible.mean())
      stats["pct_multi_view"] = float((cams_at_visible >= 2).sum() / vis_global.sum() * 100)
    else:
      stats["avg_cameras_per_visible_obs"] = 0.0
      stats["pct_multi_view"] = 0.0

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
  return compute_all_episode_stats(episode_id, tracks_root, depth_root)


def main():
  parser = argparse.ArgumentParser(description="Compute DROID dataset statistics for tech report")
  parser.add_argument("--depth_root", type=str, default=config.paths.depth)
  parser.add_argument("--extrinsics_root", type=str, default=config.paths.extrinsics)
  parser.add_argument("--tracks_root", type=str, default=config.paths.tracks)
  parser.add_argument(
    "--output_dir",
    type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats_output"),
  )
  parser.add_argument("--max_episodes", type=int, default=-1, help="Max episodes to analyze (-1 = all)")
  parser.add_argument("--workers", type=int, default=0, help="Parallel workers (0 = auto = num CPUs)")
  parser.add_argument(
    "--metrics_dir",
    type=str,
    default="",
    help="Optional directory of per-episode metrics.json from compute_metrics.py "
    "to integrate depth residual and extrinsics quality into summary.",
  )
  args = parser.parse_args()

  output_dir = os.path.abspath(args.output_dir)
  os.makedirs(output_dir, exist_ok=True)

  if args.workers <= 0:
    args.workers = min(os.cpu_count() or 4, 32)

  print("DROID Pipeline: Dataset Statistics")

  print("\nPipeline coverage...")
  coverage, episode_lists = count_episodes_per_stage(args.depth_root, args.extrinsics_root, args.tracks_root)
  print(json.dumps(coverage, indent=2))

  completed_eps = episode_lists["all_completed"]
  if args.max_episodes > 0:
    completed_eps = completed_eps[: args.max_episodes]

  print(f"\nPer-episode stats ({len(completed_eps)} episodes, {args.workers} workers)...")

  all_stats = []
  errors_list = []

  with ProcessPoolExecutor(max_workers=args.workers) as pool:
    futures = {
      pool.submit(_worker, episode_id, args.tracks_root, args.depth_root): episode_id for episode_id in completed_eps
    }

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
  print(f"\n{n_complete} episodes processed, {n_errors} errors")

  if n_complete == 0:
    print("No episodes with valid tracks found.")
    return

  summary = {"coverage": coverage}

  numeric_keys = [k for k in all_stats[0] if isinstance(all_stats[0][k], (int, float)) and k != "episode_id"]
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

  pcts = [s["pct_multi_view"] for s in all_stats if "pct_multi_view" in s]
  avg_cams = [s["avg_cameras_per_visible_obs"] for s in all_stats if "avg_cameras_per_visible_obs" in s]
  if pcts:
    summary["multi_view_consistency"] = {
      "avg_pct_multi_view": round(float(np.mean(pcts)), 1),
      "avg_cameras_per_visible_point": round(float(np.mean(avg_cams)), 2),
      "n_episodes": len(pcts),
    }

  metrics_dir = os.path.abspath(os.path.expanduser(args.metrics_dir)) if args.metrics_dir else ""
  if metrics_dir and os.path.isdir(metrics_dir):
    print(f"\nIntegrating metrics from {metrics_dir}/*/metrics.json...")
    metric_rows = []
    for path in sorted(glob.glob(os.path.join(metrics_dir, "*", "metrics.json"))):
      with open(path) as f:
        metric_rows.append(json.load(f))

    def _extract_metric(key):
      return [r[key] for r in metric_rows if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]

    def _extract_family(prefix):
      """The metrics write one column per camera or camera pair; the summary pools all of them."""
      keys = sorted({k for r in metric_rows for k in r if k.startswith(f"{prefix}_")})
      return [v for key in keys for v in _extract_metric(key)]

    static_means = _extract_family("depth_residual_static_mm")
    robot_means = _extract_family("depth_residual_robot_mm")

    if static_means or robot_means:
      summary["depth_residual_mm"] = {
        "description": "Predicted 3D depth vs raw sensor depth (primary self-consistency metric).",
        "static_mean": round(float(np.mean(static_means)), 2) if static_means else None,
        "robot_mean": round(float(np.mean(robot_means)), 2) if robot_means else None,
        "n_episodes": len(metric_rows),
      }

    chamfer = _extract_family("chamfer")
    overlap = _extract_family("overlap")
    if chamfer:
      summary["extrinsics_quality"] = {
        "mean_chamfer": round(float(np.mean(chamfer)), 4),
        "median_chamfer": round(float(np.median(chamfer)), 4),
        "mean_overlap": round(float(np.mean(overlap)), 2) if overlap else None,
        "n_episodes": len(chamfer),
      }

  tracks_mbs = [s["tracks_size_mb"] for s in all_stats if "tracks_size_mb" in s]
  if tracks_mbs:
    summary["disk_usage"] = {
      "avg_tracks_mb": round(float(np.mean(tracks_mbs)), 1),
      "total_tracks_gb": round(float(np.sum(tracks_mbs)) / 1000, 1),
    }

  summary_path = os.path.join(output_dir, "dataset_summary.json")
  with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
  print(f"\nSummary -> {summary_path}")

  if all_stats:
    all_keys = set()
    for s in all_stats:
      all_keys.update(s.keys())
    fieldnames = sorted(all_keys)

    csv_path = os.path.join(output_dir, "per_episode_stats.csv")
    with open(csv_path, "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
      writer.writeheader()
      writer.writerows(all_stats)
    print(f"Per-episode stats -> {csv_path}")

  failure_path = os.path.join(output_dir, "failure_analysis.json")
  with open(failure_path, "w") as f:
    json.dump(
      {
        "coverage": coverage,
        "failed_episodes": {
          "at_extrinsics": episode_lists["failed_at_extrinsics"],
          "at_tracks": episode_lists["failed_at_tracks"],
        },
        "compute_errors": errors_list,
      },
      f,
      indent=2,
    )
  print(f"Failure analysis -> {failure_path}")

  print("\nSUMMARY")
  print(json.dumps(summary, indent=2))

  return summary


if __name__ == "__main__":
  main()
