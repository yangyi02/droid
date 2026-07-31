#!/usr/bin/env python3
"""Batch quality metrics evaluation for DROID episodes.

Evaluates pre-computed pipeline outputs (depth, extrinsics, tracks) to
produce a metrics CSV for episode selection.  Designed for parallel
execution on GCP with multi-GPU sharding.

Usage:
  # Single GPU — all episodes
  python evaluate_episodes.py

  # 4-GPU sharding (run each on a separate GPU)
  python evaluate_episodes.py --rank 0 --world_size 4
  python evaluate_episodes.py --rank 1 --world_size 4
  python evaluate_episodes.py --rank 2 --world_size 4
  python evaluate_episodes.py --rank 3 --world_size 4

Output:
  metrics.csv — one row per episode with ~30+ metric columns.
  All ranks append to the same file (file-locked), no merge step needed.
"""

import argparse
import csv
import fcntl
import os
import random
import time
import traceback

import numpy as np

from core.io import load_depth_data, load_extrinsics, get_accelerator
from core.metrics import evaluate_episode
from core.physics import PyBulletRenderer


def load_track_data(episode_id, tracks_root):
  """Load pre-computed track data from disk.

  Returns:
    (final_traj_3d, final_per_cam_vis, n_static, n_robot) or
    (None, None, 0, 0) if not found.
  """
  ep_dir = os.path.abspath(
      os.path.expanduser(os.path.join(tracks_root, episode_id)))

  tracks_path = os.path.join(ep_dir, "tracks_3d.npz")
  meta_path = os.path.join(ep_dir, "track_metadata.npz")

  if not os.path.exists(tracks_path) or not os.path.exists(meta_path):
    return None, None, 0, 0

  tracks_data = np.load(tracks_path)
  meta_data = np.load(meta_path)

  final_traj_3d = tracks_data["traj_3d"]  # (T, N, 3)
  n_static = int(meta_data["n_static"])
  n_robot = int(meta_data["n_robot"])

  # Load per-camera visibility
  final_per_cam_vis = {}
  for cam_dir_name in os.listdir(ep_dir):
    cam_dir = os.path.join(ep_dir, cam_dir_name)
    vis_path = os.path.join(cam_dir, "tracks_2d.npz")
    if os.path.isdir(cam_dir) and os.path.exists(vis_path):
      cam_data = np.load(vis_path)
      final_per_cam_vis[cam_dir_name] = cam_data["vis_2d"]

  if not final_per_cam_vis:
    return None, None, 0, 0

  return final_traj_3d, final_per_cam_vis, n_static, n_robot


def evaluate_single_episode(episode_id, depth_root, extrinsics_root,
                            tracks_root, device, pb_renderer):
  """Evaluate all metrics for one episode.

  Returns:
    dict of metric_name → value, or None on failure.
  """
  # Load depth data (with video for depth coverage stats)
  scene_constants = load_depth_data(
      episode_id, depth_root, load_video="first_frame")
  scene_state = load_extrinsics(scene_constants, extrinsics_root)

  # Load track data (if available)
  final_traj_3d, final_per_cam_vis, n_static, n_robot = \
      load_track_data(episode_id, tracks_root)

  has_tracks = final_traj_3d is not None

  # Compute all metrics
  metrics = evaluate_episode(
      scene_constants, scene_state, device,
      final_traj_3d=final_traj_3d,
      final_per_cam_vis=final_per_cam_vis,
      n_static=n_static,
      n_robot=n_robot,
      tracks_root=tracks_root,
      compute_extrinsics_metrics=True,
      pb_renderer=pb_renderer,
  )
  metrics["has_tracks"] = has_tracks

  return metrics


def main():
  parser = argparse.ArgumentParser(
      description="Batch quality metrics evaluation for DROID episodes")
  parser.add_argument("--rank", type=int, default=0,
                      help="Rank of this process (for multi-GPU sharding)")
  parser.add_argument("--world_size", type=int, default=1,
                      help="Total number of processes")
  parser.add_argument("--limit", type=int, default=-1,
                      help="Limit total episodes to process (-1 = all)")
  parser.add_argument("--depth_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/depth")
  parser.add_argument("--extrinsics_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/extrinsics")
  parser.add_argument("--tracks_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/tracks")
  parser.add_argument("--output_dir", type=str,
                      default="~/droid_data/output/mv-tap/droid/metrics")
  parser.add_argument("--require_tracks", action="store_true",
                      help="Only evaluate episodes with track data")
  args = parser.parse_args()

  # Discover available episodes (from depth output, the first pipeline stage)
  depth_abs = os.path.abspath(os.path.expanduser(args.depth_root))
  ext_abs = os.path.abspath(os.path.expanduser(args.extrinsics_root))
  tracks_abs = os.path.abspath(os.path.expanduser(args.tracks_root))
  output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
  os.makedirs(output_dir, exist_ok=True)

  # Find episodes that have both depth and extrinsics
  depth_eps = set(d for d in os.listdir(depth_abs)
                  if os.path.isdir(os.path.join(depth_abs, d)))
  ext_eps = set(d for d in os.listdir(ext_abs)
                if os.path.isdir(os.path.join(ext_abs, d)))
  available_eps = sorted(depth_eps & ext_eps)

  if args.require_tracks and os.path.exists(tracks_abs):
    tracks_eps = set(d for d in os.listdir(tracks_abs)
                     if os.path.isdir(os.path.join(tracks_abs, d)))
    available_eps = sorted(set(available_eps) & tracks_eps)

  print(f"Found {len(available_eps)} episodes with depth + extrinsics")

  # Deterministic shuffle for load balancing
  random.seed(42)
  random.shuffle(available_eps)

  if args.limit > 0:
    available_eps = available_eps[:args.limit]

  # Shard across ranks
  target_eps = available_eps[args.rank::args.world_size]
  print(f"Rank {args.rank}/{args.world_size}: "
        f"{len(target_eps)} episodes assigned")

  # Setup
  device = get_accelerator()
  pb_renderer = PyBulletRenderer()

  # Output CSV (shared across all ranks, file-locked)
  csv_path = os.path.join(output_dir, "metrics.csv")

  # Check which episodes are already evaluated (resume-friendly)
  done_eps = set()
  if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
    with open(csv_path, "r") as f:
      reader = csv.DictReader(f)
      for row in reader:
        done_eps.add(row.get("episode_id", ""))

  todo_eps = [ep for ep in target_eps if ep not in done_eps]
  print(f"{len(todo_eps)} remaining ({len(done_eps)} already done)")

  succeeded = 0
  failed = 0

  for idx, ep_id in enumerate(todo_eps):
    t0 = time.time()
    print(f"\n[{idx + 1}/{len(todo_eps)}] Episode: {ep_id}")

    try:
      metrics = evaluate_single_episode(
          ep_id, args.depth_root, args.extrinsics_root,
          args.tracks_root, device, pb_renderer)

      if metrics is None:
        print(f"  [WARN] Skipped (no data)")
        failed += 1
        continue

      # Write to CSV (append mode, file-locked, header-safe)
      with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        # Check if header exists (another rank may have written it)
        needs_header = (f.tell() == 0)
        if not needs_header:
          f.seek(0, 2)  # seek to end
          needs_header = (f.tell() == 0)
        writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
        if needs_header:
          writer.writeheader()
        writer.writerow(metrics)
        fcntl.flock(f, fcntl.LOCK_UN)

      elapsed = time.time() - t0
      chamfer = metrics.get("chamfer_total", float("nan"))
      depth_res = metrics.get("depth_residual_overall_median_mm", float("nan"))
      print(f"  [OK] Done in {elapsed:.1f}s | "
            f"chamfer={chamfer:.4f} | "
            f"depth_residual_median={depth_res:.1f}mm")
      succeeded += 1

    except Exception as e:
      print(f"  [FAIL] Failed: {e}")
      traceback.print_exc()
      failed += 1

      # Log failure
      fail_path = os.path.join(output_dir, "failures.txt")
      with open(fail_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(f"{ep_id}\t{str(e)}\n")
        fcntl.flock(f, fcntl.LOCK_UN)

  print(f"\nEvaluation complete!")
  print(f"   Succeeded: {succeeded}/{len(todo_eps)}")
  print(f"   Failed:    {failed}/{len(todo_eps)}")
  print(f"   Output:    {csv_path}")


if __name__ == "__main__":
  main()
