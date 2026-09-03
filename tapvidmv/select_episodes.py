#!/usr/bin/env python3
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.io


def load_metrics(csv_path):
  with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
  print(f"Loaded {len(rows)} episodes from {csv_path}")
  return rows


def safe_float(val, default=float("nan")):
  if val is None or val == "" or val == "nan":
    return default
  try:
    return float(val)
  except (ValueError, TypeError):
    return default


def apply_quality_filter(rows, max_chamfer, max_depth_residual,
                         min_static_points, min_frames, min_ee_travel):
  filtered = []
  n_frozen = 0
  for row in rows:
    chamfer = safe_float(row.get("chamfer_total"))
    depth_res = safe_float(row.get("depth_residual_overall_median_mm"))
    n_static = safe_float(row.get("n_static"), 0)
    n_frames = safe_float(row.get("n_frames"), 0)
    ee_travel = safe_float(row.get("ee_travel_m"), 0)

    if np.isnan(chamfer) or np.isnan(depth_res):
      continue

    if chamfer > max_chamfer:
      continue
    if depth_res > max_depth_residual:
      continue
    if n_static < min_static_points:
      continue
    if n_frames < min_frames:
      continue
    if ee_travel < min_ee_travel:
      n_frozen += 1
      continue

    filtered.append(row)

  print(f"  {len(filtered)}/{len(rows)} episodes passed quality filter"
        f" ({n_frozen} dropped for moving less than {min_ee_travel} m)")
  return filtered


def _spread_order(n):
  order, segments = [], [(0, n - 1)]
  while segments:
    nxt = []
    for lo, hi in segments:
      mid = (lo + hi) // 2
      order.append(mid)
      nxt.extend([(lo, mid - 1), (mid + 1, hi)])
    segments = [(lo, hi) for lo, hi in nxt if lo <= hi]
  return order


def scene_of(row):
  parts = row["episode_id"].split("+")
  return parts[1] if len(parts) >= 2 else row.get("site", "UNKNOWN")


def sample_diverse(rows, n_target):
  by_scene = {}
  for row in rows:
    by_scene.setdefault(scene_of(row), []).append(row)

  ordered = {}
  for scene, scene_rows in by_scene.items():
    scene_rows.sort(key=lambda r: safe_float(r.get("ee_travel_m"), 0))
    ordered[scene] = [scene_rows[i] for i in _spread_order(len(scene_rows))]

  scenes = sorted(ordered, key=lambda s: (-len(ordered[s]), s))
  selected, round_idx = [], 0
  while len(selected) < n_target:
    took_any = False
    for scene in scenes:
      if round_idx < len(ordered[scene]):
        selected.append(ordered[scene][round_idx])
        took_any = True
        if len(selected) == n_target:
          break
    if not took_any:
      break
    round_idx += 1

  per_scene = {}
  for r in selected:
    per_scene[scene_of(r)] = per_scene.get(scene_of(r), 0) + 1
  print(f"  {len(selected)} episodes over {len(per_scene)} scenes "
        f"(max {max(per_scene.values())} from any one scene)")
  return selected


def main():
  parser = argparse.ArgumentParser(
      description="Select evaluation episodes from metrics CSV")
  parser.add_argument("--input", type=str,
                      default=os.path.join(core.io.OUTPUT_ROOT, "metrics", "metrics.csv"),
                      help="Path to metrics CSV")
  parser.add_argument("--n", type=int, default=50,
                      help="Number of episodes to select")
  parser.add_argument("--max_chamfer", type=float, default=0.10,
                      help="Max chamfer_total threshold (metres)")
  parser.add_argument("--max_depth_residual", type=float, default=30.0,
                      help="Max depth_residual_overall_median_mm threshold")
  parser.add_argument("--min_static_points", type=int, default=50,
                      help="Min number of static track points")
  parser.add_argument("--min_frames", type=int, default=30,
                      help="Min number of frames")
  parser.add_argument("--min_ee_travel", type=float, default=0.3,
                      help="Min end-effector path length in metres. Drops "
                           "episodes where the arm barely moves, which pass "
                           "every quality test but are useless to track")
  parser.add_argument("--output_dir", type=str,
                      default=os.path.dirname(os.path.abspath(__file__)),
                      help="Output directory (default: tapvidmv/, next to this script)")
  args = parser.parse_args()

  rows = load_metrics(os.path.expanduser(args.input))

  filtered = apply_quality_filter(
      rows, args.max_chamfer, args.max_depth_residual,
      args.min_static_points, args.min_frames, args.min_ee_travel)

  if len(filtered) < args.n:
    for mult in [1.5, 2.0, 3.0, 5.0]:
      filtered = apply_quality_filter(
          rows,
          args.max_chamfer * mult,
          args.max_depth_residual * mult,
          max(10, args.min_static_points // 2),
          max(10, args.min_frames // 2))
      if len(filtered) >= args.n:
        break

  selected = sample_diverse(filtered, args.n)

  output_dir = os.path.expanduser(args.output_dir)
  os.makedirs(output_dir, exist_ok=True)
  list_path = os.path.join(output_dir, f"episodes_eval{args.n}.txt")
  with open(list_path, "w") as f:
    for row in selected:
      f.write(row["episode_id"] + "\n")
  print(f"\nEpisode list: {list_path}")

  csv_path = os.path.join(output_dir, f"episodes_eval{args.n}_details.csv")
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sorted(selected[0].keys()))
    writer.writeheader()
    for row in selected:
      writer.writerow(row)
  print(f"Detailed CSV: {csv_path}")

  print(f"\nSelected {len(selected)} episodes")
  sites = {}
  for row in selected:
    s = row.get("site", "?")
    sites[s] = sites.get(s, 0) + 1
  print(f"   Sites: {dict(sorted(sites.items()))}")

  chamfers = [safe_float(r.get("chamfer_total")) for r in selected]
  chamfers = [c for c in chamfers if not np.isnan(c)]
  if chamfers:
    print(f"   Chamfer: median={np.median(chamfers):.4f}, "
          f"range=[{min(chamfers):.4f}, {max(chamfers):.4f}]")

  travels = [safe_float(r.get("ee_travel_m")) for r in selected]
  travels = [t for t in travels if not np.isnan(t)]
  if travels:
    print(f"   EE travel: median={np.median(travels):.2f}m, "
          f"range=[{min(travels):.2f}, {max(travels):.2f}]m")


if __name__ == "__main__":
  main()
