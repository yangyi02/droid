#!/usr/bin/env python3
"""Select evaluation episodes from pre-computed metrics CSV.

Reads the merged metrics CSV produced by compute_metrics.py and
applies stratified sampling to select a diverse, high-quality subset.

Usage:
  # Select 50 episodes (reads from default metrics output directory)
  python tapvidmv/select_episodes.py --n 50

  # Select with stricter quality thresholds
  python tapvidmv/select_episodes.py --n 50 \
    --max_chamfer 0.05 --max_depth_residual 20

Output:
  episodes_eval50.txt — selected episode IDs, one per line.
  episodes_eval50_details.csv — full metrics for selected episodes.
"""

import argparse
import csv
import os
import sys

import numpy as np

# tapvidmv/ sits one level below the repo root, so running this file directly
# puts tapvidmv/ — not the repo root — on sys.path. Prepend the repo root so
# `core` resolves the same way it does for the top-level pipeline scripts.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.io import OUTPUT_ROOT


def load_metrics(csv_path):
  """Load metrics CSV into list of dicts."""
  with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
  print(f"Loaded {len(rows)} episodes from {csv_path}")
  return rows


def safe_float(val, default=float("nan")):
  """Convert string to float, handling nan/missing."""
  if val is None or val == "" or val == "nan":
    return default
  try:
    return float(val)
  except (ValueError, TypeError):
    return default


def apply_quality_filter(rows, max_chamfer, max_depth_residual,
                         min_static_points, min_frames, min_ee_travel):
  """Filter episodes by quality thresholds and a floor on robot motion.

  The motion floor is not a quality test, it is a usefulness test. An episode
  where the arm never moves scores *better* on chamfer and depth residual --
  no motion blur, no FK error to accumulate -- so quality thresholds alone
  actively favour it, while it is worth nothing to a tracking benchmark.
  """
  filtered = []
  n_frozen = 0
  for row in rows:
    chamfer = safe_float(row.get("chamfer_total"))
    depth_res = safe_float(row.get("depth_residual_overall_median_mm"))
    n_static = safe_float(row.get("n_static"), 0)
    n_frames = safe_float(row.get("n_frames"), 0)
    ee_travel = safe_float(row.get("ee_travel_m"), 0)

    # Skip episodes with missing critical metrics
    if np.isnan(chamfer) or np.isnan(depth_res):
      continue

    # Apply thresholds
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
  """Indices 0..n-1 ordered so that every prefix is spread across the range.

  Bisection, breadth first: the median comes first, then the two quartiles,
  then the eighths. Taking the first k of this order samples the range evenly
  for any k, which is what a scene needs when its quota is not known until the
  round-robin has run.

  Deliberately not `linspace(0, n-1, k)`: its first index is always 0, so a
  scene with a quota of one always contributes its slowest episode. That is the
  bug this selection used to have -- with 13 site groups it guaranteed 13
  minimum-motion episodes in a set of 50.
  """
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
  """The scene id: the middle field of a DROID episode id.

  `AUTOLab+0d4edc83+2023-10-21-19h-02m-53s` -> `0d4edc83`. Episodes sharing it
  come from one session: same table, same camera rig, usually the same task.
  Two of them are near-duplicates for a tracking benchmark, however different
  their metrics look, which is why this and not `site` is the unit to spread
  across -- there are 62 scenes against 13 sites.
  """
  parts = row["episode_id"].split("+")
  return parts[1] if len(parts) >= 2 else row.get("site", "UNKNOWN")


def sample_diverse(rows, n_target):
  """Pick `n_target` episodes spread as widely as possible over scenes.

  Quotas are equal per scene, not proportional to scene size: proportional
  quotas hand most of the budget to whichever session happened to record the
  most episodes, which is the opposite of what a benchmark wants. Scenes are
  filled round-robin, so a scene too small for its quota simply passes the
  remainder on to the others.

  Within a scene, episodes are taken evenly spaced along end-effector travel,
  which spans the range of motion present there. That is only safe because the
  quality filter has already dropped the barely-moving episodes: taking the
  lowest-travel episode of every group is exactly what made the old selection
  fill up with frozen arms.

  There is no randomness left here, so the same metrics CSV always yields the
  same set.
  """
  by_scene = {}
  for row in rows:
    by_scene.setdefault(scene_of(row), []).append(row)

  # Order each scene's episodes so that any prefix is spread over that scene's
  # range of motion instead of clustered at one end.
  ordered = {}
  for scene, scene_rows in by_scene.items():
    scene_rows.sort(key=lambda r: safe_float(r.get("ee_travel_m"), 0))
    ordered[scene] = [scene_rows[i] for i in _spread_order(len(scene_rows))]

  # Round-robin across scenes, largest first so ties break predictably.
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
                      default=os.path.join(OUTPUT_ROOT, "metrics", "metrics.csv"),
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

  # Load
  rows = load_metrics(os.path.expanduser(args.input))

  # Filter
  filtered = apply_quality_filter(
      rows, args.max_chamfer, args.max_depth_residual,
      args.min_static_points, args.min_frames, args.min_ee_travel)

  if len(filtered) < args.n:
    print(f"  [WARN] Only {len(filtered)} episodes pass filter, "
          f"requested {args.n}. Relaxing thresholds...")
    # Relax thresholds progressively
    for mult in [1.5, 2.0, 3.0, 5.0]:
      filtered = apply_quality_filter(
          rows,
          args.max_chamfer * mult,
          args.max_depth_residual * mult,
          max(10, args.min_static_points // 2),
          max(10, args.min_frames // 2))
      if len(filtered) >= args.n:
        break

  # Select
  selected = sample_diverse(filtered, args.n)

  output_dir = os.path.expanduser(args.output_dir)
  os.makedirs(output_dir, exist_ok=True)
  list_path = os.path.join(output_dir, f"episodes_eval{args.n}.txt")
  with open(list_path, "w") as f:
    for row in selected:
      f.write(row["episode_id"] + "\n")
  print(f"\nEpisode list: {list_path}")

  # Write detailed CSV
  csv_path = os.path.join(output_dir, f"episodes_eval{args.n}_details.csv")
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sorted(selected[0].keys()))
    writer.writeheader()
    for row in selected:
      writer.writerow(row)
  print(f"Detailed CSV: {csv_path}")

  # Summary
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
