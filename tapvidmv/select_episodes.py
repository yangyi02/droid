#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import operator
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config

config = get_config()

# Written by compute_metrics.py, one per episode, the way every other stage writes its output.
METRICS_FILE = "metrics.json"

OPS = {"<=": operator.le, ">=": operator.ge}

# A cut is a metrics column, or the prefix of the one-per-camera columns it is judged on: the
# metrics store every view and this is where the worst of them condemns an episode, because a
# benchmark is only as good as its worst view. The quality numbers are half again the worst of
# eight episodes that looked fine, so they pass those with headroom rather than mean anything
# yet: calibrate against a full metrics run, using the rejection counts this prints to see which
# one binds. The counts below them are floors, not thresholds to tune.
CUTS = {
  "cross_view_px": ("<=", 25.0),
  "cross_view_wrist_px": ("<=", 250.0),
  "depth_residual_static_mm": ("<=", 30.0),
  "depth_residual_robot_mm": ("<=", 30.0),
  "chamfer": ("<=", 0.08),
  "track_jitter_mm": ("<=", 10.0),
  "vis_percent": (">=", 20.0),
  "n_static": (">=", 50),
  "n_frames": (">=", 30),
  "ee_travel_m": (">=", 0.3),
}


def load_metrics(metrics_root):
  rows = []
  for path in sorted(glob.glob(os.path.join(os.path.expanduser(metrics_root), "*", METRICS_FILE))):
    with open(path) as f:
      rows.append(json.load(f))

  print(f"Loaded {len(rows)} episodes from {metrics_root}/*/{METRICS_FILE}")
  return rows


def safe_float(value):
  """A missing or unwritten column reads as nan, and nan fails every comparison, so it fails the cut."""
  return float("nan") if value in (None, "", "nan") else float(value)


def cut_value(row, column, op):
  """The column itself, or the worst camera when the metrics wrote one column per camera.

  The op says which end is worst: a ceiling is failed by the largest value, a floor by the
  smallest. A family with nothing finite in it reads as nan, so it fails the cut.
  """
  if column in row:
    return safe_float(row[column])

  values = [v for v in (safe_float(v) for k, v in row.items() if k.startswith(f"{column}_")) if not np.isnan(v)]
  if not values:
    return float("nan")
  return min(values) if op == ">=" else max(values)


def apply_cuts(rows, cuts):
  """Drop what the metrics can already condemn, and say which threshold did it.

  An episode is counted against every cut it fails, so the counts do not sum to the number
  dropped. They are here to show which threshold is binding before you go and move one.
  """
  rejected = Counter()
  kept = []
  for row in rows:
    failing = [column for column, (op, limit) in cuts.items() if not OPS[op](cut_value(row, column, op), limit)]
    rejected.update(failing)
    if not failing:
      kept.append(row)

  print(f"  {len(kept)}/{len(rows)} episodes pass the quality cuts")
  for column, count in rejected.most_common():
    op, limit = cuts[column]
    print(f"    {count:5d} fail {column} {op} {limit}")
  return kept


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


def ee_travel(row):
  return safe_float(row.get("ee_travel_m"))


def sample_diverse(rows, n_target):
  by_scene = {}
  for row in rows:
    by_scene.setdefault(row["scene"], []).append(row)

  ordered = {}
  for scene, scene_rows in by_scene.items():
    scene_rows.sort(key=ee_travel)
    ordered[scene] = [scene_rows[i] for i in _spread_order(len(scene_rows))]

  scenes = [scene for _, scene in sorted((-len(rows), scene) for scene, rows in ordered.items())]
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

  per_scene = Counter(row["scene"] for row in selected)
  if not per_scene:
    print("  Nothing survived the cuts. Move one with --cut before writing a list of nothing.")
    sys.exit(1)

  print(f"  {len(selected)} episodes over {len(per_scene)} scenes (max {max(per_scene.values())} from any one scene)")
  return selected


def report(selected):
  print(f"\nSelected {len(selected)} episodes")
  print(f"  Sites: {dict(sorted(Counter(row['site'] for row in selected).items()))}")
  for column in ("cross_view_px", "cross_view_wrist_px", "depth_residual_static_mm", "ee_travel_m"):
    values = [v for v in (cut_value(row, column, CUTS[column][0]) for row in selected) if not np.isnan(v)]
    if values:
      print(f"  {column}: median={np.median(values):.3f}, range=[{min(values):.3f}, {max(values):.3f}]")


def main():
  parser = argparse.ArgumentParser(description="Select evaluation episodes from metrics CSV")
  parser.add_argument("--input", default=config.paths.metrics, help="Directory of per-episode metrics")
  parser.add_argument("--n", type=int, default=150, help="Size of the candidate pool to hand to the human pass")
  parser.add_argument(
    "--cut",
    action="append",
    default=[],
    metavar="COLUMN=VALUE",
    help=f"Move one threshold, e.g. --cut cross_view_px=2.0. Columns: {', '.join(CUTS)}",
  )
  parser.add_argument(
    "--output_dir",
    default=os.path.dirname(os.path.abspath(__file__)),
    help="Output directory (default: tapvidmv/, next to this script)",
  )
  args = parser.parse_args()

  cuts = dict(CUTS)
  for override in args.cut:
    column, value = override.split("=")
    cuts[column] = (cuts[column][0], float(value))

  rows = load_metrics(os.path.expanduser(args.input))
  selected = sample_diverse(apply_cuts(rows, cuts), args.n)

  output_dir = os.path.expanduser(args.output_dir)
  os.makedirs(output_dir, exist_ok=True)

  list_path = os.path.join(output_dir, f"episodes_eval{args.n}.txt")
  with open(list_path, "w") as f:
    f.writelines(row["episode_id"] + "\n" for row in selected)

  csv_path = os.path.join(output_dir, f"episodes_eval{args.n}_details.csv")
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sorted({key for row in selected for key in row}), restval="")
    writer.writeheader()
    writer.writerows(selected)

  report(selected)
  print(f"\nEpisode list: {list_path}\nDetailed CSV: {csv_path}")


if __name__ == "__main__":
  main()
