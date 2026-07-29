#!/usr/bin/env python3
"""Select evaluation episodes from pre-computed metrics CSV.

Reads the merged metrics CSV produced by evaluate_episodes.py and
applies stratified sampling to select a diverse, high-quality subset.

Usage:
  # Select 50 episodes (reads from default metrics output directory)
  python select_episodes.py --n 50

  # Select with stricter quality thresholds
  python select_episodes.py --n 50 \
    --max_chamfer 0.05 --max_depth_residual 20

Output:
  episodes_eval50.txt — selected episode IDs, one per line.
  episodes_eval50_details.csv — full metrics for selected episodes.
"""

import argparse
import csv
import os
import random
import sys

import numpy as np


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
                         min_static_points, min_frames):
  """Filter episodes by quality thresholds."""
  filtered = []
  for row in rows:
    chamfer = safe_float(row.get("chamfer_total"))
    depth_res = safe_float(row.get("depth_residual_overall_median_mm"))
    n_static = safe_float(row.get("n_static"), 0)
    n_frames = safe_float(row.get("n_frames"), 0)

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

    filtered.append(row)

  print(f"  ✅ {len(filtered)}/{len(rows)} episodes passed quality filter")
  return filtered


def stratified_sample(rows, n_target, seed=42):
  """Sample episodes stratified by site, with motion diversity.

  Strategy:
    1. Allocate quotas proportional to site frequency (min 1 per site).
    2. Within each site, sort by ee_travel_m and pick evenly spaced
       indices to ensure a mix of small/large motion episodes.
    3. If a site has fewer episodes than its quota, take all of them
       and redistribute the remainder.
  """
  rng = random.Random(seed)

  # Group by site
  by_site = {}
  for row in rows:
    site = row.get("site", "UNKNOWN")
    by_site.setdefault(site, []).append(row)

  n_sites = len(by_site)
  print(f"  {n_sites} unique sites: {sorted(by_site.keys())}")

  # Allocate quotas proportional to site size (min 1)
  total = len(rows)
  quotas = {}
  for site, site_rows in by_site.items():
    quota = max(1, round(len(site_rows) / total * n_target))
    quotas[site] = quota

  # Adjust to hit exactly n_target
  allocated = sum(quotas.values())
  sites_by_size = sorted(quotas.keys(),
                         key=lambda s: len(by_site[s]), reverse=True)
  idx = 0
  while allocated != n_target:
    site = sites_by_size[idx % len(sites_by_size)]
    if allocated < n_target:
      quotas[site] += 1
      allocated += 1
    elif allocated > n_target and quotas[site] > 1:
      quotas[site] -= 1
      allocated -= 1
    idx += 1
    if idx > n_target * 10:  # safety
      break

  print(f"  Site quotas: {dict(sorted(quotas.items()))}")

  # Within each site, pick evenly spaced by motion
  selected = []
  for site in sorted(by_site.keys()):
    site_rows = by_site[site]
    quota = min(quotas.get(site, 1), len(site_rows))

    # Sort by end-effector travel (motion diversity)
    site_rows.sort(key=lambda r: safe_float(r.get("ee_travel_m"), 0))

    if len(site_rows) <= quota:
      # Take all
      selected.extend(site_rows)
    else:
      # Evenly spaced indices
      indices = np.linspace(0, len(site_rows) - 1, quota).astype(int)
      for i in indices:
        selected.append(site_rows[i])

    print(f"    {site}: {quota}/{len(site_rows)} selected")

  # Fill remaining slots if we're under target (from largest sites)
  remaining = n_target - len(selected)
  if remaining > 0:
    selected_ids = set(r["episode_id"] for r in selected)
    pool = [r for r in rows if r["episode_id"] not in selected_ids]
    rng.shuffle(pool)
    selected.extend(pool[:remaining])

  return selected[:n_target]


def main():
  parser = argparse.ArgumentParser(
      description="Select evaluation episodes from metrics CSV")
  parser.add_argument("--input", type=str,
                      default="~/droid_data/output/mv-tap/droid/metrics/metrics.csv",
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
  parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for sampling")
  parser.add_argument("--output_dir", type=str, default=".",
                      help="Output directory")
  args = parser.parse_args()

  # Load
  rows = load_metrics(args.input)

  # Filter
  filtered = apply_quality_filter(
      rows, args.max_chamfer, args.max_depth_residual,
      args.min_static_points, args.min_frames)

  if len(filtered) < args.n:
    print(f"  ⚠️ Only {len(filtered)} episodes pass filter, "
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
  selected = stratified_sample(filtered, args.n, seed=args.seed)

  # Write episode list
  os.makedirs(args.output_dir, exist_ok=True)
  list_path = os.path.join(args.output_dir, f"episodes_eval{args.n}.txt")
  with open(list_path, "w") as f:
    for row in selected:
      f.write(row["episode_id"] + "\n")
  print(f"\nEpisode list: {list_path}")

  # Write detailed CSV
  csv_path = os.path.join(args.output_dir, f"episodes_eval{args.n}_details.csv")
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sorted(selected[0].keys()))
    writer.writeheader()
    for row in selected:
      writer.writerow(row)
  print(f"Detailed CSV: {csv_path}")

  # Summary
  print(f"\n{'=' * 60}")
  print(f"✅ Selected {len(selected)} episodes")
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
  print(f"{'=' * 60}")


if __name__ == "__main__":
  main()
