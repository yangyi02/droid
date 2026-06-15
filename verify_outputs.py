#!/usr/bin/env python3
"""DROID Pipeline Output Verification Tool.

Snapshots and compares the outputs of all three pipeline stages to ensure
refactoring does not alter results.

Usage:
  # Step 1: BEFORE refactoring — snapshot golden outputs
  python verify_outputs.py snapshot --episode_id SUCCESS_42_ep0 \
      --depth_root ~/droid_data/output/mv-tap/droid/depth \
      --extrinsics_root ~/droid_data/output/mv-tap/droid/extrinsics \
      --tracks_root ~/droid_data/output/mv-tap/droid/tracks \
      --snapshot_dir ~/droid_verify/golden

  # Step 2: AFTER refactoring — compare against golden
  python verify_outputs.py compare --episode_id SUCCESS_42_ep0 \
      --depth_root ~/droid_data/output/mv-tap/droid/depth \
      --extrinsics_root ~/droid_data/output/mv-tap/droid/extrinsics \
      --tracks_root ~/droid_data/output/mv-tap/droid/tracks \
      --snapshot_dir ~/droid_verify/golden

  # Quick check: just verify output file structure exists
  python verify_outputs.py check --episode_id SUCCESS_42_ep0 \
      --depth_root ~/droid_data/output/mv-tap/droid/depth \
      --extrinsics_root ~/droid_data/output/mv-tap/droid/extrinsics \
      --tracks_root ~/droid_data/output/mv-tap/droid/tracks
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np


# ============================================================================
# Output Contract Definitions
# ============================================================================
# Each stage's expected output structure, used for both checking and comparing.

DEPTH_ROBOT_KEYS = [
    "joint_positions", "gripper_positions", "T_ee_base_all",
    "T_cam_ee_init", "valid_indices", "wrist_serial",
]

DEPTH_PER_CAM_FILES = {
    "video_left.mp4": "binary",
    "video_right.mp4": "binary",
    "video_left_raw.mp4": "binary",
    "video_right_raw.mp4": "binary",
    "raw_depth.npz": "numpy",         # key: depth (uint16 mm)
    "calibration.npz": "numpy",       # keys: K_calib_left, K_calib_right, ...
}

DEPTH_WRIST_EXTRA = {
    "original_raw_depth.npz": "numpy",  # key: depth
    "gripper_mask.npz": "numpy",        # key: mask
    "gripper_depth.npz": "numpy",       # key: depth
}

EXTRINSICS_PER_CAM_FILES = {
    "extrinsics.json": "json",
    "extrinsics_stage1.json": "json",
    "extrinsics_stage2.json": "json",
}

TRACKS_GLOBAL_FILES = {
    "tracks_3d.npz": "numpy",          # keys: traj_3d, vis_global
    "track_metadata.npz": "numpy",     # keys: n_env, n_robot, point_type
}

TRACKS_PER_CAM_FILES = {
    "tracks_2d.npz": "numpy",          # keys: traj_2d, vis_2d
    "intrinsics.npy": "numpy_single",
    "extrinsics_w2c.npy": "numpy_single",
}


# ============================================================================
# Core Comparison Functions
# ============================================================================

def md5_file(path):
  """Compute MD5 hash of a file."""
  h = hashlib.md5()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
      h.update(chunk)
  return h.hexdigest()


def compare_numpy_file(path_a, path_b, atol=0.0, rtol=0.0):
  """Compare two .npz or .npy files array-by-array.

  Returns:
    list of (key, status, detail) tuples.
  """
  results = []

  if path_a.endswith(".npy"):
    a = {"data": np.load(path_a)}
    b = {"data": np.load(path_b)}
  else:
    a = dict(np.load(path_a, allow_pickle=True))
    b = dict(np.load(path_b, allow_pickle=True))

  all_keys = sorted(set(list(a.keys()) + list(b.keys())))
  for key in all_keys:
    if key not in a:
      results.append((key, "FAIL", "missing in A"))
      continue
    if key not in b:
      results.append((key, "FAIL", "missing in B"))
      continue

    va, vb = a[key], b[key]

    # Handle string/object arrays
    if va.dtype.kind in ("U", "S", "O"):
      if np.array_equal(va, vb):
        results.append((key, "PASS", "exact match"))
      else:
        results.append((key, "FAIL", f"value mismatch: {va} vs {vb}"))
      continue

    # Shape check
    if va.shape != vb.shape:
      results.append((key, "FAIL", f"shape {va.shape} vs {vb.shape}"))
      continue

    # Numeric comparison
    if atol == 0.0 and rtol == 0.0:
      if np.array_equal(va, vb):
        results.append((key, "PASS", "bit-identical"))
      else:
        diff = np.abs(va.astype(np.float64) - vb.astype(np.float64))
        results.append((key, "FAIL",
                        f"max_diff={diff.max():.6e}, "
                        f"mean_diff={diff.mean():.6e}, "
                        f"num_diff={np.sum(va != vb)}/{va.size}"))
    else:
      if np.allclose(va, vb, atol=atol, rtol=rtol):
        diff = np.abs(va.astype(np.float64) - vb.astype(np.float64))
        results.append((key, "PASS",
                        f"within tol (max_diff={diff.max():.6e})"))
      else:
        diff = np.abs(va.astype(np.float64) - vb.astype(np.float64))
        results.append((key, "FAIL",
                        f"max_diff={diff.max():.6e}, "
                        f"mean_diff={diff.mean():.6e}"))

  return results


def compare_json_file(path_a, path_b, atol=1e-10):
  """Compare two JSON files (extrinsics format).

  Returns:
    list of (key, status, detail) tuples.
  """
  results = []
  with open(path_a) as f:
    a = json.load(f)
  with open(path_b) as f:
    b = json.load(f)

  all_keys = sorted(set(list(a.keys()) + list(b.keys())))
  for key in all_keys:
    if key not in a:
      results.append((key, "FAIL", "missing in A"))
      continue
    if key not in b:
      results.append((key, "FAIL", "missing in B"))
      continue

    va, vb = a[key], b[key]

    if isinstance(va, bool):
      if va == vb:
        results.append((key, "PASS", f"bool={va}"))
      else:
        results.append((key, "FAIL", f"{va} vs {vb}"))
    elif isinstance(va, (list, float, int)):
      arr_a = np.array(va, dtype=np.float64)
      arr_b = np.array(vb, dtype=np.float64)
      if arr_a.shape != arr_b.shape:
        results.append((key, "FAIL", f"shape {arr_a.shape} vs {arr_b.shape}"))
      elif np.allclose(arr_a, arr_b, atol=atol):
        results.append((key, "PASS",
                        f"shape={arr_a.shape}, "
                        f"max_diff={np.abs(arr_a - arr_b).max():.2e}"))
      else:
        diff = np.abs(arr_a - arr_b)
        results.append((key, "FAIL",
                        f"max_diff={diff.max():.6e}"))
    else:
      if va == vb:
        results.append((key, "PASS", "exact"))
      else:
        results.append((key, "FAIL", f"{type(va).__name__} mismatch"))

  return results


# ============================================================================
# Discovery: find cameras for an episode
# ============================================================================

def discover_cameras(ep_dir):
  """List camera serial subdirectories in an episode directory."""
  if not os.path.isdir(ep_dir):
    return []
  return sorted([
      d for d in os.listdir(ep_dir)
      if os.path.isdir(os.path.join(ep_dir, d))
  ])


# ============================================================================
# Command: check — verify output file structure
# ============================================================================

def cmd_check(args):
  """Verify that all expected output files exist for an episode."""
  ep = args.episode_id
  all_ok = True

  print(f"\n{'='*60}")
  print(f"  Checking output structure for episode: {ep}")
  print(f"{'='*60}")

  # --- Stage 1: depth ---
  depth_dir = os.path.abspath(os.path.expanduser(
      os.path.join(args.depth_root, ep)))
  print(f"\n📂 Stage 1 (depth): {depth_dir}")

  if not os.path.isdir(depth_dir):
    print(f"  ❌ Directory not found!")
    all_ok = False
  else:
    # robot.npz
    robot_path = os.path.join(depth_dir, "robot.npz")
    if os.path.exists(robot_path):
      data = np.load(robot_path, allow_pickle=True)
      keys = list(data.keys())
      print(f"  ✅ robot.npz — keys: {keys}")
      for key in keys:
        arr = data[key]
        if hasattr(arr, 'shape'):
          print(f"     {key}: shape={arr.shape}, dtype={arr.dtype}")
        else:
          print(f"     {key}: {arr}")
    else:
      print(f"  ❌ robot.npz not found")
      all_ok = False

    cameras = discover_cameras(depth_dir)
    print(f"  📷 Cameras: {cameras}")
    for cam_id in cameras:
      cam_dir = os.path.join(depth_dir, cam_id)
      print(f"\n  Camera [{cam_id}]:")
      for fname, ftype in DEPTH_PER_CAM_FILES.items():
        fpath = os.path.join(cam_dir, fname)
        if os.path.exists(fpath):
          size_mb = os.path.getsize(fpath) / 1024 / 1024
          detail = f"{size_mb:.1f} MB"
          if ftype == "numpy":
            try:
              data = np.load(fpath, allow_pickle=True)
              keys = list(data.keys())
              shapes = {k: data[k].shape for k in keys if hasattr(data[k], 'shape')}
              detail += f" | keys={keys}, shapes={shapes}"
            except Exception as e:
              detail += f" | load error: {e}"
          print(f"    ✅ {fname} ({detail})")
        else:
          # Some files are optional (e.g., raw videos)
          if fname.startswith("video_"):
            print(f"    ⚪ {fname} (optional, not found)")
          else:
            print(f"    ❌ {fname} not found")
            all_ok = False

      # Check wrist-specific files
      for fname, ftype in DEPTH_WRIST_EXTRA.items():
        fpath = os.path.join(cam_dir, fname)
        if os.path.exists(fpath):
          data = np.load(fpath, allow_pickle=True)
          keys = list(data.keys())
          shapes = {k: data[k].shape for k in keys if hasattr(data[k], 'shape')}
          print(f"    ✅ {fname} (wrist) — keys={keys}, shapes={shapes}")

  # --- Stage 2: extrinsics ---
  ext_dir = os.path.abspath(os.path.expanduser(
      os.path.join(args.extrinsics_root, ep)))
  print(f"\n📂 Stage 2 (extrinsics): {ext_dir}")

  if not os.path.isdir(ext_dir):
    print(f"  ❌ Directory not found!")
    all_ok = False
  else:
    cameras = discover_cameras(ext_dir)
    print(f"  📷 Cameras: {cameras}")
    for cam_id in cameras:
      cam_dir = os.path.join(ext_dir, cam_id)
      print(f"\n  Camera [{cam_id}]:")
      for fname, ftype in EXTRINSICS_PER_CAM_FILES.items():
        fpath = os.path.join(cam_dir, fname)
        if os.path.exists(fpath):
          with open(fpath) as f:
            data = json.load(f)
          keys = list(data.keys())
          shapes = {}
          for k, v in data.items():
            if isinstance(v, list):
              arr = np.array(v)
              shapes[k] = arr.shape
          print(f"    ✅ {fname} — keys={keys}, shapes={shapes}")
        else:
          if "stage" in fname:
            print(f"    ⚪ {fname} (intermediate, not found)")
          else:
            print(f"    ❌ {fname} not found")
            all_ok = False

  # --- Stage 3: tracks ---
  tracks_dir = os.path.abspath(os.path.expanduser(
      os.path.join(args.tracks_root, ep)))
  print(f"\n📂 Stage 3 (tracks): {tracks_dir}")

  if not os.path.isdir(tracks_dir):
    print(f"  ❌ Directory not found!")
    all_ok = False
  else:
    for fname, ftype in TRACKS_GLOBAL_FILES.items():
      fpath = os.path.join(tracks_dir, fname)
      if os.path.exists(fpath):
        data = np.load(fpath, allow_pickle=True)
        keys = list(data.keys())
        shapes = {k: data[k].shape for k in keys if hasattr(data[k], 'shape')}
        print(f"  ✅ {fname} — keys={keys}, shapes={shapes}")
      else:
        print(f"  ❌ {fname} not found")
        all_ok = False

    cameras = discover_cameras(tracks_dir)
    print(f"  📷 Cameras: {cameras}")
    for cam_id in cameras:
      cam_dir = os.path.join(tracks_dir, cam_id)
      print(f"\n  Camera [{cam_id}]:")
      for fname, ftype in TRACKS_PER_CAM_FILES.items():
        fpath = os.path.join(cam_dir, fname)
        if os.path.exists(fpath):
          if ftype == "numpy":
            data = np.load(fpath, allow_pickle=True)
            keys = list(data.keys())
            shapes = {k: data[k].shape for k in keys}
            print(f"    ✅ {fname} — keys={keys}, shapes={shapes}")
          else:
            arr = np.load(fpath)
            print(f"    ✅ {fname} — shape={arr.shape}, dtype={arr.dtype}")
        else:
          print(f"    ❌ {fname} not found")
          all_ok = False

  # Summary
  print(f"\n{'='*60}")
  if all_ok:
    print(f"  ✅ ALL CHECKS PASSED for episode {ep}")
  else:
    print(f"  ❌ SOME CHECKS FAILED for episode {ep}")
  print(f"{'='*60}\n")
  return all_ok


# ============================================================================
# Command: snapshot — save golden copy
# ============================================================================

def cmd_snapshot(args):
  """Copy all outputs for an episode to a golden snapshot directory."""
  ep = args.episode_id
  snap_dir = os.path.abspath(os.path.expanduser(
      os.path.join(args.snapshot_dir, ep)))

  if os.path.exists(snap_dir):
    print(f"⚠️ Snapshot directory already exists: {snap_dir}")
    print(f"   Removing and re-creating...")
    shutil.rmtree(snap_dir)
  os.makedirs(snap_dir, exist_ok=True)

  stages = {
      "depth": args.depth_root,
      "extrinsics": args.extrinsics_root,
      "tracks": args.tracks_root,
  }

  total_files = 0
  total_bytes = 0

  for stage_name, root in stages.items():
    src_dir = os.path.abspath(os.path.expanduser(
        os.path.join(root, ep)))
    dst_dir = os.path.join(snap_dir, stage_name)

    if not os.path.isdir(src_dir):
      print(f"  ⚪ {stage_name}: source not found at {src_dir}, skipping")
      continue

    print(f"  📸 {stage_name}: {src_dir} → {dst_dir}")
    shutil.copytree(src_dir, dst_dir)

    # Count files
    for dirpath, _, filenames in os.walk(dst_dir):
      for fn in filenames:
        fpath = os.path.join(dirpath, fn)
        total_files += 1
        total_bytes += os.path.getsize(fpath)

  # Write manifest
  manifest = {
      "episode_id": ep,
      "stages": list(stages.keys()),
      "total_files": total_files,
      "total_bytes": total_bytes,
  }
  with open(os.path.join(snap_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

  print(f"\n✅ Snapshot saved: {total_files} files, "
        f"{total_bytes / 1024 / 1024:.1f} MB → {snap_dir}")


# ============================================================================
# Command: compare — diff current outputs against golden snapshot
# ============================================================================

def cmd_compare(args):
  """Compare current outputs against golden snapshot."""
  ep = args.episode_id
  snap_dir = os.path.abspath(os.path.expanduser(
      os.path.join(args.snapshot_dir, ep)))
  atol = args.atol

  if not os.path.isdir(snap_dir):
    print(f"❌ Golden snapshot not found: {snap_dir}")
    print(f"   Run 'snapshot' command first.")
    sys.exit(1)

  stages = {
      "depth": args.depth_root,
      "extrinsics": args.extrinsics_root,
      "tracks": args.tracks_root,
  }

  total_pass = 0
  total_fail = 0
  failures = []

  for stage_name, root in stages.items():
    current_dir = os.path.abspath(os.path.expanduser(
        os.path.join(root, ep)))
    golden_dir = os.path.join(snap_dir, stage_name)

    if not os.path.isdir(golden_dir):
      print(f"\n  ⚪ {stage_name}: no golden data, skipping")
      continue

    print(f"\n{'='*60}")
    print(f"  Comparing Stage: {stage_name}")
    print(f"  Golden:  {golden_dir}")
    print(f"  Current: {current_dir}")
    print(f"{'='*60}")

    if not os.path.isdir(current_dir):
      print(f"  ❌ Current output not found!")
      total_fail += 1
      failures.append(f"{stage_name}: directory missing")
      continue

    # Walk golden directory and compare each file
    for dirpath, _, filenames in os.walk(golden_dir):
      for fn in sorted(filenames):
        if fn == "manifest.json":
          continue

        golden_path = os.path.join(dirpath, fn)
        rel_path = os.path.relpath(golden_path, golden_dir)
        current_path = os.path.join(current_dir, rel_path)

        if not os.path.exists(current_path):
          print(f"  ❌ MISSING: {rel_path}")
          total_fail += 1
          failures.append(f"{stage_name}/{rel_path}: file missing")
          continue

        # Route by file type
        if fn.endswith(".npz"):
          results = compare_numpy_file(golden_path, current_path, atol=atol)
          all_pass = all(r[1] == "PASS" for r in results)
          icon = "✅" if all_pass else "❌"
          print(f"  {icon} {rel_path}:")
          for key, status, detail in results:
            s_icon = "✅" if status == "PASS" else "❌"
            print(f"     {s_icon} [{key}]: {detail}")
            if status == "PASS":
              total_pass += 1
            else:
              total_fail += 1
              failures.append(f"{stage_name}/{rel_path}[{key}]: {detail}")

        elif fn.endswith(".npy"):
          results = compare_numpy_file(golden_path, current_path, atol=atol)
          all_pass = all(r[1] == "PASS" for r in results)
          icon = "✅" if all_pass else "❌"
          print(f"  {icon} {rel_path}:")
          for key, status, detail in results:
            s_icon = "✅" if status == "PASS" else "❌"
            print(f"     {s_icon} {detail}")
            if status == "PASS":
              total_pass += 1
            else:
              total_fail += 1
              failures.append(f"{stage_name}/{rel_path}: {detail}")

        elif fn.endswith(".json"):
          results = compare_json_file(golden_path, current_path, atol=atol)
          all_pass = all(r[1] == "PASS" for r in results)
          icon = "✅" if all_pass else "❌"
          print(f"  {icon} {rel_path}:")
          for key, status, detail in results:
            s_icon = "✅" if status == "PASS" else "❌"
            print(f"     {s_icon} [{key}]: {detail}")
            if status == "PASS":
              total_pass += 1
            else:
              total_fail += 1
              failures.append(f"{stage_name}/{rel_path}[{key}]: {detail}")

        elif fn.endswith(".mp4"):
          # Binary comparison for videos
          hash_a = md5_file(golden_path)
          hash_b = md5_file(current_path)
          if hash_a == hash_b:
            print(f"  ✅ {rel_path}: bit-identical (md5={hash_a[:12]}...)")
            total_pass += 1
          else:
            size_a = os.path.getsize(golden_path)
            size_b = os.path.getsize(current_path)
            print(f"  ❌ {rel_path}: MD5 differs "
                  f"(golden={hash_a[:12]}... current={hash_b[:12]}... "
                  f"sizes={size_a} vs {size_b})")
            total_fail += 1
            failures.append(f"{stage_name}/{rel_path}: md5 mismatch")

        else:
          # Generic binary comparison
          hash_a = md5_file(golden_path)
          hash_b = md5_file(current_path)
          if hash_a == hash_b:
            print(f"  ✅ {rel_path}: identical")
            total_pass += 1
          else:
            print(f"  ❌ {rel_path}: differs")
            total_fail += 1
            failures.append(f"{stage_name}/{rel_path}: content differs")

  # Summary
  print(f"\n{'='*60}")
  print(f"  COMPARISON SUMMARY")
  print(f"  ✅ Pass: {total_pass}")
  print(f"  ❌ Fail: {total_fail}")
  if failures:
    print(f"\n  Failures:")
    for f in failures:
      print(f"    • {f}")
  print(f"{'='*60}\n")

  if total_fail > 0:
    sys.exit(1)


# ============================================================================
# Command: batch_check — check multiple episodes
# ============================================================================

def cmd_batch_check(args):
  """Check output structure for all episodes found in depth_root."""
  depth_abs = os.path.abspath(os.path.expanduser(args.depth_root))
  eps = sorted([
      d for d in os.listdir(depth_abs)
      if os.path.isdir(os.path.join(depth_abs, d))
  ])
  if args.limit > 0:
    eps = eps[:args.limit]

  print(f"🔍 Batch checking {len(eps)} episodes from {depth_abs}")

  results = {"pass": [], "fail": []}
  for ep in eps:
    args.episode_id = ep
    try:
      ok = cmd_check(args)
      results["pass" if ok else "fail"].append(ep)
    except Exception as e:
      print(f"  ❌ {ep}: exception: {e}")
      results["fail"].append(ep)

  print(f"\n{'='*60}")
  print(f"  BATCH SUMMARY: {len(results['pass'])} pass, "
        f"{len(results['fail'])} fail out of {len(eps)}")
  if results["fail"]:
    print(f"  Failed: {results['fail'][:10]}{'...' if len(results['fail']) > 10 else ''}")
  print(f"{'='*60}\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="DROID Pipeline Output Verification Tool",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )
  subparsers = parser.add_subparsers(dest="command", help="Command to run")

  # Common args
  common = argparse.ArgumentParser(add_help=False)
  common.add_argument("--depth_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/depth")
  common.add_argument("--extrinsics_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/extrinsics")
  common.add_argument("--tracks_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/tracks")

  # check
  p_check = subparsers.add_parser("check", parents=[common],
                                  help="Verify output file structure")
  p_check.add_argument("--episode_id", type=str, required=True)

  # snapshot
  p_snap = subparsers.add_parser("snapshot", parents=[common],
                                 help="Save golden output snapshot")
  p_snap.add_argument("--episode_id", type=str, required=True)
  p_snap.add_argument("--snapshot_dir", type=str, required=True)

  # compare
  p_cmp = subparsers.add_parser("compare", parents=[common],
                                help="Compare against golden snapshot")
  p_cmp.add_argument("--episode_id", type=str, required=True)
  p_cmp.add_argument("--snapshot_dir", type=str, required=True)
  p_cmp.add_argument("--atol", type=float, default=0.0,
                     help="Absolute tolerance for numpy comparison "
                          "(0.0 = bit-identical)")

  # batch_check
  p_batch = subparsers.add_parser("batch_check", parents=[common],
                                  help="Check structure of all episodes")
  p_batch.add_argument("--limit", type=int, default=-1)

  args = parser.parse_args()

  if args.command == "check":
    cmd_check(args)
  elif args.command == "snapshot":
    cmd_snapshot(args)
  elif args.command == "compare":
    cmd_compare(args)
  elif args.command == "batch_check":
    cmd_batch_check(args)
  else:
    parser.print_help()
