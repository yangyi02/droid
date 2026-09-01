"""Export DROID pipeline outputs to the multi-view TAPVid-3D raw file format.

Converts the internal pipeline representation (scene_constants, scene_state,
tracks) to the canonical directory layout:

  <split>/<sequence>/
  ├── tracks_xyz.npy           (F, P, 3) float32 — world-space 3D
  ├── queries_xytv.npy         (P, 4) float32 — (x, y, t, v) per track
  └── <view_id>/
      ├── images_jpeg_bytes.npy  (F,) object — JPEG-encoded uint8 arrays
      ├── intrinsics.npy         (4,) float32 — (fx, fy, cx, cy)
      ├── extrinsics_w2c.npy     (F, 4, 4) float32 — world-to-camera
      ├── visibility.npy         (F, P) bool
      ├── depth.npy              (F, H, W) float32 — optional
      └── foreground_mask.npy    (F, H, W) bool — optional

Pixel centre convention: integer-centre (pixel (0,0) = centre of top-left
pixel). Projection: x = fx * X/Z + cx, y = fy * Y/Z + cy.

Usage (standalone):
  python tapvidmv/export_tapvidmv.py \\
      --episode_id "ILIAD+5e938e3b+2023-07-20" \\
      --output_root data/output/mv-tap/droid/tapvidmv   # default; repo-relative

Usage (from pipeline.ipynb / Python):
  from tapvidmv.export_tapvidmv import export_to_tapvid3d
  export_to_tapvid3d(
      scene_constants, scene_state,
      final_traj_3d, final_per_cam_tracks, final_per_cam_vis,
      output_root=os.path.join(OUTPUT_ROOT, "tapvidmv"),
      include_depth=True, include_foreground_mask=True)
"""

import argparse
import os
import sys

import cv2
import numpy as np

# tapvidmv/ sits one level below the repo root, so running this file directly
# puts tapvidmv/ — not the repo root — on sys.path. Prepend the repo root so
# `core` resolves the same way it does for the top-level pipeline scripts.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.io import OUTPUT_ROOT, load_depth_data, load_extrinsics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_jpeg(rgb_frame, quality=95):
  """Encode a single (H, W, 3) uint8 RGB frame to JPEG bytes as uint8 array."""
  bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
  ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
  if not ok:
    raise RuntimeError("JPEG encoding failed")
  return np.frombuffer(buf, dtype=np.uint8).copy()


def _sample_queries(per_cam_vis, per_cam_tracks, view_index_map, seed=42):
  """Sample one (x, y, t, v) query per track from visible (frame, view) pairs.

  Args:
    per_cam_vis:    {cam_id: (F, P) bool}
    per_cam_tracks: {cam_id: (F, P, 2) float32}
    view_index_map: {cam_id: int} — maps cam serial to 0-indexed view index.
    seed: RNG seed for reproducibility.

  Returns:
    queries_xytv: (P, 4) float32
  """
  cam_ids = list(view_index_map.keys())
  P = per_cam_vis[cam_ids[0]].shape[1]
  rng = np.random.default_rng(seed)

  queries = np.zeros((P, 4), dtype=np.float32)

  for p in range(P):
    # Collect all (frame, view) where this track is visible.
    candidates = []
    for cam_id in cam_ids:
      vis_p = per_cam_vis[cam_id][:, p]  # (F,)
      visible_frames = np.where(vis_p)[0]
      v_idx = view_index_map[cam_id]
      for t in visible_frames:
        candidates.append((t, v_idx, cam_id))

    if len(candidates) == 0:
      # Should not happen per spec (tracks with no visibility are excluded),
      # but handle gracefully: place a sentinel.
      queries[p] = [-1, -1, 0, 0]
      continue

    # Uniform random selection.
    chosen = candidates[rng.integers(len(candidates))]
    t, v_idx, cam_id = chosen
    x, y = per_cam_tracks[cam_id][t, p]
    queries[p] = [x, y, float(t), float(v_idx)]

  return queries


def _filter_always_invisible_tracks(
    final_traj_3d, per_cam_tracks, per_cam_vis, cam_ids
):
  """Remove tracks that are never visible in any view at any frame.

  Returns filtered copies of traj_3d, per_cam_tracks, per_cam_vis, and a
  boolean mask of kept track indices.
  """
  F, P, _ = final_traj_3d.shape
  any_visible = np.zeros(P, dtype=bool)
  for cam_id in cam_ids:
    any_visible |= per_cam_vis[cam_id].any(axis=0)  # (P,)

  if any_visible.all():
    return final_traj_3d, per_cam_tracks, per_cam_vis, any_visible

  n_kept = any_visible.sum()
  print(f"  Filtering tracks: {P} → {n_kept} "
        f"({P - n_kept} never-visible tracks removed)")
  filtered_traj = final_traj_3d[:, any_visible, :]
  filtered_tracks = {c: per_cam_tracks[c][:, any_visible, :]
                     for c in cam_ids}
  filtered_vis = {c: per_cam_vis[c][:, any_visible] for c in cam_ids}
  return filtered_traj, filtered_tracks, filtered_vis, any_visible


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_to_tapvid3d(
    scene_constants,
    scene_state,
    final_traj_3d,
    final_per_cam_tracks,
    final_per_cam_vis,
    output_root=os.path.join(OUTPUT_ROOT, "tapvidmv"),
    include_depth=True,
    include_foreground_mask=True,
    jpeg_quality=95,
    query_seed=42,
):
  """Convert DROID pipeline outputs to the TAPVid-3D multi-view file format.

  Args:
    scene_constants: Pipeline scene data dict (see core/io.py).
    scene_state: Per-camera extrinsics dict.
        scene_state[cam_id]['extrinsics'] is (F, 4, 4) camera-to-world.
    final_traj_3d: (F, P, 3) float32, 3D world-space trajectories.
    final_per_cam_tracks: {cam_id: (F, P, 2) float32} per-view 2D tracks.
    final_per_cam_vis: {cam_id: (F, P) bool} per-view visibility.
    output_root: Root directory for output.
    include_depth: Whether to export depth.npy per view.
    include_foreground_mask: Whether to export foreground_mask.npy per view
        (only for wrist camera where SAM masks are available).
    jpeg_quality: JPEG encoding quality (1–100).
    query_seed: RNG seed for query sampling (spec says 42 for DROID).

  Returns:
    seq_dir: Absolute path to the created sequence directory.
  """
  episode_id = scene_constants["meta"]["episode_id"]
  wrist_serial = scene_constants["meta"].get("wrist_serial")
  cam_ids = sorted(scene_constants["camera"].keys())
  F = final_traj_3d.shape[0]

  # Assign deterministic 0-indexed view IDs (sorted cam serial order).
  view_index_map = {cam_id: i for i, cam_id in enumerate(cam_ids)}

  print(f"\nExporting episode [{episode_id}] to TAPVid-3D format")
  print(f"  Views: {len(cam_ids)} | Frames: {F} | "
        f"Points: {final_traj_3d.shape[1]}")
  print(f"  View index map: {view_index_map}")

  # --- Filter never-visible tracks ---
  traj_3d, cam_tracks, cam_vis, _ = _filter_always_invisible_tracks(
      final_traj_3d, final_per_cam_tracks, final_per_cam_vis, cam_ids)
  P = traj_3d.shape[1]

  # --- Output directory ---
  seq_dir = os.path.abspath(os.path.expanduser(
      os.path.join(output_root, episode_id)))
  os.makedirs(seq_dir, exist_ok=True)

  # --- Shared files (sequence level) ---

  # tracks_xyz.npy: (F, P, 3) float32
  np.save(os.path.join(seq_dir, "tracks_xyz.npy"),
          traj_3d.astype(np.float32))
  print(f"  tracks_xyz.npy: ({F}, {P}, 3)")

  # queries_xytv.npy: (P, 4) float32
  queries = _sample_queries(cam_vis, cam_tracks, view_index_map,
                            seed=query_seed)
  np.save(os.path.join(seq_dir, "queries_xytv.npy"), queries)
  print(f"  queries_xytv.npy: ({P}, 4)")

  # --- Per-view files ---
  for cam_id in cam_ids:
    view_id = str(view_index_map[cam_id])
    view_dir = os.path.join(seq_dir, view_id)
    os.makedirs(view_dir, exist_ok=True)

    cam_data = scene_constants["camera"][cam_id]

    # images_jpeg_bytes.npy: (F,) object array of 1-D uint8 arrays
    video = cam_data["video_rgb"]  # (F, H, W, 3) uint8
    jpeg_list = []
    for t in range(F):
      jpeg_list.append(_encode_jpeg(video[t], quality=jpeg_quality))
    jpeg_arr = np.empty(F, dtype=object)
    jpeg_arr[:] = jpeg_list
    np.save(os.path.join(view_dir, "images_jpeg_bytes.npy"), jpeg_arr)

    # intrinsics.npy: (4,) float32 — (fx, fy, cx, cy)
    K = cam_data["K_mat"]  # (3, 3)
    intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                          dtype=np.float32)
    np.save(os.path.join(view_dir, "intrinsics.npy"), intrinsics)

    # extrinsics_w2c.npy: (F, 4, 4) float32 — world-to-camera
    # Pipeline stores camera-to-world; invert to get w2c.
    c2w = scene_state[cam_id]["extrinsics"]  # (F, 4, 4)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    np.save(os.path.join(view_dir, "extrinsics_w2c.npy"), w2c)

    # visibility.npy: (F, P) bool
    np.save(os.path.join(view_dir, "visibility.npy"),
            cam_vis[cam_id].astype(bool))

    # depth.npy (optional): (F, H, W) float32
    if include_depth and "raw_depth" in cam_data:
      depth = cam_data["raw_depth"].astype(np.float32)
      # Ensure invalid values are zero (pipeline uses 0 for missing).
      depth[~np.isfinite(depth)] = 0.0
      np.save(os.path.join(view_dir, "depth.npy"), depth)

    # foreground_mask.npy (optional): (F, H, W) bool
    # Only the wrist camera has SAM gripper masks in the pipeline.
    if (include_foreground_mask
        and cam_id == wrist_serial
        and "sam_real_masks" in cam_data):
      mask = cam_data["sam_real_masks"].astype(bool)
      np.save(os.path.join(view_dir, "foreground_mask.npy"), mask)

    H, W = video[0].shape[:2]
    parts = [f"  view {view_id} [{cam_id}]: "
             f"imgs({F},JPEG) intr(4,) extr({F},4,4) vis({F},{P})"]
    if include_depth and "raw_depth" in cam_data:
      parts.append(f" depth({F},{H},{W})")
    if (include_foreground_mask and cam_id == wrist_serial
        and "sam_real_masks" in cam_data):
      parts.append(f" fg_mask({F},{H},{W})")
    print("".join(parts))

  print(f"\n  TAPVid-3D export complete → {seq_dir}")
  return seq_dir


# ---------------------------------------------------------------------------
# Single-episode processing (used by batch CLI)
# ---------------------------------------------------------------------------

def process_episode(episode_id, args):
  """Load pipeline outputs and export one episode to TAPVid-3D format."""
  print(f"\nLoading pipeline outputs for [{episode_id}]...")
  scene_constants = load_depth_data(
      episode_id, args.depth_root, load_video="full")
  scene_state = load_extrinsics(scene_constants, args.extrinsics_root)

  # Load tracks.
  tracks_dir = os.path.abspath(os.path.expanduser(
      os.path.join(args.tracks_root, episode_id)))
  data_3d = np.load(os.path.join(tracks_dir, "tracks_3d.npz"))
  final_traj_3d = data_3d["traj_3d"]  # (F, P, 3)

  cam_ids = sorted(scene_constants["camera"].keys())
  final_per_cam_tracks = {}
  final_per_cam_vis = {}
  for cam_id in cam_ids:
    d = np.load(os.path.join(tracks_dir, cam_id, "tracks_2d.npz"))
    final_per_cam_tracks[cam_id] = d["traj_2d"]
    final_per_cam_vis[cam_id] = d["vis_2d"]

  # Export.
  export_to_tapvid3d(
      scene_constants=scene_constants,
      scene_state=scene_state,
      final_traj_3d=final_traj_3d,
      final_per_cam_tracks=final_per_cam_tracks,
      final_per_cam_vis=final_per_cam_vis,
      output_root=args.output_root,
      include_depth=not args.no_depth,
      include_foreground_mask=not args.no_foreground_mask,
      jpeg_quality=args.jpeg_quality,
      query_seed=args.query_seed,
  )


# ---------------------------------------------------------------------------
# Standalone CLI (supports rank/world_size parallel sharding)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  import random
  import traceback

  parser = argparse.ArgumentParser(
      description="Export DROID pipeline outputs to TAPVid-3D format")
  # Parallel sharding args (same as compute_tracks.py / compute_extrinsics.py)
  parser.add_argument("--rank", type=int, default=0,
                      help="Rank of the process (for multi-worker sharding)")
  parser.add_argument("--world_size", type=int, default=1,
                      help="Total number of parallel workers")
  parser.add_argument("--limit", type=int, default=-1,
                      help="Limit total number of episodes to process")
  # Single-episode mode (optional, overrides discovery)
  parser.add_argument("--episode_id", type=str, default=None,
                      help="Process a single episode (overrides discovery)")
  # Paths
  parser.add_argument("--output_root", type=str,
                      default=os.path.join(OUTPUT_ROOT, "tapvidmv"),
                      help="Root output directory")
  parser.add_argument("--depth_root", type=str,
                      default=os.path.join(OUTPUT_ROOT, "depth"))
  parser.add_argument("--extrinsics_root", type=str,
                      default=os.path.join(OUTPUT_ROOT, "extrinsics"))
  parser.add_argument("--tracks_root", type=str,
                      default=os.path.join(OUTPUT_ROOT, "tracks"))
  # Options
  parser.add_argument("--no_depth", action="store_true",
                      help="Skip depth.npy export")
  parser.add_argument("--no_foreground_mask", action="store_true",
                      help="Skip foreground_mask.npy export")
  parser.add_argument("--jpeg_quality", type=int, default=95)
  parser.add_argument("--query_seed", type=int, default=42)
  args = parser.parse_args()

  print("DROID → TAPVid-3D Export")

  if args.episode_id:
    # Single-episode mode
    process_episode(args.episode_id, args)
  else:
    # Batch mode: discover episodes from tracks output
    # NOTE: avoid per-entry os.path.isdir / os.path.exists here — on gcsfuse
    # mounts each call is a GCS API request and thousands of them stall.
    # Instead, just list directory names and handle missing files in
    # process_episode.
    tracks_abs = os.path.abspath(os.path.expanduser(args.tracks_root))
    output_abs = os.path.abspath(os.path.expanduser(args.output_root))
    available_eps = sorted(os.listdir(tracks_abs))
    print(f"Discovered {len(available_eps)} episodes in {tracks_abs}")

    # Deterministic shuffle for load balancing across ranks
    random.seed(42)
    random.shuffle(available_eps)

    if args.limit > 0:
      available_eps = available_eps[:args.limit]

    # Shard across ranks (shard BEFORE resume check to avoid slow stat calls
    # on episodes assigned to other ranks)
    target_eps = available_eps[args.rank::args.world_size]

    # Skip episodes that already have output (resume-friendly).
    # Use a single listdir instead of per-episode os.path.exists (gcsfuse).
    done_dir = output_abs
    if os.path.isdir(done_dir):
      done_eps = set(os.listdir(done_dir))
    else:
      done_eps = set()
    todo_eps = [ep for ep in target_eps if ep not in done_eps]

    print(f"Rank {args.rank}/{args.world_size}: "
          f"{len(todo_eps)} episodes to export "
          f"({len(target_eps) - len(todo_eps)} already done)")

    succeeded_eps = []

    for idx, ep_id in enumerate(todo_eps):
      print(f"\n[{idx + 1}/{len(todo_eps)}] Episode: {ep_id}")
      try:
        process_episode(ep_id, args)
        succeeded_eps.append(ep_id)
      except Exception as e:
        print(f"  [FAIL] Episode {ep_id} failed: {e}")
        traceback.print_exc()

    print(f"\nExport complete! "
          f"{len(succeeded_eps)}/{len(todo_eps)} episodes succeeded.")
