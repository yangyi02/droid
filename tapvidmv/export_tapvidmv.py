import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.io import OUTPUT_ROOT, load_depth_data, load_extrinsics
from core.runner import (add_sharding_args, list_episode_dirs,
                         run_episodes, shard_episodes)


def read_episode_list(path):
  path = os.path.abspath(os.path.expanduser(path))
  with open(path) as f:
    return {line.split("#")[0].strip() for line in f
            if line.split("#")[0].strip()}


def _encode_jpeg(rgb_frame, quality=95):
  bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
  ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
  return np.frombuffer(buf, dtype=np.uint8).copy()


def _sample_queries(per_cam_vis, per_cam_tracks, view_index_map, seed=42):
  cam_ids = list(view_index_map.keys())
  P = per_cam_vis[cam_ids[0]].shape[1]
  rng = np.random.default_rng(seed)

  queries = np.zeros((P, 4), dtype=np.float32)

  for p in range(P):
    candidates = []
    for cam_id in cam_ids:
      vis_p = per_cam_vis[cam_id][:, p]
      visible_frames = np.where(vis_p)[0]
      v_idx = view_index_map[cam_id]
      for t in visible_frames:
        candidates.append((t, v_idx, cam_id))

    if len(candidates) == 0:
      queries[p] = [-1, -1, 0, 0]
      continue

    chosen = candidates[rng.integers(len(candidates))]
    t, v_idx, cam_id = chosen
    x, y = per_cam_tracks[cam_id][t, p]
    queries[p] = [x, y, float(t), float(v_idx)]

  return queries


def _filter_always_invisible_tracks(
    final_traj_3d, per_cam_tracks, per_cam_vis, cam_ids
):
  F, P, _ = final_traj_3d.shape
  any_visible = np.zeros(P, dtype=bool)
  for cam_id in cam_ids:
    any_visible |= per_cam_vis[cam_id].any(axis=0)

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
  episode_id = scene_constants["meta"]["episode_id"]
  wrist_serial = scene_constants["meta"].get("wrist_serial")
  cam_ids = sorted(scene_constants["camera"].keys())
  F = final_traj_3d.shape[0]

  view_index_map = {cam_id: i for i, cam_id in enumerate(cam_ids)}

  print(f"\nExporting episode [{episode_id}] to TAPVid-3D format")
  print(f"  Views: {len(cam_ids)} | Frames: {F} | "
        f"Points: {final_traj_3d.shape[1]}")
  print(f"  View index map: {view_index_map}")

  traj_3d, cam_tracks, cam_vis, _ = _filter_always_invisible_tracks(
      final_traj_3d, final_per_cam_tracks, final_per_cam_vis, cam_ids)
  P = traj_3d.shape[1]

  seq_dir = os.path.abspath(os.path.expanduser(
      os.path.join(output_root, episode_id)))
  os.makedirs(seq_dir, exist_ok=True)


  np.save(os.path.join(seq_dir, "tracks_xyz.npy"),
          traj_3d.astype(np.float32))
  print(f"  tracks_xyz.npy: ({F}, {P}, 3)")

  queries = _sample_queries(cam_vis, cam_tracks, view_index_map,
                            seed=query_seed)
  np.save(os.path.join(seq_dir, "queries_xytv.npy"), queries)
  print(f"  queries_xytv.npy: ({P}, 4)")

  for cam_id in cam_ids:
    view_id = str(view_index_map[cam_id])
    view_dir = os.path.join(seq_dir, view_id)
    os.makedirs(view_dir, exist_ok=True)

    cam_data = scene_constants["camera"][cam_id]

    video = cam_data["video_rgb"]
    jpeg_list = []
    for t in range(F):
      jpeg_list.append(_encode_jpeg(video[t], quality=jpeg_quality))
    jpeg_arr = np.empty(F, dtype=object)
    jpeg_arr[:] = jpeg_list
    np.save(os.path.join(view_dir, "images_jpeg_bytes.npy"), jpeg_arr)

    K = cam_data["K_mat"]
    intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                          dtype=np.float32)
    np.save(os.path.join(view_dir, "intrinsics.npy"), intrinsics)

    c2w = scene_state[cam_id]["extrinsics"]
    w2c = np.linalg.inv(c2w).astype(np.float32)
    np.save(os.path.join(view_dir, "extrinsics_w2c.npy"), w2c)

    np.save(os.path.join(view_dir, "visibility.npy"),
            cam_vis[cam_id].astype(bool))

    if include_depth and "raw_depth" in cam_data:
      depth = cam_data["raw_depth"].astype(np.float32)
      depth[~np.isfinite(depth)] = 0.0
      np.save(os.path.join(view_dir, "depth.npy"), depth)

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


def process_episode(episode_id, args):
  print(f"\nLoading pipeline outputs for [{episode_id}]...")
  scene_constants = load_depth_data(
      episode_id, args.depth_root, load_video="full")
  scene_state = load_extrinsics(scene_constants, args.extrinsics_root)

  tracks_dir = os.path.abspath(os.path.expanduser(
      os.path.join(args.tracks_root, episode_id)))
  data_3d = np.load(os.path.join(tracks_dir, "tracks_3d.npz"))
  final_traj_3d = data_3d["traj_3d"]

  cam_ids = sorted(scene_constants["camera"].keys())
  final_per_cam_tracks = {}
  final_per_cam_vis = {}
  for cam_id in cam_ids:
    d = np.load(os.path.join(tracks_dir, cam_id, "tracks_2d.npz"))
    final_per_cam_tracks[cam_id] = d["traj_2d"]
    final_per_cam_vis[cam_id] = d["vis_2d"]

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


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Export DROID pipeline outputs to TAPVid-3D format")
  add_sharding_args(parser)
  parser.add_argument("--episode_id", type=str, default=None,
                      help="Process a single episode (overrides everything else)")
  parser.add_argument("--episode_list", type=str,
                      default=os.path.join(
                          os.path.dirname(os.path.abspath(__file__)),
                          "episodes_eval50.txt"),
                      help="File of episode ids, one per line, to export: the "
                           "selected release set. Exporting is the expensive "
                           "step, so it runs after selection rather than over "
                           "everything. Pass 'all' to export every episode "
                           "that has tracks instead")
  parser.add_argument("--output_root", type=str,
                      default=os.path.join(OUTPUT_ROOT, "tapvidmv"),
                      help="Root output directory")
  parser.add_argument("--depth_root", type=str,
                      default=os.path.join(OUTPUT_ROOT, "depth"))
  parser.add_argument("--extrinsics_root", type=str,
                      default=os.path.join(OUTPUT_ROOT, "extrinsics"))
  parser.add_argument("--tracks_root", type=str,
                      default=os.path.join(OUTPUT_ROOT, "tracks"))
  parser.add_argument("--no_depth", action="store_true",
                      help="Skip depth.npy export")
  parser.add_argument("--no_foreground_mask", action="store_true",
                      help="Skip foreground_mask.npy export")
  parser.add_argument("--jpeg_quality", type=int, default=95)
  parser.add_argument("--query_seed", type=int, default=42)
  args = parser.parse_args()

  print("DROID \u2192 TAPVid-3D Export")

  if args.episode_id:
    process_episode(args.episode_id, args)
  else:
    with_tracks = list_episode_dirs(args.tracks_root)
    if args.episode_list == "all":
      available = with_tracks
      print(f"Exporting all {len(available)} episodes with tracks")
    else:
      available = read_episode_list(args.episode_list)
      print(f"Read {len(available)} episodes from {args.episode_list}")
    run_episodes(
        shard_episodes(available, args.rank, args.world_size, args.limit),
        lambda ep_id: process_episode(ep_id, args),
        rank=args.rank, world_size=args.world_size,
        done=list_episode_dirs(args.output_root),
        stage="Export")
