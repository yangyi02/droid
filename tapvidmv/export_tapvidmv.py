import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
import core.io
import core.runner

config = get_config()


def read_episode_list(path):
  path = os.path.abspath(os.path.expanduser(path))
  with open(path) as f:
    return {line.split("#")[0].strip() for line in f if line.split("#")[0].strip()}


def _encode_jpeg(rgb_frame, quality=95):
  bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
  ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
  return np.frombuffer(buf, dtype=np.uint8).copy()


def _build_queries(uv, query_view):
  """Each point is queried at t=0 in the view that seeded it, on that view's exact pixel."""
  xy = np.round(uv[query_view, 0, np.arange(uv.shape[2])])
  t = np.zeros((len(xy), 1), dtype=np.float32)
  return np.concatenate([xy, t, query_view[:, None]], axis=1).astype(np.float32)


def export_to_tapvid3d(
  episode,
  poses,
  tracks_3d,
  uv,
  vis,
  query_view,
  output_root=config.paths.tapvidmv,
  include_depth=True,
  include_foreground_mask=True,
  jpeg_quality=95,
):
  episode_id = episode["meta"]["episode_id"]
  wrist_cam_id = episode["meta"]["wrist_serial"]
  cam_ids = list(episode["camera"])
  F = tracks_3d.shape[0]

  print(f"\nExporting episode [{episode_id}] to TAPVid-3D format")
  print(f"  Views: {cam_ids} | Frames: {F} | Points: {tracks_3d.shape[1]}")

  P = tracks_3d.shape[1]

  seq_dir = os.path.abspath(os.path.expanduser(os.path.join(output_root, episode_id)))
  os.makedirs(seq_dir, exist_ok=True)

  np.save(os.path.join(seq_dir, "tracks_xyz.npy"), tracks_3d.astype(np.float32))
  print(f"  tracks_xyz.npy: ({F}, {P}, 3)")

  queries = _build_queries(uv, query_view)
  np.save(os.path.join(seq_dir, "queries_xytv.npy"), queries)
  print(f"  queries_xytv.npy: ({P}, 4)")

  for view, cam_id in enumerate(cam_ids):
    view_id = str(view)
    view_dir = os.path.join(seq_dir, view_id)
    os.makedirs(view_dir, exist_ok=True)

    cam_data = episode["camera"][cam_id]

    video = cam_data["video_rgb"]
    jpeg_list = []
    for t in range(F):
      jpeg_list.append(_encode_jpeg(video[t], quality=jpeg_quality))
    jpeg_arr = np.empty(F, dtype=object)
    jpeg_arr[:] = jpeg_list
    np.save(os.path.join(view_dir, "images_jpeg_bytes.npy"), jpeg_arr)

    K = cam_data["K"]
    intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
    np.save(os.path.join(view_dir, "intrinsics.npy"), intrinsics)

    c2w = poses[cam_id]["extrinsics"]
    w2c = np.linalg.inv(c2w).astype(np.float32)
    np.save(os.path.join(view_dir, "extrinsics_w2c.npy"), w2c)

    np.save(os.path.join(view_dir, "visibility.npy"), vis[view].astype(bool))

    if include_depth and "raw_depth" in cam_data:
      depth = cam_data["raw_depth"].astype(np.float32)
      depth[~np.isfinite(depth)] = 0.0
      np.save(os.path.join(view_dir, "depth.npy"), depth)

    if include_foreground_mask and cam_id == wrist_cam_id:
      mask = cam_data["sam_real_masks"].astype(bool)
      np.save(os.path.join(view_dir, "foreground_mask.npy"), mask)

    H, W = video[0].shape[:2]
    parts = [f"  view {view_id} [{cam_id}]: imgs({F},JPEG) intr(4,) extr({F},4,4) vis({F},{P})"]
    if include_depth and "raw_depth" in cam_data:
      parts.append(f" depth({F},{H},{W})")
    if include_foreground_mask and cam_id == wrist_cam_id:
      parts.append(f" fg_mask({F},{H},{W})")
    print("".join(parts))

  print(f"\n  TAPVid-3D export complete → {seq_dir}")
  return seq_dir


def process_episode(episode_id, args):
  print(f"\nLoading pipeline outputs for [{episode_id}]...")
  episode = core.io.load_depth_data(episode_id, args.depth_root, load_video=True)
  poses = core.io.load_extrinsics(episode, args.extrinsics_root)

  tracks = core.io.load_track_data(episode_id, args.tracks_root)

  export_to_tapvid3d(
    episode=episode,
    poses=poses,
    tracks_3d=tracks["tracks_3d"],
    uv=tracks["uv"],
    vis=tracks["vis"],
    query_view=tracks["query_view"],
    output_root=args.output_root,
    include_depth=not args.no_depth,
    include_foreground_mask=not args.no_foreground_mask,
    jpeg_quality=args.jpeg_quality,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Export DROID pipeline outputs to TAPVid-3D format")
  parser.add_argument("--rank", type=int, default=0)
  parser.add_argument("--world_size", type=int, default=1)
  parser.add_argument("--limit", type=int, default=-1)
  parser.add_argument(
    "--episode_id",
    type=str,
    default=None,
    help="Process a single episode (overrides everything else)",
  )
  parser.add_argument(
    "--episode_list",
    type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "episodes_eval50.txt"),
    help="File of episode ids, one per line, to export: the "
    "selected release set. Exporting is the expensive "
    "step, so it runs after selection rather than over "
    "everything. Pass 'all' to export every episode "
    "that has tracks instead",
  )
  parser.add_argument("--output_root", type=str, default=config.paths.tapvidmv, help="Root output directory")
  parser.add_argument("--depth_root", type=str, default=config.paths.depth)
  parser.add_argument("--extrinsics_root", type=str, default=config.paths.extrinsics)
  parser.add_argument("--tracks_root", type=str, default=config.paths.tracks)
  parser.add_argument("--no_depth", action="store_true", help="Skip depth.npy export")
  parser.add_argument("--no_foreground_mask", action="store_true", help="Skip foreground_mask.npy export")
  parser.add_argument("--jpeg_quality", type=int, default=95)
  args = parser.parse_args()

  print("DROID \u2192 TAPVid-3D Export")

  if args.episode_id:
    process_episode(args.episode_id, args)
  else:
    with_tracks = core.runner.list_episode_dirs(args.tracks_root)
    if args.episode_list == "all":
      available = with_tracks
      print(f"Exporting all {len(available)} episodes with tracks")
    else:
      available = read_episode_list(args.episode_list)
      print(f"Read {len(available)} episodes from {args.episode_list}")

    def run_one(episode_id):
      process_episode(episode_id, args)

    target = core.runner.shard_episodes(available, args.rank, args.world_size, args.limit)
    done = core.runner.list_episode_dirs(args.output_root)

    core.runner.run_episodes(
      target,
      run_one,
      rank=args.rank,
      world_size=args.world_size,
      done=done,
      stage="Export",
    )
