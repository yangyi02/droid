import argparse
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
import release
import core.io
import core.runner
import core.tracks

config = get_config()

RELEASE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "release")


def _encode_jpeg(rgb_frame, quality=95):
  bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
  _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
  return np.frombuffer(buf, dtype=np.uint8).copy()


def export_to_tapvid3d(
  episode,
  poses,
  tracks_3d,
  uv,
  vis,
  masks,
  query_view,
  query_frame,
  output_root=RELEASE_ROOT,
  jpeg_quality=95,
):
  episode_id = episode["meta"]["episode_id"]
  cam_ids = list(episode["camera"])
  n_frames, n_points = tracks_3d.shape[:2]

  print(f"\nExporting episode [{episode_id}] to TAPVid-3D format")
  print(f"  Views: {cam_ids} | Frames: {n_frames} | Points: {n_points}")

  seq_dir = os.path.abspath(os.path.expanduser(os.path.join(output_root, episode_id)))
  staging = seq_dir + ".partial"
  shutil.rmtree(staging, ignore_errors=True)
  os.makedirs(staging)

  np.save(os.path.join(staging, release.TRACKS), tracks_3d.astype(np.float32))
  print(f"  tracks_xyz.npy: ({n_frames}, {n_points}, 3)")

  queries = core.tracks.build_queries(uv, query_view, query_frame)
  np.save(os.path.join(staging, release.QUERIES), queries)
  print(f"  queries_xytv.npy: ({n_points}, 4)")

  for view, cam_id in enumerate(cam_ids):
    view_id = str(view)
    view_dir = os.path.join(staging, view_id)
    os.makedirs(view_dir, exist_ok=True)

    cam_data = episode["camera"][cam_id]

    video = cam_data["video_rgb"]
    jpeg_arr = np.empty(n_frames, dtype=object)
    jpeg_arr[:] = [_encode_jpeg(video[t], quality=jpeg_quality) for t in range(n_frames)]
    np.save(os.path.join(view_dir, release.IMAGES), jpeg_arr)

    K = cam_data["K"]
    intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
    np.save(os.path.join(view_dir, release.INTRINSICS), intrinsics)

    T_cam2world = poses[cam_id]["T_cam2world"]
    T_world2cam = np.linalg.inv(T_cam2world).astype(np.float32)
    np.save(os.path.join(view_dir, release.EXTRINSICS), T_world2cam)

    np.save(os.path.join(view_dir, release.VISIBILITY), vis[view].astype(bool))

    depth_mm = np.round(cam_data["raw_depth"] * 1000).astype(np.uint16)
    np.savez_compressed(os.path.join(view_dir, release.DEPTH), depth=depth_mm)
    np.savez_compressed(os.path.join(view_dir, release.MASK), mask=masks[cam_id])

    height, width = video[0].shape[:2]
    print(
      f"  view {view_id} [{cam_id}]: imgs({n_frames},JPEG) intr(4,)"
      f" extr({n_frames},4,4) vis({n_frames},{n_points}) depth+mask({n_frames},{height},{width})"
    )

  shutil.rmtree(seq_dir, ignore_errors=True)
  os.rename(staging, seq_dir)

  print(f"\n  TAPVid-3D export complete → {seq_dir}")


def process_episode(episode_id, args):
  print(f"\nLoading pipeline outputs for [{episode_id}]...")
  episode = core.io.load_depth_data(episode_id, args.depth_root, load_video=True)
  poses = core.io.load_extrinsics(episode, args.extrinsics_root)

  tracks = core.io.load_track_data(episode_id, args.tracks_root)
  tracks = core.tracks.sample_tracks(tracks, args.points_per_class)
  uv = core.tracks.project_views(tracks["tracks_3d"], episode, poses)

  export_to_tapvid3d(
    episode=episode,
    poses=poses,
    tracks_3d=tracks["tracks_3d"],
    uv=uv,
    vis=tracks["vis"],
    masks=core.io.load_robot_masks(episode_id, args.tracks_root, episode["camera"]),
    query_view=tracks["query_view"],
    query_frame=tracks["query_frame"],
    output_root=args.output_root,
    jpeg_quality=args.jpeg_quality,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Export DROID pipeline outputs to TAPVid-3D format")
  parser.add_argument("--rank", type=int, default=0)
  parser.add_argument("--world_size", type=int, default=1)
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
  parser.add_argument("--output_root", type=str, default=RELEASE_ROOT, help="Root output directory")
  parser.add_argument("--depth_root", type=str, default=config.paths.depth)
  parser.add_argument("--extrinsics_root", type=str, default=config.paths.extrinsics)
  parser.add_argument("--tracks_root", type=str, default=config.paths.tracks)
  parser.add_argument("--points_per_class", type=int, default=config.tracks.points_per_class)
  parser.add_argument("--jpeg_quality", type=int, default=95)
  args = parser.parse_args()

  print("DROID \u2192 TAPVid-3D Export")

  if args.episode_id:
    process_episode(args.episode_id, args)
  else:
    tracks_root = os.path.abspath(os.path.expanduser(args.tracks_root))
    with_tracks = {
      e
      for e in core.runner.list_episode_dirs(tracks_root)
      if os.path.exists(os.path.join(tracks_root, e, "tracks.npz"))
    }
    if args.episode_list == "all":
      available = with_tracks
      print(f"Exporting all {len(available)} episodes with tracks")
    else:
      with open(os.path.abspath(os.path.expanduser(args.episode_list))) as f:
        available = {line.split("#")[0].strip() for line in f if line.split("#")[0].strip()}
      print(f"Read {len(available)} episodes from {args.episode_list}")

    def run_one(episode_id):
      process_episode(episode_id, args)

    config.runner.rank, config.runner.world_size = args.rank, args.world_size
    target = core.runner.select_episodes(available, config)
    done = core.runner.list_episode_dirs(args.output_root)

    core.runner.run_episodes(target, run_one, done=done, stage="Export")
