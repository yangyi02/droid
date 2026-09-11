import itertools
import json
import os
import time

import numpy as np
import torch
from absl import app
from ml_collections import config_flags

import config
import core.pointcloud
import core.geometry
import core.io
import core.physics
import core.runner


@torch.no_grad()
def evaluate_extrinsics(episode, poses, device, pb_renderer, config):
  wrist_cam_id = episode["meta"]["wrist_serial"]
  cam_ids = sorted(episode["camera"])
  pairs = list(itertools.combinations(cam_ids, 2))
  base = {c: torch.tensor(poses[c]["base_extrinsic"], dtype=torch.float32, device=device) for c in cam_ids}

  n_points, max_depth = config.extrinsics.n_points, config.extrinsics.max_depth
  robot_points, depth_batch, K = core.pointcloud.robot_clouds(episode, poses, pb_renderer, device, n_points)
  env, ee_poses = core.pointcloud.scene_clouds(episode, device, n_points, max_depth)

  chamfer, overlap = core.pointcloud.chamfer_overlap(
    env, ee_poses, base, wrist_cam_id, pairs, config.extrinsics.chamfer_match_radius
  )
  robot = core.pointcloud.robot_depth_loss(robot_points, depth_batch, K, base, max_depth)

  return (
    {f"chamfer_{a}_{b}": chamfer[a, b].item() for a, b in pairs}
    | {f"overlap_{a}_{b}": overlap[a, b].item() * 100 for a, b in pairs}
    | {f"robot_loss_{cam_id}": robot[cam_id].item() for cam_id in cam_ids}
  )


def depth_residual_mm(points_3d, cam_data, extrinsics, t):
  u, v, z = core.geometry.project_points(points_3d, cam_data["K"], extrinsics)
  z_obs = core.geometry.sample_depth(cam_data["raw_depth"][t], u, v, z)
  gap = np.abs(z_obs - z) * 1000.0
  return gap[np.isfinite(gap) & (z_obs > 0)]


def depth_residual(episode, poses, tracks):
  tracks_3d, n_static = tracks["tracks_3d"], tracks["n_static"]
  per_cam_vis = dict(zip(episode["camera"], tracks["vis"], strict=True))

  residual = {}
  for cam_id, cam_data in episode["camera"].items():
    static, robot = [], []
    for t in range(len(tracks_3d)):
      visible, extrinsics = per_cam_vis[cam_id][t], poses[cam_id]["extrinsics"][t]
      static.append(depth_residual_mm(tracks_3d[t, :n_static][visible[:n_static]], cam_data, extrinsics, t))
      robot.append(depth_residual_mm(tracks_3d[t, n_static:][visible[n_static:]], cam_data, extrinsics, t))
    residual[cam_id] = {"static": np.concatenate(static), "robot": np.concatenate(robot)}

  return residual


def mean_residual(residual):
  return {
    f"depth_residual_{kind}_mm_{cam_id}": float(np.mean(gaps[kind])) if len(gaps[kind]) else float("nan")
    for cam_id, gaps in residual.items()
    for kind in ("static", "robot")
  }


def scene_metadata(episode):
  site, scene, _ = episode["meta"]["episode_id"].split("+")

  return {
    "site": site,
    "scene": scene,
    "n_cameras": len(episode["camera"]),
    "n_frames": int(len(episode["robot"]["joint_positions"])),
    "wrist_serial": episode["meta"]["wrist_serial"],
  }


def episode_metrics(episode, poses, device, tracks, pb_renderer, config):
  return (
    {"episode_id": episode["meta"]["episode_id"]}
    | scene_metadata(episode)
    | evaluate_extrinsics(episode, poses, device, pb_renderer, config)
    | mean_residual(depth_residual(episode, poses, tracks))
  )


def process_episode(episode_id, device, pb_renderer, config):
  t0 = time.time()
  episode = core.io.load_depth_data(episode_id, config.paths.depth)
  poses = core.io.load_extrinsics(episode, config.paths.extrinsics)
  tracks = core.io.load_track_data(episode_id, config.paths.tracks)

  metrics = episode_metrics(episode, poses, device, tracks, pb_renderer, config)
  payload = {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in metrics.items()}

  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(config.paths.metrics, episode_id)))
  os.makedirs(ep_dir, exist_ok=True)
  with open(os.path.join(ep_dir, "metrics.json"), "w") as f:
    json.dump(payload, f, indent=2)

  print(f"  [OK] Done in {time.time() - t0:.1f}s")


def main(_):
  config = config_flag.value
  export_root = os.path.abspath(os.path.expanduser(config.paths.metrics))

  available = core.runner.list_episode_dirs(config.paths.tracks)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  pb_renderer = core.physics.PyBulletRenderer(config.paths.urdf, gpu=config.render.gpu)

  def run_one(episode_id):
    process_episode(episode_id, device, pb_renderer, config)

  target = core.runner.shard_episodes(available, config.runner.rank, config.runner.world_size, config.runner.limit)
  done = {e for e in target if os.path.exists(os.path.join(export_root, e, "metrics.json"))}

  core.runner.run_episodes(
    target,
    run_one,
    rank=config.runner.rank,
    world_size=config.runner.world_size,
    done=done,
    stage="Evaluation",
  )


if __name__ == "__main__":
  config_flag = config_flags.DEFINE_config_file("config", config.__file__)
  app.run(main)
