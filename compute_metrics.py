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
import core.io
import core.physics
import core.runner


@torch.no_grad()
def evaluate_extrinsics(episode, poses, device, render_pool, config):
  wrist_cam_id = episode["meta"]["wrist_serial"]
  cam_ids = sorted(episode["camera"])
  pairs = list(itertools.combinations(cam_ids, 2))
  base = {c: torch.tensor(poses[c]["base_extrinsic"], dtype=torch.float32, device=device) for c in cam_ids}

  n_points, max_depth = config.extrinsics.n_points, config.extrinsics.max_depth
  renders = render_pool.render_cameras(episode, poses)
  robot_points, depth_batch, K = core.pointcloud.robot_clouds(
    episode, poses, renders, render_pool.gripper_links, device, n_points
  )
  env, ee_poses, frames = core.pointcloud.scene_clouds(episode, device, n_points, max_depth)

  chamfer, overlap = core.pointcloud.chamfer_overlap(
    env, ee_poses, base, wrist_cam_id, pairs, config.extrinsics.chamfer_match_radius
  )
  chamfer_px, overlap_px = core.pointcloud.chamfer_overlap_px(
    env,
    ee_poses,
    base,
    wrist_cam_id,
    pairs,
    K,
    {c: episode["camera"][c]["baseline"] for c in cam_ids},
    core.pointcloud.disparity_maps(episode, frames, device),
    config.extrinsics.chamfer_px_radius,
  )
  robot = core.pointcloud.robot_depth_loss(robot_points, depth_batch, K, base, max_depth)

  return (
    {f"chamfer_{a}_{b}": chamfer[a, b].item() for a, b in pairs}
    | {f"overlap_{a}_{b}": overlap[a, b].item() * 100 for a, b in pairs}
    | {f"chamfer_px_{a}_{b}": chamfer_px[a, b].item() for a, b in pairs}
    | {f"overlap_px_{a}_{b}": overlap_px[a, b].item() * 100 for a, b in pairs}
    | {f"robot_loss_{cam_id}": robot[cam_id].item() for cam_id in cam_ids}
  )


def scene_metadata(episode):
  site, scene, _ = episode["meta"]["episode_id"].split("+")

  return {
    "site": site,
    "scene": scene,
    "n_cameras": len(episode["camera"]),
    "n_frames": int(len(episode["robot"]["joint_positions"])),
    "wrist_serial": episode["meta"]["wrist_serial"],
  }


def episode_metrics(episode, poses, device, render_pool, config):
  return (
    {"episode_id": episode["meta"]["episode_id"]}
    | scene_metadata(episode)
    | evaluate_extrinsics(episode, poses, device, render_pool, config)
  )


def process_episode(episode_id, device, render_pool, config):
  t0 = time.time()
  episode = core.io.load_depth_data(episode_id, config.paths.depth)
  poses = core.io.load_extrinsics(episode, config.paths.extrinsics)

  metrics = episode_metrics(episode, poses, device, render_pool, config)
  payload = {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in metrics.items()}

  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(config.paths.metrics, episode_id)))
  os.makedirs(ep_dir, exist_ok=True)
  with open(os.path.join(ep_dir, "metrics.json"), "w") as f:
    json.dump(payload, f, indent=2)

  print(f"  [OK] Done in {time.time() - t0:.1f}s")


def main(_):
  config = config_flag.value
  export_root = os.path.abspath(os.path.expanduser(config.paths.metrics))

  available = core.runner.list_episode_dirs(config.paths.extrinsics)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  render_pool = core.physics.RenderPool(
    core.physics.PyBulletRenderer(config.paths.urdf, gpu=config.render.gpu), config.render.workers
  )

  def run_one(episode_id):
    process_episode(episode_id, device, render_pool, config)

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
