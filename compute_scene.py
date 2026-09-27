import os

import numpy as np
import torch
from absl import app
from ml_collections import config_flags

import config
import core.io
import core.physics
import core.runner
import core.scene


def refine(episode, poses, renders, cam_id, device, config):
  cam_data, extrinsics = episode["camera"][cam_id], poses[cam_id]["extrinsics"]
  raw = torch.tensor(cam_data["raw_depth"], device=device)
  segmentation = torch.tensor(renders[cam_id], device=device)
  focal_baseline = float(cam_data["K"][0, 0] * cam_data["baseline"])
  static = core.scene.static_depth(raw, segmentation[:, 0] > 0, cam_data["K"], extrinsics, focal_baseline, config)
  return static, core.scene.refine_depth(raw, static, segmentation, cam_data["K"], extrinsics, focal_baseline, config)


def export_scene(episode, depths, export_root):
  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(export_root, episode["meta"]["episode_id"])))
  for cam_id, (static, refined) in depths.items():
    cam_dir = os.path.join(ep_dir, cam_id)
    os.makedirs(cam_dir, exist_ok=True)
    for name, depth in (("static_depth", static), ("depth", refined)):
      np.savez_compressed(os.path.join(cam_dir, f"{name}.npz"), depth=np.round(depth.cpu().numpy() * 1000).astype(np.uint16))


def process_episode(episode_id, render_pool, device, config):
  episode = core.io.load_depth_data(episode_id, config.paths.depth)
  poses = core.io.load_extrinsics(episode, config.paths.extrinsics)
  renders = render_pool.segment_cameras(episode, poses)
  depths = {cam_id: refine(episode, poses, renders, cam_id, device, config) for cam_id in episode["camera"]}
  export_scene(episode, align_cameras(episode, poses, renders, depths, device, config), config.paths.scene)


def align_cameras(episode, poses, renders, depths, device, config):
  cameras = [
    {
      "static": depths[cam_id][0],
      "raw": torch.tensor(cam_data["raw_depth"], device=device),
      "segmentation": torch.tensor(renders[cam_id], device=device),
      "K": cam_data["K"],
      "extrinsics": poses[cam_id]["extrinsics"],
      "focal_baseline": float(cam_data["K"][0, 0] * cam_data["baseline"]),
    }
    for cam_id, cam_data in episode["camera"].items()
  ]
  offsets = core.scene.depth_offsets(cameras, config)
  return {
    cam_id: tuple(core.scene.shift(depth, camera["focal_baseline"], offset) for depth in depths[cam_id])
    for cam_id, camera, offset in zip(depths, cameras, offsets)
  }


def main(_):
  config = config_flag.value
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  render_pool = core.physics.RenderPool(
    core.physics.PyBulletRenderer(config.paths.urdf, gpu=config.render.gpu), config.render.workers
  )

  available = core.runner.list_episode_dirs(config.paths.extrinsics)
  if config.paths.episode_list:
    available &= core.io.read_episode_list(config.paths.episode_list)

  target = core.runner.shard_episodes(available, config.runner.rank, config.runner.world_size, config.runner.limit)
  export_root = os.path.abspath(os.path.expanduser(config.paths.scene))
  done = {e for e in target if os.path.isdir(os.path.join(export_root, e))}

  def run_one(episode_id):
    process_episode(episode_id, render_pool, device, config)

  core.runner.run_episodes(
    target,
    run_one,
    rank=config.runner.rank,
    world_size=config.runner.world_size,
    done=done,
    stage="Stage 2+",
  )


if __name__ == "__main__":
  config_flag = config_flags.DEFINE_config_file("config", config.__file__)
  app.run(main)
