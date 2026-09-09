#!/usr/bin/env python3
import csv
import fcntl
import glob
import itertools
import os
import time

import numpy as np
import torch
from absl import app
from ml_collections import config_flags

import config
import core.alignment
import core.geometry
import core.io
import core.physics
import core.runner


@torch.no_grad()
def evaluate_extrinsics(episode, poses, device, pb_renderer, config):
  """Stage 2's own objective, read once at the extrinsics it converged to."""
  wrist_cam_id = episode["meta"]["wrist_serial"]
  cam_ids = [c for c in episode["camera"] if c != wrist_cam_id] + [wrist_cam_id]
  pairs = list(itertools.combinations(cam_ids, 2))
  base = {
    c: torch.tensor(poses[c]["base_extrinsic"], dtype=torch.float32, device=device) for c in cam_ids
  }

  n_points, max_depth = config.extrinsics.n_points, config.extrinsics.max_depth
  robot_points, depth_batch, K = core.alignment.robot_clouds(
    episode, poses, pb_renderer, device, n_points
  )
  env, ee_poses = core.alignment.scene_clouds(episode, device, n_points, max_depth)

  chamfer, overlap = core.alignment.chamfer_overlap(
    env, ee_poses, base, wrist_cam_id, pairs, config.extrinsics.chamfer_match_radius
  )
  robot = core.alignment.robot_depth_loss(robot_points, depth_batch, K, base, max_depth)

  fixed_cam_ids = [c for c in cam_ids if c != wrist_cam_id]
  label = {c: str(i + 1) for i, c in enumerate(fixed_cam_ids)} | {wrist_cam_id: "w"}
  suffix = {c: f"cam{i + 1}" for i, c in enumerate(fixed_cam_ids)} | {wrist_cam_id: "wrist"}

  return (
    {f"chamfer_{label[a]}{label[b]}": chamfer[a, b].item() for a, b in pairs}
    | {f"overlap_{label[a]}{label[b]}": overlap[a, b].item() * 100 for a, b in pairs}
    | {f"robot_loss_{suffix[cam_id]}": robot[cam_id].item() for cam_id in cam_ids}
  )


def depth_residual_mm(points_3d, K, extrinsics, raw_depth, width, height):
  u_proj, v_proj, z_proj = core.geometry.project_points(points_3d, K, extrinsics)
  ui = np.clip(np.round(u_proj).astype(int), 0, width - 1)
  vi = np.clip(np.round(v_proj).astype(int), 0, height - 1)
  z_obs = raw_depth[vi, ui]
  valid = (z_obs > 0) & (z_proj > 0)
  return np.abs(z_proj[valid] - z_obs[valid]).astype(np.float32) * 1000.0


def depth_residual_per_camera(episode, poses, tracks_3d, per_cam_vis, n_static):
  cam_ids = list(episode["camera"].keys())
  n_frames = tracks_3d.shape[0]

  per_camera = {}
  for cam_id in cam_ids:
    cam_data = episode["camera"][cam_id]
    K = cam_data["K"]
    height, width = cam_data["raw_depth"][0].shape[:2]

    cam_static, cam_robot = [], []

    for t in range(n_frames):
      raw_depth = cam_data["raw_depth"][t]
      ext = poses[cam_id]["extrinsics"][t]
      vis_t = per_cam_vis[cam_id][t]

      cam_static.append(
        depth_residual_mm(
          tracks_3d[t, :n_static][vis_t[:n_static]], K, ext, raw_depth, width, height
        )
      )
      cam_robot.append(
        depth_residual_mm(
          tracks_3d[t, n_static:][vis_t[n_static:]], K, ext, raw_depth, width, height
        )
      )

    per_camera[cam_id] = {"static": np.concatenate(cam_static), "robot": np.concatenate(cam_robot)}

  return per_camera


def motion_stats(episode):
  robot = episode["robot"]
  joints = robot["joint_positions"]
  gripper = robot["gripper_positions"]

  joint_ranges = joints.max(axis=0) - joints.min(axis=0)
  joint_stds = joints.std(axis=0)

  T_ee2base = robot["T_ee_base_all"]
  ee_positions = T_ee2base[:, :3, 3]
  ee_deltas = np.linalg.norm(np.diff(ee_positions, axis=0), axis=1)
  ee_travel = float(np.sum(ee_deltas))

  return {
    "joint_range_mean_rad": float(np.mean(joint_ranges)),
    "joint_range_max_rad": float(np.max(joint_ranges)),
    "joint_std_mean_rad": float(np.mean(joint_stds)),
    "gripper_range": float(gripper.max() - gripper.min()),
    "ee_travel_m": ee_travel,
    "n_frames": int(len(joints)),
  }


def scene_metadata(episode):
  site, robot_id, _ = episode["meta"]["episode_id"].split("+")

  return {
    "site": site,
    "robot_id": robot_id,
    "n_cameras": len(episode["camera"]),
    "wrist_serial": episode["meta"]["wrist_serial"],
  }


def robot_coverage(episode, poses, pb_renderer):
  robot = episode["robot"]
  pb_renderer.update_robot_pose(
    robot["joint_positions"][0], gripper_state=robot["gripper_positions"][0]
  )

  coverage = {}
  for cam_id, cam_data in episode["camera"].items():
    height, width = cam_data["raw_depth"][0].shape
    mask = pb_renderer.render_mask(poses[cam_id]["extrinsics"][0], cam_data["K"], width, height)
    coverage[f"robot_percent_{cam_id[:8]}"] = float(mask.mean() * 100)

  return coverage


def episode_metrics(
  episode, poses, device, tracks_3d, per_cam_vis, n_static, n_robot, pb_renderer, config
):
  metrics = {"episode_id": episode["meta"]["episode_id"]}
  metrics.update(scene_metadata(episode))
  metrics.update(robot_coverage(episode, poses, pb_renderer))
  metrics.update(motion_stats(episode))
  metrics.update(evaluate_extrinsics(episode, poses, device, pb_renderer, config))

  metrics["n_static"] = n_static
  metrics["n_robot"] = n_robot
  metrics["n_total_tracks"] = n_static + n_robot
  metrics["n_track_frames"] = tracks_3d.shape[0]

  per_camera = depth_residual_per_camera(episode, poses, tracks_3d, per_cam_vis, n_static)
  metrics.update(
    {
      f"depth_residual_{kind}_mean_mm_{cam_id[:8]}": (
        float(v[kind].mean()) if len(v[kind]) else float("nan")
      )
      for cam_id, v in per_camera.items()
      for kind in ("static", "robot")
    }
  )
  metrics.update(
    {f"vis_percent_{cam_id[:8]}": float(vis.mean() * 100) for cam_id, vis in per_cam_vis.items()}
  )

  return metrics


def load_track_data(episode_id, tracks_root):
  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(tracks_root, episode_id)))

  tracks_data = np.load(os.path.join(ep_dir, "tracks_3d.npz"))
  meta_data = np.load(os.path.join(ep_dir, "track_metadata.npz"))

  per_cam_tracks_2d, per_cam_vis = {}, {}
  for tracks_path in sorted(glob.glob(os.path.join(ep_dir, "*", "tracks_2d.npz"))):
    cam_id = os.path.basename(os.path.dirname(tracks_path))
    cam_data = np.load(tracks_path)
    per_cam_tracks_2d[cam_id] = cam_data["traj_2d"]
    per_cam_vis[cam_id] = cam_data["vis_2d"]

  return {
    "traj_3d": tracks_data["traj_3d"],
    "per_cam_tracks_2d": per_cam_tracks_2d,
    "per_cam_vis": per_cam_vis,
    "n_static": int(meta_data["n_static"]),
    "n_robot": int(meta_data["n_robot"]),
  }


def process_episode(episode_id, device, pb_renderer, csv_path, config):
  t0 = time.time()
  episode = core.io.load_depth_data(episode_id, config.paths.depth, load_video=None)
  poses = core.io.load_extrinsics(episode, config.paths.extrinsics)
  tracks = load_track_data(episode_id, config.paths.tracks)

  metrics = episode_metrics(
    episode,
    poses,
    device,
    tracks["traj_3d"],
    tracks["per_cam_vis"],
    tracks["n_static"],
    tracks["n_robot"],
    pb_renderer,
    config,
  )
  _append_row(csv_path, metrics)

  static_means = [v for k, v in metrics.items() if k.startswith("depth_residual_static_mean_mm_")]
  robot_means = [v for k, v in metrics.items() if k.startswith("depth_residual_robot_mean_mm_")]
  print(
    f"  [OK] Done in {time.time() - t0:.1f}s | "
    f"static={np.mean(static_means):.1f}mm | "
    f"robot={np.mean(robot_means):.1f}mm"
  )


def _read_done(csv_path):
  if not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0):
    return set()
  with open(csv_path, "r") as f:
    return {row.get("episode_id", "") for row in csv.DictReader(f)}


def _append_row(csv_path, metrics):
  os.makedirs(os.path.dirname(csv_path), exist_ok=True)
  with open(csv_path, "a", newline="") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0, 2)
    writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
    if f.tell() == 0:
      writer.writeheader()
    writer.writerow(metrics)
    fcntl.flock(f, fcntl.LOCK_UN)


def main(_):
  config = config_flag.value
  output_dir = os.path.abspath(os.path.expanduser(config.paths.metrics))
  csv_path = os.path.join(output_dir, "metrics.csv")

  available = (
    core.runner.list_episode_dirs(config.paths.depth)
    & core.runner.list_episode_dirs(config.paths.extrinsics)
    & core.runner.list_episode_dirs(config.paths.tracks)
  )

  device = core.io.get_accelerator()
  pb_renderer = core.physics.PyBulletRenderer(config.paths.urdf, gpu=config.render.gpu)

  def run_one(episode_id):
    process_episode(episode_id, device, pb_renderer, csv_path, config)

  core.runner.run_episodes(
    core.runner.shard_episodes(
      available, config.runner.rank, config.runner.world_size, config.runner.limit
    ),
    run_one,
    rank=config.runner.rank,
    world_size=config.runner.world_size,
    done=_read_done(csv_path),
    stage="Evaluation",
  )


if __name__ == "__main__":
  config_flag = config_flags.DEFINE_config_file("config", config.__file__)
  app.run(main)
