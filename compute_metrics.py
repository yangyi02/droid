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


def seen_surface(cam_data, extrinsics, points_3d, t):
  u, v, z = core.geometry.project_points(points_3d, cam_data["K"], extrinsics)
  z_obs = core.geometry.sample_depth(cam_data["raw_depth"][t], u, v, z)
  return core.geometry.unproject_pixels(u, v, np.where(z_obs > 0, z_obs, np.nan), cam_data["K"], extrinsics)


def cross_view_px(episode, poses, tracks):
  tracks_3d = tracks["tracks_3d"]
  per_cam_vis = dict(zip(episode["camera"], tracks["vis"], strict=True))
  wrist_cam_id = episode["meta"]["wrist_serial"]

  error = {}
  for a, b in itertools.combinations(sorted(episode["camera"]), 2):
    gaps = []
    for t in range(len(tracks_3d)):
      for src, dst in ((a, b), (b, a)):
        points = tracks_3d[t][per_cam_vis[src][t] & per_cam_vis[dst][t]]
        if not len(points):
          continue
        surface = seen_surface(episode["camera"][src], poses[src]["extrinsics"][t], points, t)
        K, extrinsics = episode["camera"][dst]["K"], poses[dst]["extrinsics"][t]
        u, v, z = core.geometry.project_points(points, K, extrinsics)
        u_seen, v_seen, z_seen = core.geometry.project_points(surface, K, extrinsics)
        keep = np.isfinite(surface).all(axis=1) & (z > 0) & (z_seen > 0)
        gaps.append(np.hypot(u_seen - u, v_seen - v)[keep])

    gaps = np.concatenate(gaps) if gaps else np.zeros(0)
    name = "cross_view_wrist_px" if wrist_cam_id in (a, b) else "cross_view_px"
    error[f"{name}_{a}_{b}"] = float(np.percentile(gaps, 95)) if len(gaps) else float("nan")

  return error


def track_stats(tracks):
  tracks_3d, n_static = tracks["tracks_3d"], tracks["n_static"]
  n_robot = tracks_3d.shape[1] - n_static
  accel = np.diff(tracks_3d[:, n_static:], n=2, axis=0)
  jitter = np.percentile(np.linalg.norm(accel, axis=-1), 95) * 1000.0 if accel.size else float("nan")

  return {
    "track_jitter_mm": float(jitter),
    "n_static": n_static,
    "n_robot": n_robot,
    "n_total_tracks": n_static + n_robot,
    "n_track_frames": len(tracks_3d),
  }


def track_visibility(episode, tracks):
  return {
    f"vis_percent_{cam_id}": float(vis.mean() * 100)
    for cam_id, vis in zip(episode["camera"], tracks["vis"], strict=True)
  }


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
  site, scene, _ = episode["meta"]["episode_id"].split("+")

  return {
    "site": site,
    "scene": scene,
    "n_cameras": len(episode["camera"]),
    "wrist_serial": episode["meta"]["wrist_serial"],
  }


def robot_coverage(episode, poses, pb_renderer):
  robot = episode["robot"]
  pb_renderer.update_robot_pose(robot["joint_positions"][0], gripper_state=robot["gripper_positions"][0])

  coverage = {}
  for cam_id, cam_data in episode["camera"].items():
    height, width = cam_data["raw_depth"][0].shape
    mask = pb_renderer.render_mask(poses[cam_id]["extrinsics"][0], cam_data["K"], width, height)
    coverage[f"robot_percent_{cam_id}"] = float(mask.mean() * 100)

  return coverage


def episode_metrics(episode, poses, device, tracks, pb_renderer, config):
  return (
    {"episode_id": episode["meta"]["episode_id"]}
    | scene_metadata(episode)
    | motion_stats(episode)
    | track_stats(tracks)
    | track_visibility(episode, tracks)
    | robot_coverage(episode, poses, pb_renderer)
    | evaluate_extrinsics(episode, poses, device, pb_renderer, config)
    | mean_residual(depth_residual(episode, poses, tracks))
    | cross_view_px(episode, poses, tracks)
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
