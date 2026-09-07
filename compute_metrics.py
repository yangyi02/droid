#!/usr/bin/env python3
import csv
import fcntl
import itertools
import os
import time

import numpy as np
import torch
from absl import app
from ml_collections import config_flags

import compute_extrinsics
import config
import core.geometry
import core.io
import core.physics
import core.runner


@torch.no_grad()
def evaluate_extrinsics(scene_constants, scene_state, device, pb_renderer):
  wrist_cam = scene_constants["meta"]["wrist_serial"]
  ext_cams = [c for c in scene_constants["camera"] if c != wrist_cam]
  cams = ext_cams + [wrist_cam]
  pairs = list(itertools.combinations(cams, 2))
  label = {c: str(i + 1) for i, c in enumerate(ext_cams)} | {wrist_cam: "w"}
  suffix = {c: f"cam{i + 1}" for i, c in enumerate(ext_cams)} | {wrist_cam: "wrist"}
  n_frames = len(scene_constants["robot"]["joint_positions"])
  T_ee_all = scene_constants["robot"]["T_ee_base_all"]

  metrics = {}
  base = {}
  for cam in cams:
    base[cam] = torch.tensor(scene_state[cam]["base_extrinsic"], dtype=torch.float32, device=device)

  for cam in cams:
    K_mat = scene_constants["camera"][cam]["K_mat"]
    cache_pts, cache_obs = [], []

    for t in range(n_frames):
      pb_renderer.update_robot_pose(
        scene_constants["robot"]["joint_positions"][t],
        scene_constants["robot"]["gripper_positions"][t],
      )
      d_obs = scene_constants["camera"][cam]["raw_depth"][t].astype(np.float32)
      T_cam = scene_state[cam]["extrinsics"][t]

      if cam == wrist_cam:
        pts = core.physics.get_foreground_gripper_points(T_cam, K_mat, d_obs, pb_renderer, device)
        if pts is None:
          continue
        T_world_to_ee = np.linalg.inv(T_ee_all[t])
        pts_world = (T_cam @ pts)[:3, :].T
        pts = torch.tensor(
          (T_world_to_ee[:3, :3] @ pts_world.T + T_world_to_ee[:3, 3:4]).T,
          dtype=torch.float32,
          device=device,
        )
      else:
        pts = core.physics.get_foreground_robot_points(T_cam, K_mat, d_obs, pb_renderer, device)
        if pts is None:
          continue

      cache_pts.append(pts)
      cache_obs.append(torch.tensor(d_obs, dtype=torch.float32, device=device)[None, ...])

    if not cache_pts:
      metrics[f"robot_loss_{suffix[cam]}"] = float("nan")
      continue

    K = torch.tensor(K_mat, dtype=torch.float32, device=device)
    loss = core.physics.depth_loss_batched(
      torch.stack(cache_pts), base[cam], K, torch.stack(cache_obs)
    )
    metrics[f"robot_loss_{suffix[cam]}"] = loss.item()

  chamfer_sum = {p: 0.0 for p in pairs}
  overlap_sum = {p: 0.0 for p in pairs}
  n_valid = 0

  for t in range(n_frames):
    env = {}
    for cam in cams:
      env[cam] = compute_extrinsics.camera_frame_points(
        t, scene_constants["camera"][cam], device, n_points=5000
      )
    if any(pts is None for pts in env.values()):
      continue

    ee_pose = torch.tensor(T_ee_all[t], dtype=torch.float32, device=device)
    world = {}
    for cam in cams:
      to_world = ee_pose @ base[cam] if cam == wrist_cam else base[cam]
      world[cam] = (to_world @ env[cam])[:3, :].T.unsqueeze(0)

    for a, b in pairs:
      loss, overlap = compute_extrinsics.batched_chamfer_distance(world[a], world[b])
      chamfer_sum[a, b] += loss.item()
      overlap_sum[a, b] += overlap.item()
    n_valid += 1

  for a, b in pairs:
    metrics[f"chamfer_{label[a]}{label[b]}"] = chamfer_sum[a, b] / n_valid
    metrics[f"overlap_{label[a]}{label[b]}"] = overlap_sum[a, b] / n_valid * 100

  return metrics


def compute_depth_residual_mm(pts_3d, K, extrinsics, raw_depth, w_img, h_img):
  u_proj, v_proj, z_proj = core.geometry.project_points(pts_3d, K, extrinsics)
  ui = np.clip(np.round(u_proj).astype(int), 0, w_img - 1)
  vi = np.clip(np.round(v_proj).astype(int), 0, h_img - 1)
  z_obs = raw_depth[vi, ui]
  valid = (z_obs > 0) & (z_proj > 0)
  return np.abs(z_proj[valid] - z_obs[valid]).astype(np.float32) * 1000.0


def compute_depth_residual_per_camera(
  scene_constants, scene_state, final_traj_3d, final_per_cam_vis, n_static
):
  camera_ids = list(scene_constants["camera"].keys())
  T_frames = final_traj_3d.shape[0]

  per_camera = {}
  for cam_id in camera_ids:
    cam_data = scene_constants["camera"][cam_id]
    K = cam_data["K_mat"]
    h_img, w_img = cam_data["raw_depth"][0].shape[:2]

    cam_static, cam_robot = [], []

    for t in range(T_frames):
      raw_depth = cam_data["raw_depth"][t]
      ext = scene_state[cam_id]["extrinsics"][t]
      vis_t = final_per_cam_vis[cam_id][t]

      cam_static.append(
        compute_depth_residual_mm(
          final_traj_3d[t, :n_static][vis_t[:n_static]], K, ext, raw_depth, w_img, h_img
        )
      )
      cam_robot.append(
        compute_depth_residual_mm(
          final_traj_3d[t, n_static:][vis_t[n_static:]], K, ext, raw_depth, w_img, h_img
        )
      )

    per_camera[cam_id] = {"static": np.concatenate(cam_static), "robot": np.concatenate(cam_robot)}

  return per_camera


def compute_track_depth_consistency(
  scene_constants, scene_state, final_traj_3d, final_per_cam_vis, n_static
):
  per_camera = compute_depth_residual_per_camera(
    scene_constants, scene_state, final_traj_3d, final_per_cam_vis, n_static
  )

  def _stats(arrs):
    concat = np.concatenate(arrs)
    if len(concat) == 0:
      return float("nan"), float("nan")
    return float(np.median(concat)), float(np.mean(concat))

  s_med, s_mean = _stats([v["static"] for v in per_camera.values()])
  r_med, r_mean = _stats([v["robot"] for v in per_camera.values()])

  return {
    "depth_residual_static_median_mm": s_med,
    "depth_residual_static_mean_mm": s_mean,
    "depth_residual_robot_median_mm": r_med,
    "depth_residual_robot_mean_mm": r_mean,
  }


def compute_track_visibility_stats(final_per_cam_vis):
  return {
    f"vis_pct_{cam_id[:8]}": float(vis.mean() * 100) for cam_id, vis in final_per_cam_vis.items()
  }


def compute_motion_stats(scene_constants):
  robot = scene_constants["robot"]
  joints = robot["joint_positions"]
  gripper = robot["gripper_positions"]

  joint_ranges = joints.max(axis=0) - joints.min(axis=0)
  joint_stds = joints.std(axis=0)

  T_ee_all = robot["T_ee_base_all"]
  ee_positions = T_ee_all[:, :3, 3]
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


def compute_scene_metadata(scene_constants):
  site, robot_id, _ = scene_constants["meta"]["episode_id"].split("+")

  return {
    "site": site,
    "robot_id": robot_id,
    "n_cameras": len(scene_constants["camera"]),
    "wrist_serial": scene_constants["meta"]["wrist_serial"],
  }


def compute_robot_coverage(scene_constants, scene_state, pb_renderer):
  robot = scene_constants["robot"]
  pb_renderer.update_robot_pose(
    robot["joint_positions"][0], gripper_state=robot["gripper_positions"][0]
  )

  coverage = {}
  for cam_id, cam_data in scene_constants["camera"].items():
    h_img, w_img = cam_data["raw_depth"][0].shape
    mask = pb_renderer.render_mask(
      scene_state[cam_id]["extrinsics"][0], cam_data["K_mat"], w_img, h_img
    )
    coverage[f"robot_pct_{cam_id[:8]}"] = float(mask.mean() * 100)

  return coverage


def compute_episode_metrics(
  scene_constants,
  scene_state,
  device,
  final_traj_3d,
  final_per_cam_vis,
  n_static,
  n_robot,
  pb_renderer,
):
  metrics = {"episode_id": scene_constants["meta"]["episode_id"]}
  metrics.update(compute_scene_metadata(scene_constants))
  metrics.update(compute_robot_coverage(scene_constants, scene_state, pb_renderer))
  metrics.update(compute_motion_stats(scene_constants))
  metrics.update(evaluate_extrinsics(scene_constants, scene_state, device, pb_renderer))

  metrics["n_static"] = n_static
  metrics["n_robot"] = n_robot
  metrics["n_total_tracks"] = n_static + n_robot
  metrics["n_track_frames"] = final_traj_3d.shape[0]

  metrics.update(
    compute_track_depth_consistency(
      scene_constants, scene_state, final_traj_3d, final_per_cam_vis, n_static
    )
  )
  metrics.update(compute_track_visibility_stats(final_per_cam_vis))

  return metrics


def load_track_data(episode_id, tracks_root):
  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(tracks_root, episode_id)))

  tracks_data = np.load(os.path.join(ep_dir, "tracks_3d.npz"))
  meta_data = np.load(os.path.join(ep_dir, "track_metadata.npz"))

  per_cam_tracks, per_cam_vis = {}, {}
  for cam_dir_name in sorted(os.listdir(ep_dir)):
    cam_dir = os.path.join(ep_dir, cam_dir_name)
    cam_data = np.load(os.path.join(cam_dir, "tracks_2d.npz"))
    per_cam_tracks[cam_dir_name] = cam_data["traj_2d"]
    per_cam_vis[cam_dir_name] = cam_data["vis_2d"]

  return {
    "traj_3d": tracks_data["traj_3d"],
    "per_cam_tracks": per_cam_tracks,
    "per_cam_vis": per_cam_vis,
    "n_static": int(meta_data["n_static"]),
    "n_robot": int(meta_data["n_robot"]),
  }


def process_episode(episode_id, device, pb_renderer, csv_path, config):
  t0 = time.time()
  scene_constants = core.io.load_depth_data(episode_id, config.paths.depth, load_video=None)
  scene_state = core.io.load_extrinsics(scene_constants, config.paths.extrinsics)
  tracks = load_track_data(episode_id, config.paths.tracks)

  metrics = compute_episode_metrics(
    scene_constants,
    scene_state,
    device,
    tracks["traj_3d"],
    tracks["per_cam_vis"],
    tracks["n_static"],
    tracks["n_robot"],
    pb_renderer,
  )
  _append_row(csv_path, metrics)

  print(
    f"  [OK] Done in {time.time() - t0:.1f}s | "
    f"static={metrics['depth_residual_static_median_mm']:.1f}mm | "
    f"robot={metrics['depth_residual_robot_median_mm']:.1f}mm"
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
