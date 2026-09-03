#!/usr/bin/env python3
import argparse
import csv
import fcntl
import os
import time

import numpy as np
import pybullet as p
import torch
import torch.nn.functional as F

from compute_extrinsics import (batched_chamfer_distance,
                                get_cam_points_local_t)
from core.geometry import project_points
from core.io import OUTPUT_ROOT, load_depth_data, load_extrinsics, get_accelerator
from core.physics import (PyBulletRenderer, compute_robot_loss_batched,
                          compute_wrist_loss_batched,
                          get_foreground_gripper_points,
                          get_foreground_robot_points)
from core.runner import (add_sharding_args, list_episode_dirs,
                         run_episodes, shard_episodes)


@torch.no_grad()
def evaluate_extrinsics(scene_constants, scene_state, device,
                        pb_renderer=None):
  wrist_cam = scene_constants["meta"]["wrist_serial"]
  ext_cams = [c for c in scene_constants["camera"].keys() if c != wrist_cam]
  cam1, cam2 = ext_cams[0], ext_cams[1]
  n_frames = len(scene_constants["robot"]["joint_positions"])
  T_ee_all = scene_constants["robot"]["T_ee_base_all"]

  metrics = {}

  if pb_renderer is not None:
    for cam_id, key_prefix in [(cam1, "cam1"), (cam2, "cam2"),
                                (wrist_cam, "wrist")]:
      try:
        is_wrist = (cam_id == wrist_cam)
        K_np = scene_constants["camera"][cam_id]["K_mat"]
        K_t = torch.tensor(K_np, dtype=torch.float32, device=device)

        cache_X, cache_obs = [], []
        for t in range(n_frames):
          joints = scene_constants["robot"]["joint_positions"][t]
          gripper = scene_constants["robot"]["gripper_positions"][t]
          pb_renderer.update_robot_pose(joints, gripper)

          d_obs = scene_constants["camera"][cam_id]["raw_depth"][t].astype(
              np.float32)
          T_cam_np = scene_state[cam_id]["extrinsics"][t]

          if is_wrist:
            pts = get_foreground_gripper_points(
                T_cam_np, K_np, d_obs, pb_renderer, device)
            if pts is None:
              continue
            T_world_to_ee = np.linalg.inv(T_ee_all[t])
            pts_world = (T_cam_np @ pts)[:3, :].T
            pts_ee = (T_world_to_ee[:3, :3] @ pts_world.T
                      + T_world_to_ee[:3, 3:4]).T
            cache_X.append(
                torch.tensor(pts_ee, dtype=torch.float32, device=device))
          else:
            pts = get_foreground_robot_points(
                T_cam_np, K_np, d_obs, pb_renderer, device)
            if pts is None:
              continue
            cache_X.append(pts)

          cache_obs.append(
              torch.tensor(d_obs, dtype=torch.float32,
                           device=device)[None, ...])

        if not cache_X:
          metrics[f"robot_loss_{key_prefix}"] = float("nan")
          continue

        batch_X = torch.stack(cache_X)
        batch_obs = torch.stack(cache_obs)
        T_opt = torch.tensor(
            scene_state[cam_id]["base_extrinsic"],
            dtype=torch.float32, device=device)

        if is_wrist:
          loss = compute_wrist_loss_batched(batch_X, T_opt, K_t, batch_obs)
        else:
          loss = compute_robot_loss_batched(batch_X, T_opt, K_t, batch_obs)
        metrics[f"robot_loss_{key_prefix}"] = loss.item()
      except Exception as e:
        metrics[f"robot_loss_{key_prefix}"] = float("nan")

  try:
    T1 = torch.tensor(
        scene_state[cam1]["base_extrinsic"],
        dtype=torch.float32, device=device)
    T2 = torch.tensor(
        scene_state[cam2]["base_extrinsic"],
        dtype=torch.float32, device=device)
    Tw = torch.tensor(
        scene_state[wrist_cam]["base_extrinsic"],
        dtype=torch.float32, device=device)

    sum_l12, sum_l1w, sum_l2w = 0.0, 0.0, 0.0
    sum_o12, sum_o1w, sum_o2w = 0.0, 0.0, 0.0
    n_valid = 0

    for t in range(n_frames):
      pc1 = get_cam_points_local_t(t, scene_constants["camera"][cam1], device, n_points=5000)
      pc2 = get_cam_points_local_t(t, scene_constants["camera"][cam2], device, n_points=5000)
      pcw = get_cam_points_local_t(
          t, scene_constants["camera"][wrist_cam], device, n_points=5000)
      if pc1 is None or pc2 is None or pcw is None:
        continue

      T_ee_t = torch.tensor(T_ee_all[t], dtype=torch.float32, device=device)

      w1 = (T1 @ pc1)[:3, :].T.unsqueeze(0)
      w2 = (T2 @ pc2)[:3, :].T.unsqueeze(0)
      ww = ((T_ee_t @ Tw) @ pcw)[:3, :].T.unsqueeze(0)

      l12, o12 = batched_chamfer_distance(w1, w2, device)
      l1w, o1w = batched_chamfer_distance(w1, ww, device)
      l2w, o2w = batched_chamfer_distance(w2, ww, device)

      sum_l12 += l12.item()
      sum_l1w += l1w.item()
      sum_l2w += l2w.item()
      sum_o12 += o12
      sum_o1w += o1w
      sum_o2w += o2w
      n_valid += 1

    metrics["chamfer_12"] = sum_l12 / n_valid
    metrics["chamfer_1w"] = sum_l1w / n_valid
    metrics["chamfer_2w"] = sum_l2w / n_valid
    metrics["chamfer_total"] = (sum_l12 + sum_l1w + sum_l2w) / n_valid
    metrics["bg_overlap_pct"] = (sum_o12 + sum_o1w + sum_o2w) / (3.0 * n_valid) * 100
  except Exception as e:
    metrics["chamfer_total"] = float("nan")
    metrics["bg_overlap_pct"] = float("nan")

  return metrics


def print_metrics(metrics, stage_name=""):
  chamfer = metrics.get("chamfer_total", float("nan"))
  rob1 = metrics.get("robot_loss_cam1", float("nan"))
  rob2 = metrics.get("robot_loss_cam2", float("nan"))
  robw = metrics.get("robot_loss_wrist", float("nan"))
  overlap = metrics.get("bg_overlap_pct", float("nan"))
  track = metrics.get("track_reproj_mean_px", float("nan"))
  track_med = metrics.get("track_reproj_median_px", float("nan"))
  wrist_bg = metrics.get("track_reproj_wrist_bg_mean_px", float("nan"))
  wrist_bg_med = metrics.get("track_reproj_wrist_bg_median_px", float("nan"))
  static_robot = metrics.get("track_reproj_static_robot_mean_px", float("nan"))
  static_robot_med = metrics.get("track_reproj_static_robot_median_px", float("nan"))

  header = f"Metrics after {stage_name}" if stage_name else "Metrics"
  print(f"\n{header}")
  print(f"  Chamfer total: {chamfer:.4f}")
  print(f"  Robot depth:   cam1={rob1:.4f}  cam2={rob2:.4f}  wrist={robw:.4f}")
  print(f"  BG overlap:    {overlap:.1f}%")
  if not np.isnan(wrist_bg):
    med_s = f"  median={wrist_bg_med:.2f}" if not np.isnan(wrist_bg_med) else ""
    print(f"  Track wristBG: mean={wrist_bg:.2f} px{med_s}  primary")
  if not np.isnan(static_robot):
    med_s2 = f"  median={static_robot_med:.2f}" if not np.isnan(static_robot_med) else ""
    print(f"  Track robot:   mean={static_robot:.2f} px{med_s2}")
  if not np.isnan(track) and np.isnan(wrist_bg) and np.isnan(static_robot):
    print(f"  Track reproj:  mean={track:.2f} px")

  shift_keys = [k for k in sorted(metrics.keys()) if k.startswith("shift_mm_")]
  if shift_keys:
    shifts = [f"{k.replace('shift_mm_', '')}={metrics[k]:.1f}mm"
              for k in shift_keys]
    print(f"  Shift from 0:  {', '.join(shifts)}")
  print()


def compute_depth_residual_mm(pts_3d, K, extrinsics, raw_depth, w_img, h_img):
  if len(pts_3d) == 0:
    return np.array([], dtype=np.float32)
  u_proj, v_proj, z_proj = project_points(pts_3d, K, extrinsics)
  ui = np.clip(np.round(u_proj).astype(int), 0, w_img - 1)
  vi = np.clip(np.round(v_proj).astype(int), 0, h_img - 1)
  z_obs = raw_depth[vi, ui]
  valid = (z_obs > 0.05) & (z_proj > 0)
  if not valid.any():
    return np.array([], dtype=np.float32)
  return np.abs(z_proj[valid] - z_obs[valid]).astype(np.float32) * 1000.0


def compute_depth_residual_per_camera(
    scene_constants, scene_state,
    final_traj_3d, final_per_cam_vis, n_static, n_robot):
  camera_ids = list(scene_constants["camera"].keys())
  T_frames = final_traj_3d.shape[0]

  per_camera = {}
  for cam_id in camera_ids:
    cam_data = scene_constants["camera"][cam_id]
    K = cam_data["K_mat"]
    h_img, w_img = cam_data["raw_depth"][0].shape[:2]

    cam_static, cam_robot, cam_all = [], [], []

    for t in range(T_frames):
      raw_depth = cam_data["raw_depth"][t]
      ext = scene_state[cam_id]["extrinsics"][t]
      vis_t = final_per_cam_vis[cam_id][t]

      if n_static > 0:
        cam_static.append(compute_depth_residual_mm(
            final_traj_3d[t, :n_static][vis_t[:n_static]],
            K, ext, raw_depth, w_img, h_img))

      if n_robot > 0:
        cam_robot.append(compute_depth_residual_mm(
            final_traj_3d[t, n_static:][vis_t[n_static:]],
            K, ext, raw_depth, w_img, h_img))

      cam_all.append(compute_depth_residual_mm(
          final_traj_3d[t, vis_t], K, ext, raw_depth, w_img, h_img))

    per_camera[cam_id] = {
        "static": np.concatenate(cam_static) if cam_static else np.array([], dtype=np.float32),
        "robot": np.concatenate(cam_robot) if cam_robot else np.array([], dtype=np.float32),
        "all": np.concatenate(cam_all) if cam_all else np.array([], dtype=np.float32),
    }

  return per_camera


def compute_track_depth_consistency(
    scene_constants, scene_state,
    final_traj_3d, final_per_cam_vis, n_static, n_robot):
  per_camera = compute_depth_residual_per_camera(
      scene_constants, scene_state,
      final_traj_3d, final_per_cam_vis, n_static, n_robot)

  all_static = [v["static"] for v in per_camera.values()]
  all_robot = [v["robot"] for v in per_camera.values()]
  all_overall = [v["all"] for v in per_camera.values()]

  def _stats(arrs):
    concat = np.concatenate(arrs) if arrs else np.array([])
    if len(concat) == 0:
      return float("nan"), float("nan")
    return float(np.median(concat)), float(np.mean(concat))

  s_med, s_mean = _stats(all_static)
  r_med, r_mean = _stats(all_robot)
  o_med, o_mean = _stats(all_overall)

  return {
      "depth_residual_static_median_mm": s_med,
      "depth_residual_static_mean_mm": s_mean,
      "depth_residual_robot_median_mm": r_med,
      "depth_residual_robot_mean_mm": r_mean,
      "depth_residual_overall_median_mm": o_med,
      "depth_residual_overall_mean_mm": o_mean,
  }


def compute_track_visibility_stats(
    final_per_cam_vis, n_static, n_robot):
  stats = {}
  all_static_vis, all_robot_vis, all_total_vis = [], [], []

  for cam_id, vis in final_per_cam_vis.items():
    vis_static = vis[:, :n_static] if n_static > 0 else np.zeros((vis.shape[0], 0), dtype=bool)
    vis_robot = vis[:, n_static:] if n_robot > 0 else np.zeros((vis.shape[0], 0), dtype=bool)

    s_pct = float(vis_static.mean() * 100) if vis_static.size > 0 else 0.0
    r_pct = float(vis_robot.mean() * 100) if vis_robot.size > 0 else 0.0
    t_pct = float(vis.mean() * 100)

    stats[f"vis_static_pct_{cam_id[:8]}"] = s_pct
    stats[f"vis_robot_pct_{cam_id[:8]}"] = r_pct
    stats[f"vis_total_pct_{cam_id[:8]}"] = t_pct

    all_static_vis.append(s_pct)
    all_robot_vis.append(r_pct)
    all_total_vis.append(t_pct)

  stats["vis_static_pct_avg"] = float(np.mean(all_static_vis)) if all_static_vis else 0.0
  stats["vis_robot_pct_avg"] = float(np.mean(all_robot_vis)) if all_robot_vis else 0.0
  stats["vis_total_pct_avg"] = float(np.mean(all_total_vis)) if all_total_vis else 0.0

  return stats


def compute_reprojection_error(traj_3d, traj_2d, vis_2d,
                               intrinsics, extrinsics_w2c):
  T, N, _ = traj_3d.shape
  fx, fy, cx, cy = intrinsics

  ones = np.ones((T, N, 1), dtype=traj_3d.dtype)
  pts_homo = np.concatenate([traj_3d, ones], axis=2)

  pts_cam = np.einsum('tij,tnj->tni', extrinsics_w2c, pts_homo)

  z = pts_cam[:, :, 2]
  valid = vis_2d & (z > 0.01)

  if not valid.any():
    return np.array([], dtype=np.float32)

  with np.errstate(divide='ignore', invalid='ignore'):
    u_proj = fx * pts_cam[:, :, 0] / z + cx
    v_proj = fy * pts_cam[:, :, 1] / z + cy

  du = u_proj - traj_2d[:, :, 0]
  dv = v_proj - traj_2d[:, :, 1]
  errors = np.sqrt(du * du + dv * dv)

  return errors[valid].astype(np.float32)


def compute_reprojection_stats(traj_3d, per_cam_tracks_2d, tracks_root,
                               episode_id):
  ep_dir = os.path.abspath(
      os.path.expanduser(os.path.join(tracks_root, episode_id)))

  all_errors = []
  cam_dirs = sorted([
      d for d in os.listdir(ep_dir)
      if os.path.isdir(os.path.join(ep_dir, d))
  ])

  for cam_dir_name in cam_dirs:
    cam_dir = os.path.join(ep_dir, cam_dir_name)
    tracks_2d_path = os.path.join(cam_dir, "tracks_2d.npz")
    intrinsics_path = os.path.join(cam_dir, "intrinsics.npy")
    extrinsics_path = os.path.join(cam_dir, "extrinsics_w2c.npy")

    if not all(os.path.exists(p) for p in
               [tracks_2d_path, intrinsics_path, extrinsics_path]):
      continue

    cam_data = np.load(tracks_2d_path)
    traj_2d = cam_data["traj_2d"]
    vis_2d = cam_data["vis_2d"]
    intrinsics = np.load(intrinsics_path)
    extrinsics_w2c = np.load(extrinsics_path)

    errs = compute_reprojection_error(
        traj_3d, traj_2d, vis_2d, intrinsics, extrinsics_w2c)
    if len(errs) > 0:
      all_errors.append(errs)

  if not all_errors:
    return {}

  all_errs = np.concatenate(all_errors)
  return {
      "reproj_mean_px": float(np.mean(all_errs)),
      "reproj_median_px": float(np.median(all_errs)),
      "reproj_p95_px": float(np.percentile(all_errs, 95)),
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
  ep_id = scene_constants["meta"]["episode_id"]
  parts = ep_id.split("+")
  site = parts[0] if parts else "UNKNOWN"
  robot_id = parts[1] if len(parts) > 1 else "UNKNOWN"

  camera_ids = list(scene_constants["camera"].keys())
  first_cam = scene_constants["camera"][camera_ids[0]]

  if "raw_depth" in first_cam:
    h, w = first_cam["raw_depth"][0].shape[:2]
  elif "video_rgb" in first_cam:
    h, w = first_cam["video_rgb"][0].shape[:2]
  elif "first_frame_rgb" in first_cam:
    h, w = first_cam["first_frame_rgb"].shape[:2]
  else:
    h, w = 0, 0

  return {
      "site": site,
      "robot_id": robot_id,
      "n_cameras": len(camera_ids),
      "image_resolution": f"{h}x{w}",
      "wrist_serial": scene_constants["meta"].get("wrist_serial", ""),
  }


def compute_depth_coverage_stats(scene_constants):
  stats = {}
  all_coverage = []

  for cam_id in scene_constants["camera"]:
    cam_data = scene_constants["camera"][cam_id]
    if "raw_depth" not in cam_data:
      continue

    depth = cam_data["raw_depth"]
    valid = (depth > 0.05) & (depth < 10.0)
    coverage = float(valid.mean() * 100)
    median_depth = float(np.median(depth[valid])) if valid.any() else float("nan")
    depth_range = float(depth[valid].max() - depth[valid].min()) if valid.any() else float("nan")

    stats[f"depth_coverage_pct_{cam_id[:8]}"] = coverage
    stats[f"depth_median_m_{cam_id[:8]}"] = median_depth
    stats[f"depth_range_m_{cam_id[:8]}"] = depth_range
    all_coverage.append(coverage)

  stats["depth_coverage_pct_avg"] = float(np.mean(all_coverage)) if all_coverage else 0.0

  return stats


def evaluate_episode(scene_constants, scene_state, device,
                     final_traj_3d=None, final_per_cam_vis=None,
                     n_static=0, n_robot=0,
                     tracks_root=None,
                     compute_extrinsics_metrics=True,
                     pb_renderer=None):
  ep_id = scene_constants["meta"]["episode_id"]
  metrics = {"episode_id": ep_id}

  metrics.update(compute_scene_metadata(scene_constants))

  metrics.update(compute_motion_stats(scene_constants))

  metrics.update(compute_depth_coverage_stats(scene_constants))

  if compute_extrinsics_metrics:
    try:
      ext_metrics = evaluate_extrinsics(
          scene_constants, scene_state, device,
          pb_renderer=pb_renderer)
      metrics.update(ext_metrics)
    except Exception as e:
      metrics["extrinsics_error"] = str(e)

  if final_traj_3d is not None and final_per_cam_vis is not None:
    metrics["n_static"] = n_static
    metrics["n_robot"] = n_robot
    metrics["n_total_tracks"] = n_static + n_robot
    metrics["n_track_frames"] = final_traj_3d.shape[0]

    metrics.update(compute_track_depth_consistency(
        scene_constants, scene_state,
        final_traj_3d, final_per_cam_vis, n_static, n_robot))

    metrics.update(compute_track_visibility_stats(
        final_per_cam_vis, n_static, n_robot))

    if tracks_root is not None:
      try:
        reproj = compute_reprojection_stats(
            final_traj_3d, None, tracks_root, ep_id)
        metrics.update(reproj)
      except Exception as e:
        metrics["reproj_error"] = str(e)

  return metrics


def load_track_data(episode_id, tracks_root):
  ep_dir = os.path.abspath(
      os.path.expanduser(os.path.join(tracks_root, episode_id)))

  tracks_path = os.path.join(ep_dir, "tracks_3d.npz")
  meta_path = os.path.join(ep_dir, "track_metadata.npz")
  if not os.path.exists(tracks_path) or not os.path.exists(meta_path):
    return None

  tracks_data = np.load(tracks_path)
  meta_data = np.load(meta_path)

  per_cam_tracks, per_cam_vis = {}, {}
  for cam_dir_name in os.listdir(ep_dir):
    cam_dir = os.path.join(ep_dir, cam_dir_name)
    vis_path = os.path.join(cam_dir, "tracks_2d.npz")
    if os.path.isdir(cam_dir) and os.path.exists(vis_path):
      cam_data = np.load(vis_path)
      per_cam_tracks[cam_dir_name] = cam_data["traj_2d"]
      per_cam_vis[cam_dir_name] = cam_data["vis_2d"]

  if not per_cam_vis:
    return None

  return {
      "traj_3d": tracks_data["traj_3d"],
      "vis_global": tracks_data["vis_global"],
      "per_cam_tracks": per_cam_tracks,
      "per_cam_vis": per_cam_vis,
      "n_static": int(meta_data["n_static"]),
      "n_robot": int(meta_data["n_robot"]),
  }


def evaluate_single_episode(episode_id, depth_root, extrinsics_root,
                            tracks_root, device, pb_renderer):
  scene_constants = load_depth_data(
      episode_id, depth_root, load_video="first_frame")
  scene_state = load_extrinsics(scene_constants, extrinsics_root)

  tracks = load_track_data(episode_id, tracks_root)
  has_tracks = tracks is not None
  final_traj_3d = tracks["traj_3d"] if has_tracks else None
  final_per_cam_vis = tracks["per_cam_vis"] if has_tracks else None
  n_static = tracks["n_static"] if has_tracks else 0
  n_robot = tracks["n_robot"] if has_tracks else 0

  metrics = evaluate_episode(
      scene_constants, scene_state, device,
      final_traj_3d=final_traj_3d,
      final_per_cam_vis=final_per_cam_vis,
      n_static=n_static,
      n_robot=n_robot,
      tracks_root=tracks_root,
      compute_extrinsics_metrics=True,
      pb_renderer=pb_renderer,
  )
  metrics["has_tracks"] = has_tracks

  return metrics


def _read_done(csv_path):
  if not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0):
    return set()
  with open(csv_path, "r") as f:
    return {row.get("episode_id", "") for row in csv.DictReader(f)}


def _append_row(csv_path, metrics):
  with open(csv_path, "a", newline="") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0, 2)
    writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
    if f.tell() == 0:
      writer.writeheader()
    writer.writerow(metrics)
    fcntl.flock(f, fcntl.LOCK_UN)


def _log_failure(path, ep_id, err):
  with open(path, "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(f"{ep_id}\t{err}\n")
    fcntl.flock(f, fcntl.LOCK_UN)


def main():
  parser = argparse.ArgumentParser(
      description="Batch quality metrics evaluation for DROID episodes")
  add_sharding_args(parser)
  parser.add_argument("--depth_root", type=str,
                      default=os.path.join(OUTPUT_ROOT, "depth"))
  parser.add_argument("--extrinsics_root", type=str,
                      default=os.path.join(OUTPUT_ROOT, "extrinsics"))
  parser.add_argument("--tracks_root", type=str,
                      default=os.path.join(OUTPUT_ROOT, "tracks"))
  parser.add_argument("--output_dir", type=str,
                      default=os.path.join(OUTPUT_ROOT, "metrics"))
  parser.add_argument("--require_tracks", action="store_true",
                      help="Only evaluate episodes with track data")
  args = parser.parse_args()

  output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
  os.makedirs(output_dir, exist_ok=True)
  csv_path = os.path.join(output_dir, "metrics.csv")
  fail_path = os.path.join(output_dir, "failures.txt")

  available = list_episode_dirs(args.depth_root) & list_episode_dirs(
      args.extrinsics_root)
  if args.require_tracks:
    available &= list_episode_dirs(args.tracks_root)
  print(f"Found {len(available)} episodes with depth + extrinsics")

  device = get_accelerator()
  pb_renderer = PyBulletRenderer()

  def evaluate(ep_id):
    t0 = time.time()
    try:
      metrics = evaluate_single_episode(
          ep_id, args.depth_root, args.extrinsics_root,
          args.tracks_root, device, pb_renderer)
    except Exception as e:
      _log_failure(fail_path, ep_id, e)
      raise
    _append_row(csv_path, metrics)
    print(f"  [OK] Done in {time.time() - t0:.1f}s | "
          f"chamfer={metrics.get('chamfer_total', float('nan')):.4f} | "
          f"depth_residual_median="
          f"{metrics.get('depth_residual_overall_median_mm', float('nan')):.1f}mm")

  run_episodes(
      shard_episodes(available, args.rank, args.world_size, args.limit),
      evaluate,
      rank=args.rank, world_size=args.world_size,
      done=_read_done(csv_path), stage="Evaluation")
  print(f"   Output: {csv_path}")


if __name__ == "__main__":
  main()
