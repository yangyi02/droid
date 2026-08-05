#!/usr/bin/env python3
"""Batch quality metrics evaluation for DROID episodes.

Evaluates pre-computed pipeline outputs (depth, extrinsics, tracks) to
produce a metrics CSV for episode selection.  Designed for parallel
execution on GCP with multi-GPU sharding.

Contains all metric computation functions (formerly in core/metrics.py)
and extrinsics evaluation functions (formerly in core/pybullet_extrinsics.py).

Usage:
  # Single GPU — all episodes
  python evaluate_episodes.py

  # 4-GPU sharding (run each on a separate GPU)
  python evaluate_episodes.py --rank 0 --world_size 4
  python evaluate_episodes.py --rank 1 --world_size 4
  python evaluate_episodes.py --rank 2 --world_size 4
  python evaluate_episodes.py --rank 3 --world_size 4

Output:
  metrics.csv — one row per episode with ~30+ metric columns.
  All ranks append to the same file (file-locked), no merge step needed.
"""

import argparse
import csv
import fcntl
import os
import random
import time
import traceback

import numpy as np
import pybullet as p
import torch
import torch.nn.functional as F

from compute_extrinsics import (batched_chamfer_distance,
                                get_cam_points_local_t)
from core.geometry import project_points
from core.io import load_depth_data, load_extrinsics, get_accelerator
from core.physics import PyBulletRenderer


# ===========================================================================
# PyBullet-based extrinsics evaluation helpers
# (from core/pybullet_extrinsics.py)
# ===========================================================================

def get_foreground_robot_points(T_init, K, obs_depth, pb_renderer, device,
                                max_pts=2000):
  """Extract robot point cloud via PyBullet depth rendering and reprojection.

  Renders the full robot body at the given extrinsic pose, selects pixels
  where the rendered depth is nonzero, and lifts them to world-frame 3D
  points.

  Args:
    T_init: (4, 4) NumPy camera-to-world extrinsic matrix.
    K: (3, 3) NumPy intrinsic matrix.
    obs_depth: (H, W) NumPy observed depth map (used only for resolution).
    pb_renderer: ``PyBulletRenderer`` instance (from ``core.physics``).
    device: torch device string or object.
    max_pts: Target number of output points.

  Returns:
    (max_pts, 3) float32 tensor on *device*, or ``None`` if too few pixels.
  """
  h_img, w_img = obs_depth.shape
  render_d = pb_renderer.render_depth(T_init, K, w_img, h_img)

  v_r, u_r = np.where(render_d > 0)
  z_r = render_d[v_r, u_r]
  if len(z_r) < max_pts:
    return None

  P_cam_r = np.stack([
      (u_r - K[0, 2]) * z_r / K[0, 0],
      (v_r - K[1, 2]) * z_r / K[1, 1],
      z_r,
      np.ones_like(z_r),
  ])
  pts_robot_world = (T_init @ P_cam_r)[:3, :].T

  idx = np.random.choice(
      len(pts_robot_world), max_pts,
      replace=(len(pts_robot_world) < max_pts),
  )
  return torch.tensor(
      pts_robot_world[idx], dtype=torch.float32, device=device,
  )


def get_foreground_gripper_points(T_cam_world, K, obs_depth, pb_renderer,
                                  device, max_pts=2000):
  """Extract gripper-only point cloud using PyBullet segmentation mask.

  Renders a full camera image and filters by the ghost body's segmentation
  ID so that only gripper pixels survive.  Returns **camera-frame**
  homogeneous coordinates (4, max_pts) for hand-eye optimization.

  Args:
    T_cam_world: (4, 4) NumPy camera-to-world extrinsic matrix.
    K: (3, 3) NumPy intrinsic matrix.
    obs_depth: (H, W) NumPy observed depth map (used only for resolution).
    pb_renderer: ``PyBulletRenderer`` instance.
    device: torch device string or object.
    max_pts: Target number of output points.

  Returns:
    (4, max_pts) NumPy float64 array of camera-frame homogeneous points,
    or ``None`` if fewer than 100 valid pixels.
  """
  h_img, w_img = obs_depth.shape

  cam_pos = T_cam_world[:3, 3]
  target_pos = T_cam_world[:3, 3] + T_cam_world[:3, 2]
  view_matrix = p.computeViewMatrix(
      cam_pos.tolist(), target_pos.tolist(), (-T_cam_world[:3, 1]).tolist(),
  )
  proj_matrix = pb_renderer._get_projection_matrix(K, w_img, h_img)

  _, _, _, depth_buffer, seg_buffer = p.getCameraImage(
      w_img, h_img,
      viewMatrix=view_matrix,
      projectionMatrix=proj_matrix,
      renderer=p.ER_BULLET_HARDWARE_OPENGL,
      flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
  )

  metric_depth = 0.1 / (10.0 - 9.99 * np.reshape(depth_buffer, (h_img, w_img)))
  seg_array = np.reshape(seg_buffer, (h_img, w_img)).astype(np.int32)
  obj_ids = seg_array & 0xFFFFFF
  valid_ghost = (obj_ids == pb_renderer.ghost_id)

  v_r, u_r = np.where((metric_depth < 9.9) & valid_ghost)
  z_r = metric_depth[v_r, u_r]
  if len(z_r) < 100:
    return None

  P_cam_r = np.stack([
      (u_r - K[0, 2]) * z_r / K[0, 0],
      (v_r - K[1, 2]) * z_r / K[1, 1],
      z_r,
      np.ones_like(z_r),
  ])

  idx = np.random.choice(len(z_r), max_pts, replace=(len(z_r) < max_pts))
  return P_cam_r[:, idx]


def compute_robot_loss_batched(batch_X, T_opt, K, batch_obs):
  """Depth re-projection loss for external cameras.

  Projects world-frame robot points into the camera using *T_opt*, samples
  observed depth via differentiable ``grid_sample``, and returns the mean
  absolute depth error.

  Unlike ``compute_robot_loss`` in ``compute_extrinsics.py``, this version
  does **not** use surface normals, front-face culling, or depth tolerance.

  Args:
    batch_X: (B, N, 3) world-frame robot points.
    T_opt: (4, 4) camera-to-world extrinsic (differentiable).
    K: (3, 3) intrinsic matrix (tensor).
    batch_obs: (B, 1, H, W) observed depth maps.

  Returns:
    Scalar loss tensor.
  """
  B, _, h_img, w_img = batch_obs.shape

  P_c = (batch_X - T_opt[:3, 3]) @ T_opt[:3, :3]
  Z_pred = P_c[..., 2]

  u = K[0, 0] * P_c[..., 0] / Z_pred + K[0, 2]
  v = K[1, 1] * P_c[..., 1] / Z_pred + K[1, 2]

  grid = torch.stack([
      (u / (w_img - 1)) * 2 - 1,
      (v / (h_img - 1)) * 2 - 1,
  ], dim=-1).unsqueeze(1)

  Z_obs_raw = F.grid_sample(
      batch_obs, grid, mode='bilinear', padding_mode='border',
      align_corners=True,
  ).squeeze(1).squeeze(1)

  valid_mask = (
      (Z_pred > 0.) & (Z_pred < 1.5) &
      (Z_obs_raw > 0.) & (Z_obs_raw < 1.5) &
      (u >= 0) & (u < w_img - 1) &
      (v >= 0) & (v < h_img - 1)
  )

  diff = torch.abs(Z_obs_raw[valid_mask] - Z_pred[valid_mask])
  return torch.nan_to_num(diff.mean(), nan=0.0)


def compute_wrist_loss_batched(batch_P_ee, T_cam_ee_opt, K, batch_obs):
  """Depth re-projection loss for the wrist camera.

  Points are anchored in the end-effector frame.  The function inverts
  ``T_cam_ee_opt`` to transform them back to the camera frame before
  projection and depth comparison.

  Args:
    batch_P_ee: (B, N, 3) points in end-effector frame.
    T_cam_ee_opt: (4, 4) wrist-cam-to-EE extrinsic (differentiable).
    K: (3, 3) intrinsic matrix (tensor).
    batch_obs: (B, 1, H, W) observed depth maps.

  Returns:
    Scalar loss tensor.
  """
  B, _, h_img, w_img = batch_obs.shape

  T_ee_cam = torch.linalg.inv(T_cam_ee_opt)
  P_c = batch_P_ee @ T_ee_cam[:3, :3].T + T_ee_cam[:3, 3]
  Z_pred = P_c[..., 2]

  u = K[0, 0] * P_c[..., 0] / Z_pred + K[0, 2]
  v = K[1, 1] * P_c[..., 1] / Z_pred + K[1, 2]

  grid = torch.stack([
      (u / (w_img - 1)) * 2 - 1,
      (v / (h_img - 1)) * 2 - 1,
  ], dim=-1).unsqueeze(1)

  Z_obs_raw = F.grid_sample(
      batch_obs, grid, mode='bilinear', padding_mode='border',
      align_corners=True,
  ).squeeze(1).squeeze(1)

  valid_mask = (
      (Z_pred > 0.) & (Z_pred < 1.5) &
      (Z_obs_raw > 0.) & (Z_obs_raw < 1.5) &
      (u >= 0) & (u < w_img - 1) &
      (v >= 0) & (v < h_img - 1)
  )

  diff = torch.abs(Z_obs_raw[valid_mask] - Z_pred[valid_mask])
  return torch.nan_to_num(diff.mean(), nan=0.0)


# ===========================================================================
# Extrinsics evaluation
# ===========================================================================

@torch.no_grad()
def evaluate_extrinsics(scene_constants, scene_state, device,
                        pb_renderer=None):
  """Compute extrinsics quality metrics without re-running optimization.

  Uses PyBullet rendering to compute robot depth losses (the evaluation
  path), as opposed to the yourdfpy tensor_renderer used for optimization
  in compute_extrinsics.py.

  Args:
    scene_constants: Scene data dict.
    scene_state: Current extrinsics state.
    device: Torch device.
    pb_renderer: Optional PyBulletRenderer for robot depth losses.

  Returns:
    Dict with metrics: chamfer_total, robot_loss_*, bg_overlap_pct.
  """
  wrist_cam = scene_constants["meta"]["wrist_serial"]
  ext_cams = [c for c in scene_constants["camera"].keys() if c != wrist_cam]
  cam1, cam2 = ext_cams[0], ext_cams[1]
  n_frames = len(scene_constants["robot"]["joint_positions"])
  T_ee_all = scene_constants["robot"]["T_ee_base_all"]

  metrics = {}

  # --- Robot depth losses (PyBullet rendering path) ---
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
            # Convert camera-frame homogeneous (4, N) → EE-frame (N, 3)
            T_world_to_ee = np.linalg.inv(T_ee_all[t])
            pts_world = (T_cam_np @ pts)[:3, :].T  # → (N, 3)
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

  # --- Chamfer losses ---
  try:
    cache_Pc1, cache_Pc2, cache_Pcw, cache_Tee = [], [], [], []
    for t in range(n_frames):
      pc1 = get_cam_points_local_t(t, scene_constants["camera"][cam1], device, n_points=5000)
      pc2 = get_cam_points_local_t(t, scene_constants["camera"][cam2], device, n_points=5000)
      pcw = get_cam_points_local_t(
          t, scene_constants["camera"][wrist_cam], device, n_points=5000)
      if pc1 is not None and pc2 is not None and pcw is not None:
        cache_Pc1.append(pc1)
        cache_Pc2.append(pc2)
        cache_Pcw.append(pcw)
        cache_Tee.append(
            torch.tensor(T_ee_all[t], dtype=torch.float32, device=device))

    batch_Pc1 = torch.stack(cache_Pc1)
    batch_Pc2 = torch.stack(cache_Pc2)
    batch_Pcw = torch.stack(cache_Pcw)
    batch_Tee = torch.stack(cache_Tee)

    T1 = torch.tensor(
        scene_state[cam1]["base_extrinsic"],
        dtype=torch.float32, device=device)
    T2 = torch.tensor(
        scene_state[cam2]["base_extrinsic"],
        dtype=torch.float32, device=device)
    Tw = torch.tensor(
        scene_state[wrist_cam]["base_extrinsic"],
        dtype=torch.float32, device=device)

    bc1 = (T1 @ batch_Pc1)[:, :3, :].transpose(1, 2)
    bc2 = (T2 @ batch_Pc2)[:, :3, :].transpose(1, 2)
    bcw = torch.bmm(batch_Tee @ Tw, batch_Pcw)[:, :3, :].transpose(1, 2)

    l12, o12 = batched_chamfer_distance(bc1, bc2, device)
    l1w, o1w = batched_chamfer_distance(bc1, bcw, device)
    l2w, o2w = batched_chamfer_distance(bc2, bcw, device)

    metrics["chamfer_12"] = l12.item()
    metrics["chamfer_1w"] = l1w.item()
    metrics["chamfer_2w"] = l2w.item()
    metrics["chamfer_total"] = (l12 + l1w + l2w).item()
    metrics["bg_overlap_pct"] = (o12 + o1w + o2w) / 3.0 * 100
  except Exception as e:
    metrics["chamfer_total"] = float("nan")
    metrics["bg_overlap_pct"] = float("nan")

  return metrics


def print_metrics(metrics, stage_name=""):
  """Pretty-print key metrics in a compact one-block summary."""
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


# ===========================================================================
# Quality metrics (formerly core/metrics.py)
# ===========================================================================

# 1. Depth consistency (track 3D → projected depth vs observed depth)

def compute_depth_residual_mm(pts_3d, K, extrinsics, raw_depth, w_img, h_img):
  """Per-point depth residual in millimetres.

  Projects 3D world points into a camera view and compares the projected
  depth against the observed (sensor) depth at each pixel.

  Args:
    pts_3d: (N, 3) world-space 3D points.
    K: (3, 3) camera intrinsic matrix.
    extrinsics: (4, 4) camera-to-world transform (c2w).
    raw_depth: (H, W) observed depth map in metres.
    w_img: image width.
    h_img: image height.

  Returns:
    np.ndarray of residual errors in mm (variable length), or empty array.
  """
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
  """Compute per-camera raw depth residual error arrays.

  Returns per-camera breakdowns suitable for visualization (e.g. histograms).

  Args:
    scene_constants: Scene data dict.
    scene_state: Extrinsics state dict.
    final_traj_3d: (T, N, 3) 3D trajectories.
    final_per_cam_vis: dict of (T, N) visibility per camera.
    n_static: number of static background points.
    n_robot: number of robot points.

  Returns:
    Dict of cam_id → {'static': np.ndarray, 'robot': np.ndarray,
                       'all': np.ndarray} with raw error values in mm.
  """
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
  """Compute global depth consistency stats (aggregated across all cameras).

  Calls ``compute_depth_residual_per_camera`` and aggregates into
  median/mean summary statistics suitable for CSV export.

  Returns:
    Dict with keys:
      depth_residual_static_median_mm, depth_residual_static_mean_mm,
      depth_residual_robot_median_mm, depth_residual_robot_mean_mm,
      depth_residual_overall_median_mm, depth_residual_overall_mean_mm
  """
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


# 2. Track visibility statistics

def compute_track_visibility_stats(
    final_per_cam_vis, n_static, n_robot):
  """Compute per-category visibility percentages.

  Args:
    final_per_cam_vis: dict of cam_id → (T, N) bool arrays.
    n_static: number of static points.
    n_robot: number of robot points.

  Returns:
    Dict with visibility stats per camera and overall.
  """
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


# 3. Reprojection error (3D → 2D projected vs stored 2D tracks)

def compute_reprojection_error(traj_3d, traj_2d, vis_2d,
                               intrinsics, extrinsics_w2c):
  """Vectorized 2D reprojection error across all frames.

  Projects 3D world points into camera space and compares against
  stored 2D track positions.

  Args:
    traj_3d: (T, N, 3) world-space 3D trajectories.
    traj_2d: (T, N, 2) stored 2D track positions.
    vis_2d: (T, N) bool visibility mask.
    intrinsics: (4,) array [fx, fy, cx, cy].
    extrinsics_w2c: (T, 4, 4) world-to-camera transforms.

  Returns:
    1-D np.ndarray of per-measurement reprojection errors in pixels,
    or empty array if no valid measurements.
  """
  T, N, _ = traj_3d.shape
  fx, fy, cx, cy = intrinsics

  # Homogeneous coords: (T, N, 4)
  ones = np.ones((T, N, 1), dtype=traj_3d.dtype)
  pts_homo = np.concatenate([traj_3d, ones], axis=2)

  # Batch transform: (T, 4, 4) @ (T, N, 4) -> (T, N, 4)
  pts_cam = np.einsum('tij,tnj->tni', extrinsics_w2c, pts_homo)

  z = pts_cam[:, :, 2]  # (T, N)
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
  """Compute reprojection error stats across all cameras.

  Loads per-camera intrinsics and extrinsics_w2c from disk, then
  computes reprojection error for each camera.

  Args:
    traj_3d: (T, N, 3) world-space 3D trajectories.
    per_cam_tracks_2d: dict of cam_id -> {'traj_2d': (T,N,2), 'vis_2d': (T,N)}
        or None to load from tracks_root.
    tracks_root: path to tracks output directory.
    episode_id: episode identifier.

  Returns:
    Dict with reproj_mean_px, reproj_median_px, reproj_p95_px.
  """
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


# 4. Robot motion statistics

def compute_motion_stats(scene_constants):
  """Compute robot motion amplitude metrics from joint/gripper data.

  Args:
    scene_constants: Scene data dict (must have 'robot' sub-dict).

  Returns:
    Dict with motion metrics:
      joint_range_mean_rad: mean per-joint range (max - min) in radians.
      joint_range_max_rad: max joint range across all 7 joints.
      joint_std_mean_rad: mean per-joint std dev.
      gripper_range: range of gripper opening width.
      ee_travel_m: total end-effector path length in metres.
      n_frames: number of frames.
  """
  robot = scene_constants["robot"]
  joints = robot["joint_positions"]  # (T, 7)
  gripper = robot["gripper_positions"]  # (T,)

  # Joint motion
  joint_ranges = joints.max(axis=0) - joints.min(axis=0)  # (7,)
  joint_stds = joints.std(axis=0)  # (7,)

  # End-effector travel distance
  T_ee_all = robot["T_ee_base_all"]  # (T, 4, 4)
  ee_positions = T_ee_all[:, :3, 3]  # (T, 3)
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


# 5. Scene metadata

def compute_scene_metadata(scene_constants):
  """Extract scene-level metadata for stratification.

  Args:
    scene_constants: Scene data dict.

  Returns:
    Dict with:
      site: lab/institution name (e.g. "TRI", "ILIAD").
      robot_id: robot serial hash.
      n_cameras: number of cameras.
      image_resolution: "HxW" string.
      wrist_serial: wrist camera serial.
  """
  ep_id = scene_constants["meta"]["episode_id"]
  parts = ep_id.split("+")
  site = parts[0] if parts else "UNKNOWN"
  robot_id = parts[1] if len(parts) > 1 else "UNKNOWN"

  camera_ids = list(scene_constants["camera"].keys())
  first_cam = scene_constants["camera"][camera_ids[0]]

  # Get image dimensions — prefer raw_depth (always available),
  # fall back to video_rgb or first_frame_rgb.
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


# 6. Depth coverage statistics

def compute_depth_coverage_stats(scene_constants):
  """Compute depth map coverage and statistics per camera.

  Measures what fraction of pixels have valid (> 0.05m) depth.
  A proxy for scene complexity and depth quality.

  Args:
    scene_constants: Scene data dict.

  Returns:
    Dict with depth coverage stats per camera and overall.
  """
  stats = {}
  all_coverage = []

  for cam_id in scene_constants["camera"]:
    cam_data = scene_constants["camera"][cam_id]
    if "raw_depth" not in cam_data:
      continue

    depth = cam_data["raw_depth"]  # (T, H, W)
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


# 7. Aggregate all metrics for one episode

def evaluate_episode(scene_constants, scene_state, device,
                     final_traj_3d=None, final_per_cam_vis=None,
                     n_static=0, n_robot=0,
                     tracks_root=None,
                     compute_extrinsics_metrics=True,
                     pb_renderer=None):
  """Compute all quality metrics for a single episode.

  Orchestrates all metric functions into a single flat dict suitable
  for CSV export.

  Args:
    scene_constants: Scene data dict.
    scene_state: Extrinsics state dict.
    device: Torch device.
    final_traj_3d: (T, N, 3) 3D trajectories, or None to skip track metrics.
    final_per_cam_vis: dict of (T, N) visibility, or None.
    n_static: number of static points.
    n_robot: number of robot points.
    tracks_root: path to tracks directory (for reprojection error).
    compute_extrinsics_metrics: whether to compute extrinsics quality.
    pb_renderer: optional PyBulletRenderer for robot depth losses.

  Returns:
    Flat dict mapping metric names to float values.
  """
  ep_id = scene_constants["meta"]["episode_id"]
  metrics = {"episode_id": ep_id}

  # --- Scene metadata ---
  metrics.update(compute_scene_metadata(scene_constants))

  # --- Motion stats ---
  metrics.update(compute_motion_stats(scene_constants))

  # --- Depth coverage ---
  metrics.update(compute_depth_coverage_stats(scene_constants))

  # --- Extrinsics metrics (chamfer, robot depth loss) ---
  if compute_extrinsics_metrics:
    try:
      ext_metrics = evaluate_extrinsics(
          scene_constants, scene_state, device,
          pb_renderer=pb_renderer)
      metrics.update(ext_metrics)
    except Exception as e:
      metrics["extrinsics_error"] = str(e)

  # --- Track metrics ---
  if final_traj_3d is not None and final_per_cam_vis is not None:
    metrics["n_static"] = n_static
    metrics["n_robot"] = n_robot
    metrics["n_total_tracks"] = n_static + n_robot
    metrics["n_track_frames"] = final_traj_3d.shape[0]

    # Depth consistency
    metrics.update(compute_track_depth_consistency(
        scene_constants, scene_state,
        final_traj_3d, final_per_cam_vis, n_static, n_robot))

    # Visibility
    metrics.update(compute_track_visibility_stats(
        final_per_cam_vis, n_static, n_robot))

    # Reprojection error (requires saved intrinsics/extrinsics_w2c on disk)
    if tracks_root is not None:
      try:
        reproj = compute_reprojection_stats(
            final_traj_3d, None, tracks_root, ep_id)
        metrics.update(reproj)
      except Exception as e:
        metrics["reproj_error"] = str(e)

  return metrics


# ===========================================================================
# IO helpers
# ===========================================================================

def load_track_data(episode_id, tracks_root):
  """Load pre-computed track data from disk.

  Returns:
    (final_traj_3d, final_per_cam_vis, n_static, n_robot) or
    (None, None, 0, 0) if not found.
  """
  ep_dir = os.path.abspath(
      os.path.expanduser(os.path.join(tracks_root, episode_id)))

  tracks_path = os.path.join(ep_dir, "tracks_3d.npz")
  meta_path = os.path.join(ep_dir, "track_metadata.npz")

  if not os.path.exists(tracks_path) or not os.path.exists(meta_path):
    return None, None, 0, 0

  tracks_data = np.load(tracks_path)
  meta_data = np.load(meta_path)

  final_traj_3d = tracks_data["traj_3d"]  # (T, N, 3)
  n_static = int(meta_data["n_static"])
  n_robot = int(meta_data["n_robot"])

  # Load per-camera visibility
  final_per_cam_vis = {}
  for cam_dir_name in os.listdir(ep_dir):
    cam_dir = os.path.join(ep_dir, cam_dir_name)
    vis_path = os.path.join(cam_dir, "tracks_2d.npz")
    if os.path.isdir(cam_dir) and os.path.exists(vis_path):
      cam_data = np.load(vis_path)
      final_per_cam_vis[cam_dir_name] = cam_data["vis_2d"]

  if not final_per_cam_vis:
    return None, None, 0, 0

  return final_traj_3d, final_per_cam_vis, n_static, n_robot


# ===========================================================================
# Main entry points
# ===========================================================================

def evaluate_single_episode(episode_id, depth_root, extrinsics_root,
                            tracks_root, device, pb_renderer):
  """Evaluate all metrics for one episode.

  Returns:
    dict of metric_name → value, or None on failure.
  """
  # Load depth data (with video for depth coverage stats)
  scene_constants = load_depth_data(
      episode_id, depth_root, load_video="first_frame")
  scene_state = load_extrinsics(scene_constants, extrinsics_root)

  # Load track data (if available)
  final_traj_3d, final_per_cam_vis, n_static, n_robot = \
      load_track_data(episode_id, tracks_root)

  has_tracks = final_traj_3d is not None

  # Compute all metrics
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


def main():
  parser = argparse.ArgumentParser(
      description="Batch quality metrics evaluation for DROID episodes")
  parser.add_argument("--rank", type=int, default=0,
                      help="Rank of this process (for multi-GPU sharding)")
  parser.add_argument("--world_size", type=int, default=1,
                      help="Total number of processes")
  parser.add_argument("--limit", type=int, default=-1,
                      help="Limit total episodes to process (-1 = all)")
  parser.add_argument("--depth_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/depth")
  parser.add_argument("--extrinsics_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/extrinsics")
  parser.add_argument("--tracks_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/tracks")
  parser.add_argument("--output_dir", type=str,
                      default="~/droid_data/output/mv-tap/droid/metrics")
  parser.add_argument("--require_tracks", action="store_true",
                      help="Only evaluate episodes with track data")
  args = parser.parse_args()

  # Discover available episodes (from depth output, the first pipeline stage)
  depth_abs = os.path.abspath(os.path.expanduser(args.depth_root))
  ext_abs = os.path.abspath(os.path.expanduser(args.extrinsics_root))
  tracks_abs = os.path.abspath(os.path.expanduser(args.tracks_root))
  output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
  os.makedirs(output_dir, exist_ok=True)

  # Find episodes that have both depth and extrinsics.
  # Avoid per-entry os.path.isdir (slow on gcsfuse); just use listdir.
  depth_eps = set(os.listdir(depth_abs))
  ext_eps = set(os.listdir(ext_abs))
  available_eps = sorted(depth_eps & ext_eps)

  if args.require_tracks and os.path.exists(tracks_abs):
    tracks_eps = set(os.listdir(tracks_abs))
    available_eps = sorted(set(available_eps) & tracks_eps)

  print(f"Found {len(available_eps)} episodes with depth + extrinsics")

  # Deterministic shuffle for load balancing
  random.seed(42)
  random.shuffle(available_eps)

  if args.limit > 0:
    available_eps = available_eps[:args.limit]

  # Shard across ranks
  target_eps = available_eps[args.rank::args.world_size]
  print(f"Rank {args.rank}/{args.world_size}: "
        f"{len(target_eps)} episodes assigned")

  # Setup
  device = get_accelerator()
  pb_renderer = PyBulletRenderer()

  # Output CSV (shared across all ranks, file-locked)
  csv_path = os.path.join(output_dir, "metrics.csv")

  # Check which episodes are already evaluated (resume-friendly)
  done_eps = set()
  if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
    with open(csv_path, "r") as f:
      reader = csv.DictReader(f)
      for row in reader:
        done_eps.add(row.get("episode_id", ""))

  todo_eps = [ep for ep in target_eps if ep not in done_eps]
  print(f"{len(todo_eps)} remaining ({len(done_eps)} already done)")

  succeeded = 0
  failed = 0

  for idx, ep_id in enumerate(todo_eps):
    t0 = time.time()
    print(f"\n[{idx + 1}/{len(todo_eps)}] Episode: {ep_id}")

    try:
      metrics = evaluate_single_episode(
          ep_id, args.depth_root, args.extrinsics_root,
          args.tracks_root, device, pb_renderer)

      if metrics is None:
        print(f"  [WARN] Skipped (no data)")
        failed += 1
        continue

      # Write to CSV (append mode, file-locked, header-safe)
      with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        # Check if header exists (another rank may have written it)
        needs_header = (f.tell() == 0)
        if not needs_header:
          f.seek(0, 2)  # seek to end
          needs_header = (f.tell() == 0)
        writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
        if needs_header:
          writer.writeheader()
        writer.writerow(metrics)
        fcntl.flock(f, fcntl.LOCK_UN)

      elapsed = time.time() - t0
      chamfer = metrics.get("chamfer_total", float("nan"))
      depth_res = metrics.get("depth_residual_overall_median_mm", float("nan"))
      print(f"  [OK] Done in {elapsed:.1f}s | "
            f"chamfer={chamfer:.4f} | "
            f"depth_residual_median={depth_res:.1f}mm")
      succeeded += 1

    except Exception as e:
      print(f"  [FAIL] Failed: {e}")
      traceback.print_exc()
      failed += 1

      # Log failure
      fail_path = os.path.join(output_dir, "failures.txt")
      with open(fail_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(f"{ep_id}\t{str(e)}\n")
        fcntl.flock(f, fcntl.LOCK_UN)

  print(f"\nEvaluation complete!")
  print(f"   Succeeded: {succeeded}/{len(todo_eps)}")
  print(f"   Failed:    {failed}/{len(todo_eps)}")
  print(f"   Output:    {csv_path}")


if __name__ == "__main__":
  main()
