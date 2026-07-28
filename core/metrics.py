"""Quality metrics for episode evaluation.

Pure computation functions with no visualization imports.
Designed for batch evaluation on GCP — all functions accept
pre-loaded scene_constants / scene_state / track data and return
plain dicts or scalars.

Mirrors the inline metrics computed in pipeline.ipynb cells 2e and 3c,
plus additional motion/coverage statistics for episode selection.
"""

import numpy as np


# ===========================================================================
# 1. Depth consistency (track 3D → projected depth vs observed depth)
# ===========================================================================

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
  from core.geometry import project_points_np

  if len(pts_3d) == 0:
    return np.array([], dtype=np.float32)
  u_proj, v_proj, z_proj = project_points_np(pts_3d, K, extrinsics)
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


# ===========================================================================
# 2. Track visibility statistics
# ===========================================================================

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


# ===========================================================================
# 3. Robot motion statistics
# ===========================================================================

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


# ===========================================================================
# 4. Scene metadata
# ===========================================================================

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


# ===========================================================================
# 5. Depth coverage statistics
# ===========================================================================

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


# ===========================================================================
# 6. Aggregate all metrics for one episode
# ===========================================================================

def evaluate_episode(scene_constants, scene_state, device,
                     final_traj_3d=None, final_per_cam_vis=None,
                     n_static=0, n_robot=0,
                     compute_extrinsics_metrics=True,
                     pb_renderer=None):
  """Compute all quality metrics for a single episode.

  Orchestrates all metric functions into a single flat dict suitable
  for CSV export.  Extrinsics metrics (chamfer, robot_loss) are
  delegated to ``compute_extrinsics.evaluate_extrinsics`` if available.

  Args:
    scene_constants: Scene data dict.
    scene_state: Extrinsics state dict.
    device: Torch device.
    final_traj_3d: (T, N, 3) 3D trajectories, or None to skip track metrics.
    final_per_cam_vis: dict of (T, N) visibility, or None.
    n_static: number of static points.
    n_robot: number of robot points.
    compute_extrinsics_metrics: whether to compute extrinsics quality.
    pb_renderer: optional PyBulletRenderer to reuse.

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
      from compute_extrinsics import evaluate_extrinsics
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

  return metrics
