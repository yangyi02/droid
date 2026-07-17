"""DROID Stage 3 v2: Static Background + Robot Track Computation.

Key design changes from v1 (compute_tracks.py):
  - **No tracker model** (CoTracker/TAPNext) for background points.
  - **Static prior**: Background points are assumed to be stationary in world
    coordinates. Their 2D tracks in each view are obtained purely by projecting
    the fixed 3D world position through the per-frame camera extrinsics.
  - **Full video length**: No truncation to 40 frames.
  - **Multi-view depth consensus**: 3D positions are obtained by unprojecting
    from multiple views and cross-validating via nearest-neighbor matching.

Two point types:
  Track A (Background/Static): Multi-view depth consensus → fixed world 3D →
      project to 2D per-view per-frame using extrinsics.
  Track B (Robot): URDF forward kinematics (identical to compute_tracks.py).

Output format (same as v1 for downstream compatibility):
  final_traj_3d:        np.float32 (T, N, 3)   world coordinates
  final_vis_global:     np.bool_   (T, N)       global visibility
  final_per_cam_tracks: {cam_id: np.float32 (T, N, 2)}  per-view 2D tracks
  final_per_cam_vis:    {cam_id: np.bool_   (T, N)}      per-view visibility
"""

import argparse
import os
import warnings
from collections import defaultdict

import cv2
import numpy as np
from scipy.spatial import cKDTree

from core.geometry import project_points_np, unproject_points_np
from core.io import get_accelerator, load_depth_data, load_extrinsics
from core.physics import PyBulletRenderer
from core.tracking import URDFKinematicsTracker


# ===========================================================================
# Phase 1: Extract candidate static 3D points from each view at frame t=0
# ===========================================================================

def _unproject_dense_points(cam_data, extrinsic_t0, t=0,
                            min_depth=0.05, max_depth=5.0,
                            stride=1):
  """Unproject depth map at frame t to 3D world points.

  Args:
    cam_data: Camera data dict with 'raw_depth', 'K_mat', 'video_rgb'.
    extrinsic_t0: 4x4 camera-to-world transform at frame t.
    t: Frame index.
    min_depth: Minimum valid depth (meters).
    max_depth: Maximum valid depth (meters).
    stride: Subsampling stride for dense unprojection.

  Returns:
    pts_3d: (M, 3) world coordinates.
    uv: (M, 2) pixel coordinates (x, y).
    rgb: (M, 3) RGB values.
  """
  depth = cam_data["raw_depth"][t]
  h, w = depth.shape
  K = cam_data["K_mat"]

  # Dense pixel grid (with optional stride)
  ys = np.arange(0, h, stride)
  xs = np.arange(0, w, stride)
  xx, yy = np.meshgrid(xs, ys)
  u_flat = xx.ravel().astype(np.float32)
  v_flat = yy.ravel().astype(np.float32)

  # Sample depth
  ui = u_flat.astype(int)
  vi = v_flat.astype(int)
  z = depth[vi, ui]

  # Filter valid depth
  valid = (z > min_depth) & (z < max_depth)
  u_val = u_flat[valid]
  v_val = v_flat[valid]
  z_val = z[valid]

  # Unproject to world
  pts_3d = unproject_points_np(u_val, v_val, z_val, K, extrinsic_t0)

  # RGB
  rgb_img = cam_data["video_rgb"][t]
  rgb = rgb_img[v_val.astype(int), u_val.astype(int)]

  uv = np.stack([u_val, v_val], axis=-1)
  return pts_3d, uv, rgb


def phase1_find_static_candidates(scene_constants, scene_state, pb_renderer,
                                  num_keyframes=5, stride=4,
                                  match_radius=0.005,
                                  min_views_agree=2,
                                  num_points=1200,
                                  safe_margin=15):
  """Find reliable static background 3D points via multi-view depth consensus.

  For each pair of static (non-wrist) cameras, we:
    1. Unproject depth from both views at a keyframe.
    2. Find 3D nearest-neighbor matches (< match_radius).
    3. Filter out robot-occupied pixels.
    4. Average matched 3D positions for better accuracy.

  Then we aggregate across all keyframes and camera pairs, dedup, and
  subsample to num_points.

  Args:
    scene_constants: Scene data dict.
    scene_state: Extrinsics dict.
    pb_renderer: PyBulletRenderer instance.
    num_keyframes: Number of keyframes to use for multi-frame consensus.
    stride: Subsampling stride for dense depth unprojection.
    match_radius: Max 3D distance (m) to consider two points the same.
    min_views_agree: Minimum number of views that must agree.
    num_points: Target number of static points to output.
    safe_margin: Dilation kernel size for robot mask.

  Returns:
    static_pts_3d: (N, 3) world coordinates of static points.
    static_rgb: (N, 3) RGB colors.
  """
  camera_ids = list(scene_constants["camera"].keys())
  wrist_serial = scene_constants["meta"].get("wrist_serial")
  T_frames = len(scene_constants["camera"][camera_ids[0]]["video_rgb"])

  # Identify static (non-wrist) cameras
  static_cams = [c for c in camera_ids if c != wrist_serial]
  if len(static_cams) < 2:
    print("  ⚠️ Need at least 2 static cameras for consensus.")
    return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

  # Select keyframes spread across the video
  keyframe_indices = np.linspace(0, T_frames - 1, num_keyframes,
                                 dtype=int).tolist()
  # Remove duplicates
  keyframe_indices = sorted(set(keyframe_indices))
  print(f"  🕐 Using {len(keyframe_indices)} keyframes: {keyframe_indices}")

  # Collect all consensus 3D points
  all_consensus_pts = []
  all_consensus_rgb = []

  for t_k in keyframe_indices:
    print(f"\n  📸 Keyframe t={t_k}:")

    # Build robot mask for this keyframe
    pb_renderer.update_robot_pose(
        scene_constants["robot"]["joint_positions"][t_k],
        gripper_state=scene_constants["robot"]["gripper_positions"][t_k])

    # Unproject each static camera
    per_cam_pts = {}
    per_cam_uv = {}
    per_cam_rgb = {}

    for cam_id in static_cams:
      cam_data = scene_constants["camera"][cam_id]
      ext = scene_state[cam_id]["extrinsics"][t_k]
      h_img, w_img = cam_data["video_rgb"][0].shape[:2]

      # Robot mask (dilated)
      robot_mask = pb_renderer.render_mask(ext, cam_data["K_mat"], w_img, h_img)
      kernel = np.ones((safe_margin, safe_margin), np.uint8)
      robot_mask_dilated = cv2.dilate(
          robot_mask.astype(np.uint8), kernel, iterations=1) > 0

      # Unproject dense points
      pts_3d, uv, rgb = _unproject_dense_points(
          cam_data, ext, t=t_k, stride=stride)

      # Filter out robot points
      ui = np.clip(uv[:, 0].astype(int), 0, w_img - 1)
      vi = np.clip(uv[:, 1].astype(int), 0, h_img - 1)
      is_env = ~robot_mask_dilated[vi, ui]

      pts_3d = pts_3d[is_env]
      uv = uv[is_env]
      rgb = rgb[is_env]

      per_cam_pts[cam_id] = pts_3d
      per_cam_uv[cam_id] = uv
      per_cam_rgb[cam_id] = rgb
      print(f"    [{cam_id}] {len(pts_3d)} env points")

    # Pairwise cross-view matching
    for i, cam_a in enumerate(static_cams):
      for cam_b in static_cams[i + 1:]:
        pts_a = per_cam_pts.get(cam_a)
        pts_b = per_cam_pts.get(cam_b)
        if pts_a is None or pts_b is None:
          continue
        if len(pts_a) == 0 or len(pts_b) == 0:
          continue

        # KD-tree matching
        tree_a = cKDTree(pts_a)
        dists, indices = tree_a.query(pts_b, k=1)
        matched = dists < match_radius

        if np.sum(matched) == 0:
          print(f"    [{cam_a[:8]}↔{cam_b[:8]}] 0 matches")
          continue

        # Average matched 3D positions for better precision
        avg_pts = (pts_a[indices[matched]] + pts_b[matched]) / 2.0
        avg_rgb = per_cam_rgb[cam_a][indices[matched]]

        all_consensus_pts.append(avg_pts)
        all_consensus_rgb.append(avg_rgb)
        print(f"    [{cam_a[:8]}↔{cam_b[:8]}] {len(avg_pts)} consensus matches")

  if not all_consensus_pts:
    print("  ⚠️ No consensus points found.")
    return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

  # Concatenate all consensus points
  all_pts = np.concatenate(all_consensus_pts, axis=0)
  all_rgb = np.concatenate(all_consensus_rgb, axis=0)
  print(f"\n  📊 Total raw consensus points: {len(all_pts)}")

  # Dedup via voxel grid
  dedup_pts, dedup_rgb = _voxel_dedup(all_pts, all_rgb,
                                       voxel_size=match_radius * 2)
  print(f"  📊 After dedup: {len(dedup_pts)}")

  # Verify static-ness across multiple keyframes:
  # Project each candidate point into every static camera at every keyframe
  # and check depth consistency
  if len(keyframe_indices) > 1:
    dedup_pts, dedup_rgb = _verify_static_across_frames(
        dedup_pts, dedup_rgb, scene_constants, scene_state,
        static_cams, keyframe_indices, pb_renderer,
        depth_tolerance=0.03, min_consistent_frames=max(1, len(keyframe_indices) // 2),
        safe_margin=safe_margin)
    print(f"  📊 After multi-frame verification: {len(dedup_pts)}")

  # Subsample if too many
  if len(dedup_pts) > num_points:
    idx = np.random.permutation(len(dedup_pts))[:num_points]
    dedup_pts = dedup_pts[idx]
    dedup_rgb = dedup_rgb[idx]
    print(f"  📊 Subsampled to {num_points} points")

  print(f"  ✅ Found {len(dedup_pts)} static background points")
  return dedup_pts, dedup_rgb


def _voxel_dedup(pts, rgb, voxel_size=0.01):
  """Voxel-grid deduplication: keep median position per voxel."""
  if len(pts) == 0:
    return pts, rgb

  # Quantize to voxel grid
  voxel_indices = np.floor(pts / voxel_size).astype(np.int64)
  # Create unique key per voxel
  keys = (voxel_indices[:, 0].astype(np.int64) * 1000000 +
          voxel_indices[:, 1].astype(np.int64) * 1000 +
          voxel_indices[:, 2].astype(np.int64))

  unique_keys, inverse = np.unique(keys, return_inverse=True)
  N_unique = len(unique_keys)

  out_pts = np.zeros((N_unique, 3), dtype=np.float32)
  out_rgb = np.zeros((N_unique, 3), dtype=np.uint8)

  for i in range(N_unique):
    mask = inverse == i
    out_pts[i] = np.median(pts[mask], axis=0)
    out_rgb[i] = np.median(rgb[mask].astype(np.float32), axis=0).astype(np.uint8)

  return out_pts, out_rgb


def _verify_static_across_frames(pts, rgb, scene_constants, scene_state,
                                  static_cams, keyframe_indices, pb_renderer,
                                  depth_tolerance=0.03,
                                  min_consistent_frames=2,
                                  safe_margin=15):
  """Verify that candidate static points are depth-consistent across frames.

  A point is kept only if its projected depth matches the observed depth
  in at least one static camera for at least min_consistent_frames keyframes.
  """
  N = len(pts)
  if N == 0:
    return pts, rgb

  consistent_count = np.zeros(N, dtype=int)

  for t_k in keyframe_indices:
    # Any-cam depth consistency at this frame
    any_cam_ok = np.zeros(N, dtype=bool)

    for cam_id in static_cams:
      cam_data = scene_constants["camera"][cam_id]
      ext = scene_state[cam_id]["extrinsics"][t_k]
      K = cam_data["K_mat"]
      h_img, w_img = cam_data["video_rgb"][0].shape[:2]

      # Robot mask
      robot_mask = pb_renderer.render_mask(ext, K, w_img, h_img)
      kernel = np.ones((safe_margin, safe_margin), np.uint8)
      robot_mask_dilated = cv2.dilate(
          robot_mask.astype(np.uint8), kernel, iterations=1) > 0

      # Project points
      u, v, z_pred = project_points_np(pts, K, ext)
      ui = np.clip(np.round(u).astype(int), 0, w_img - 1)
      vi = np.clip(np.round(v).astype(int), 0, h_img - 1)

      in_bounds = ((u >= 0) & (u < w_img) &
                   (v >= 0) & (v < h_img) & (z_pred > 0))
      not_robot = ~robot_mask_dilated[vi, ui]

      z_obs = cam_data["raw_depth"][t_k, vi, ui]
      depth_ok = (z_obs > 0.05) & (np.abs(z_pred - z_obs) < depth_tolerance)

      cam_ok = in_bounds & not_robot & depth_ok
      any_cam_ok |= cam_ok

    consistent_count += any_cam_ok.astype(int)

  keep = consistent_count >= min_consistent_frames
  return pts[keep], rgb[keep]


# ===========================================================================
# Phase 2: Project static 3D points to all views (static prior)
# ===========================================================================

def phase2_project_static_tracks(static_pts_3d, scene_constants, scene_state,
                                  pb_renderer, depth_tolerance=0.05,
                                  safe_margin=15):
  """Project static 3D world points to 2D per-view per-frame.

  Since these points are static in world coordinates:
    - In static views, they stay (approximately) fixed in pixel space.
    - In the wrist view, they move as the camera moves.

  Visibility is determined by:
    1. In-bounds check (projected pixel within image).
    2. Depth consistency (projected depth ≈ observed depth).
    3. Not occluded by the robot (via PyBullet mask).

  Args:
    static_pts_3d: (N, 3) world coordinates.
    scene_constants: Scene data dict.
    scene_state: Extrinsics dict.
    pb_renderer: PyBulletRenderer instance.
    depth_tolerance: Max depth discrepancy (m) for visibility.
    safe_margin: Robot mask dilation kernel size.

  Returns:
    per_cam_tracks: {cam_id: np.float32 (T, N, 2)} 2D tracks.
    per_cam_vis: {cam_id: np.bool_ (T, N)} visibility masks.
  """
  camera_ids = list(scene_constants["camera"].keys())
  T_frames = len(scene_constants["camera"][camera_ids[0]]["video_rgb"])
  N = len(static_pts_3d)

  print(f"\n{'=' * 60}")
  print(f"📐 Phase 2: Project {N} Static Points → {len(camera_ids)} Views × {T_frames} Frames")
  print(f"{'=' * 60}")

  per_cam_tracks = {}
  per_cam_vis = {}

  for cam_id in camera_ids:
    cam_data = scene_constants["camera"][cam_id]
    K = cam_data["K_mat"]
    h_img, w_img = cam_data["video_rgb"][0].shape[:2]

    tracks = np.zeros((T_frames, N, 2), dtype=np.float32)
    vis = np.zeros((T_frames, N), dtype=bool)

    for t in range(T_frames):
      ext = scene_state[cam_id]["extrinsics"][t]

      # Project 3D → 2D
      u, v, z_pred = project_points_np(static_pts_3d, K, ext)
      tracks[t, :, 0] = u
      tracks[t, :, 1] = v

      # Bounds check
      in_bounds = ((u >= 0) & (u < w_img) &
                   (v >= 0) & (v < h_img) & (z_pred > 0))

      # Depth consistency
      ui = np.clip(np.round(u).astype(int), 0, w_img - 1)
      vi = np.clip(np.round(v).astype(int), 0, h_img - 1)
      z_obs = cam_data["raw_depth"][t, vi, ui]
      depth_ok = (z_obs > 0.05) & (np.abs(z_pred - z_obs) < depth_tolerance)

      # Robot occlusion check
      pb_renderer.update_robot_pose(
          scene_constants["robot"]["joint_positions"][t],
          gripper_state=scene_constants["robot"]["gripper_positions"][t])
      robot_mask = pb_renderer.render_mask(ext, K, w_img, h_img)
      kernel = np.ones((safe_margin, safe_margin), np.uint8)
      robot_mask_dilated = cv2.dilate(
          robot_mask.astype(np.uint8), kernel, iterations=1) > 0
      not_robot = ~robot_mask_dilated[vi, ui]

      vis[t] = in_bounds & depth_ok & not_robot

    per_cam_tracks[cam_id] = tracks
    per_cam_vis[cam_id] = vis

    vis_rate = vis.mean() * 100
    print(f"  ✅ [{cam_id}] avg visibility: {vis_rate:.1f}%")

  return per_cam_tracks, per_cam_vis


# ===========================================================================
# Phase 3: Robot tracks (identical to compute_tracks.py Track B)
# ===========================================================================

def phase3_robot_tracks(scene_constants, scene_state, pb_renderer):
  """Extract robot surface tracks via URDF forward kinematics.

  This is identical to Track B in compute_tracks.py.

  Returns:
    robot_traj_3d:        (T, N_robot, 3)
    robot_per_cam_tracks: {cam: (T, N_robot, 2)}
    robot_per_cam_vis:    {cam: (T, N_robot)}
    n_robot:              int
  """
  camera_ids = list(scene_constants["camera"].keys())
  T_frames = len(scene_constants["camera"][camera_ids[0]]["video_rgb"])

  print(f"\n{'=' * 60}")
  print("🦾 Phase 3: URDF Forward Kinematics Robot Tracking")
  print(f"{'=' * 60}")

  urdf_tracker = URDFKinematicsTracker(pb_renderer)
  robot_traj_3d_all = []
  robot_per_cam_tracks_all = {cam: [] for cam in camera_ids}
  robot_per_cam_vis_all = {cam: [] for cam in camera_ids}

  for src_cam in camera_ids:
    traj_3d_rob, traj_2d_rob, vis_rob, robot_indices = \
        urdf_tracker.extract_robot_tracks(
            src_cam, scene_constants, scene_state)

    if traj_3d_rob is None or len(robot_indices) == 0:
      continue

    # Project robot 3D to all views
    rob_per_cam_2d, rob_per_cam_vis = \
        urdf_tracker.project_to_all_views(
            traj_3d_rob, scene_constants, scene_state)

    # Use source view's native 2D for itself
    rob_per_cam_2d[src_cam] = traj_2d_rob
    rob_per_cam_vis[src_cam] = vis_rob

    robot_traj_3d_all.append(traj_3d_rob)
    for cam in camera_ids:
      robot_per_cam_tracks_all[cam].append(rob_per_cam_2d[cam])
      robot_per_cam_vis_all[cam].append(rob_per_cam_vis[cam])

  if robot_traj_3d_all:
    robot_traj_3d = np.concatenate(robot_traj_3d_all, axis=1)
    robot_per_cam_tracks = {
        cam: np.concatenate(robot_per_cam_tracks_all[cam], axis=1)
        for cam in camera_ids}
    robot_per_cam_vis = {
        cam: np.concatenate(robot_per_cam_vis_all[cam], axis=1)
        for cam in camera_ids}
    n_robot = robot_traj_3d.shape[1]
    print(f"  🦾 Total robot points: {n_robot}")
  else:
    robot_traj_3d = np.zeros((T_frames, 0, 3), dtype=np.float32)
    robot_per_cam_tracks = {
        cam: np.zeros((T_frames, 0, 2), dtype=np.float32)
        for cam in camera_ids}
    robot_per_cam_vis = {
        cam: np.zeros((T_frames, 0), dtype=bool)
        for cam in camera_ids}
    n_robot = 0
    print("  ⚠️ No robot points extracted.")

  return robot_traj_3d, robot_per_cam_tracks, robot_per_cam_vis, n_robot


# ===========================================================================
# Phase 4: Merge static + robot tracks
# ===========================================================================

def phase4_merge(static_pts_3d, static_per_cam_tracks, static_per_cam_vis,
                 robot_traj_3d, robot_per_cam_tracks, robot_per_cam_vis,
                 camera_ids, T_frames):
  """Merge static background and robot tracks.

  Returns the same output format as compute_tracks.py:
    final_traj_3d:        (T, N_total, 3)
    final_vis_global:     (T, N_total)
    final_per_cam_tracks: {cam: (T, N_total, 2)}
    final_per_cam_vis:    {cam: (T, N_total)}
    n_static:             int
    n_robot:              int
  """
  n_static = len(static_pts_3d)
  n_robot = robot_traj_3d.shape[1]

  print(f"\n{'=' * 60}")
  print("🔗 Phase 4: Merging Static Background + Robot Tracks")
  print(f"{'=' * 60}")
  print(f"  📊 Static: {n_static} | Robot: {n_robot} | Total: {n_static + n_robot}")

  # Static 3D trajectory: constant across all frames
  if n_static > 0:
    static_traj_3d = np.broadcast_to(
        static_pts_3d[None, :, :], (T_frames, n_static, 3)
    ).copy()
  else:
    static_traj_3d = np.zeros((T_frames, 0, 3), dtype=np.float32)

  # Static visibility (global = visible in at least 1 camera)
  if n_static > 0:
    static_vis_global = np.zeros((T_frames, n_static), dtype=bool)
    for cam in camera_ids:
      static_vis_global |= static_per_cam_vis[cam]
  else:
    static_vis_global = np.zeros((T_frames, 0), dtype=bool)

  # Robot visibility (global = visible in at least 1 camera)
  if n_robot > 0:
    robot_vis_global = np.zeros((T_frames, n_robot), dtype=bool)
    for cam in camera_ids:
      robot_vis_global |= robot_per_cam_vis[cam]
  else:
    robot_vis_global = np.zeros((T_frames, 0), dtype=bool)

  # Concatenate
  final_traj_3d = np.concatenate([static_traj_3d, robot_traj_3d], axis=1)
  final_vis_global = np.concatenate(
      [static_vis_global, robot_vis_global], axis=1)
  final_per_cam_tracks = {
      cam: np.concatenate(
          [static_per_cam_tracks[cam], robot_per_cam_tracks[cam]], axis=1)
      for cam in camera_ids}
  final_per_cam_vis = {
      cam: np.concatenate(
          [static_per_cam_vis[cam], robot_per_cam_vis[cam]], axis=1)
      for cam in camera_ids}

  return (final_traj_3d, final_vis_global, final_per_cam_tracks,
          final_per_cam_vis, n_static, n_robot)


# ===========================================================================
# Export (same format as compute_tracks.py)
# ===========================================================================

def export_tracks(scene_constants, scene_state, final_traj_3d,
                  final_vis_global, final_per_cam_tracks, final_per_cam_vis,
                  n_static, n_robot,
                  export_root="~/droid_data/output/mv-tap/droid/tracks2"):
  """Serialize tracking results to disk."""
  ep_id = scene_constants["meta"]["episode_id"]
  camera_ids = list(scene_constants["camera"].keys())
  ep_dir = os.path.abspath(
      os.path.expanduser(os.path.join(export_root, ep_id)))
  os.makedirs(ep_dir, exist_ok=True)

  T, N, _ = final_traj_3d.shape

  # Global 3D trajectories
  np.savez_compressed(
      os.path.join(ep_dir, "tracks_3d.npz"),
      traj_3d=final_traj_3d.astype(np.float32),
      vis_global=final_vis_global,
  )

  # Per-camera 2D tracks + visibility
  for cam_id in camera_ids:
    cam_dir = os.path.join(ep_dir, cam_id)
    os.makedirs(cam_dir, exist_ok=True)

    traj_2d = final_per_cam_tracks[cam_id].copy()
    vis = final_per_cam_vis[cam_id].copy()
    traj_2d[~vis] = -1000.0

    np.savez_compressed(
        os.path.join(cam_dir, "tracks_2d.npz"),
        traj_2d=traj_2d.astype(np.float32),
        vis_2d=vis,
    )

    # Intrinsics
    cam_data = scene_constants["camera"][cam_id]
    K = cam_data["K_mat"]
    np.save(os.path.join(cam_dir, "intrinsics.npy"),
            np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                     dtype=np.float32))

    # Extrinsics (w2c)
    np.save(os.path.join(cam_dir, "extrinsics_w2c.npy"),
            np.linalg.inv(
                scene_state[cam_id]["extrinsics"]).astype(np.float32))

  # Track metadata
  np.savez_compressed(
      os.path.join(ep_dir, "track_metadata.npz"),
      n_static=np.array(n_static),
      n_robot=np.array(n_robot),
      # 0 = static background, 1 = robot
      point_type=np.array(
          [0] * n_static + [1] * n_robot, dtype=np.uint8),
  )

  print(f"  💾 Exported {N} tracks × {T} frames to {ep_dir}")
  return ep_dir


# ===========================================================================
# Full Pipeline
# ===========================================================================

def process_episode(episode_id, pb_renderer, device,
                    depth_root, extrinsics_root, export_root,
                    num_static_points=1200, num_keyframes=5):
  """Full static+robot pipeline for a single episode.

  No tracker model needed — background points use static prior only.
  """
  print(f"\n{'=' * 60}")
  print(f"🎬 Processing Episode: {episode_id}")
  print(f"{'=' * 60}")

  # Load data (full video, no truncation)
  scene_constants = load_depth_data(episode_id, depth_root, load_video="full")
  scene_state = load_extrinsics(scene_constants, extrinsics_root)

  camera_ids = list(scene_constants["camera"].keys())
  T_frames = len(scene_constants["camera"][camera_ids[0]]["video_rgb"])
  print(f"  📊 {len(camera_ids)} cameras × {T_frames} frames (full video)")

  # Phase 1: Find static background points
  print(f"\n{'=' * 60}")
  print("🏔️ Phase 1: Multi-View Depth Consensus → Static Points")
  print(f"{'=' * 60}")
  static_pts_3d, static_rgb = phase1_find_static_candidates(
      scene_constants, scene_state, pb_renderer,
      num_keyframes=num_keyframes, num_points=num_static_points)

  # Phase 2: Project static points to all views
  if len(static_pts_3d) > 0:
    static_per_cam_tracks, static_per_cam_vis = phase2_project_static_tracks(
        static_pts_3d, scene_constants, scene_state, pb_renderer)
  else:
    static_per_cam_tracks = {
        cam: np.zeros((T_frames, 0, 2), dtype=np.float32)
        for cam in camera_ids}
    static_per_cam_vis = {
        cam: np.zeros((T_frames, 0), dtype=bool)
        for cam in camera_ids}

  # Phase 3: Robot tracks
  robot_traj_3d, robot_per_cam_tracks, robot_per_cam_vis, n_robot = \
      phase3_robot_tracks(scene_constants, scene_state, pb_renderer)

  # Phase 4: Merge
  (final_traj_3d, final_vis_global, final_per_cam_tracks,
   final_per_cam_vis, n_static, n_robot) = phase4_merge(
      static_pts_3d, static_per_cam_tracks, static_per_cam_vis,
      robot_traj_3d, robot_per_cam_tracks, robot_per_cam_vis,
      camera_ids, T_frames)

  # Export
  export_tracks(scene_constants, scene_state,
                final_traj_3d, final_vis_global,
                final_per_cam_tracks, final_per_cam_vis,
                n_static, n_robot, export_root)

  print(f"\n  ✅ Episode {episode_id}: {n_static} static + {n_robot} robot "
        f"= {n_static + n_robot} tracks exported.")
  return n_static + n_robot


# ===========================================================================
# Standalone CLI
# ===========================================================================

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="DROID Stage 3 v2: Static Background + Robot Tracks")
  parser.add_argument("--depth_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/depth")
  parser.add_argument("--extrinsics_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/extrinsics")
  parser.add_argument("--export_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/tracks2")
  parser.add_argument("--num_static_points", type=int, default=1200,
                      help="Target number of static background points")
  parser.add_argument("--num_keyframes", type=int, default=5,
                      help="Number of keyframes for multi-view consensus")
  parser.add_argument("--limit", type=int, default=-1,
                      help="Limit total number of episodes to process")
  args = parser.parse_args()

  print("🚀 DROID Stage 3 v2: Static Background + Robot Tracks")
  device = get_accelerator()
  pb_renderer = PyBulletRenderer()

  # Discover available episodes from extrinsics output
  ext_abs = os.path.abspath(os.path.expanduser(args.extrinsics_root))
  available_eps = sorted([
      d for d in os.listdir(ext_abs)
      if os.path.isdir(os.path.join(ext_abs, d))
  ])
  if args.limit > 0:
    available_eps = available_eps[:args.limit]

  print(f"📋 Processing {len(available_eps)} episodes")

  for idx, ep_id in enumerate(available_eps):
    print(f"\n🎬 [{idx + 1}/{len(available_eps)}] Episode: {ep_id}")
    try:
      process_episode(
          ep_id, pb_renderer, device,
          args.depth_root, args.extrinsics_root, args.export_root,
          num_static_points=args.num_static_points,
          num_keyframes=args.num_keyframes)
    except Exception as e:
      print(f"  ❌ Episode {ep_id} failed: {e}")
      import traceback
      traceback.print_exc()

  print(f"\n🎉 Stage 3 v2 complete!")
