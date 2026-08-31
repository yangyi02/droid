"""DROID Stage 3 v2: Static Background + Robot Track Computation.

Key design:
  - **Single grid at frame 0**: One uniform grid (e.g., 32×32) per camera at t=0.
    Points on robot -> URDF FK tracking. Points on background -> static prior.
  - **No tracker model** (CoTracker/TAPNext) needed.
  - **Static prior**: Background points are assumed stationary in world
    coordinates. 2D tracks are obtained by projecting fixed 3D positions
    through per-frame extrinsics.
  - **Full video length**: No truncation.
  - **Cross-view depth consensus at t=0**: Background 3D positions are
    verified by nearest-neighbor matching between cameras.

Two point types:
  Track A (Background/Static): Grid at t=0 -> cross-view consensus -> fixed 3D ->
      project to 2D per-view per-frame.
  Track B (Robot): Grid at t=0 -> URDF forward kinematics.

Output format (same as v1 for downstream compatibility):
  final_traj_3d:        np.float32 (T, N, 3)   world coordinates
  final_vis_global:     np.bool_   (T, N)       global visibility
  final_per_cam_tracks: {cam_id: np.float32 (T, N, 2)}  per-view 2D tracks
  final_per_cam_vis:    {cam_id: np.bool_   (T, N)}      per-view visibility
"""

import argparse
import os
import random
import traceback

import cv2
import numpy as np

from core.geometry import project_points, unproject_points
from core.io import OUTPUT_ROOT, get_accelerator, load_depth_data, load_extrinsics
from core.physics import PyBulletRenderer
from core.tracking import URDFKinematicsTracker


# Phase 1: Extract static background 3D points at frame 0

def phase1_find_static_candidates(scene_constants, scene_state, pb_renderer,
                                  match_radius=0.005,
                                  num_points=None,
                                  safe_margin=15,
                                  tau=0.015,
                                  min_run_frames=30,
                                  flicker=0.10):
  """Find reliable static background 3D points via cross-view depth consensus.

  All queries happen at frame 0 only. For each camera (including wrist):
    1. Use ALL pixels at frame 0 (dense sampling).
    2. Filter out robot-occupied pixels.
    3. Unproject remaining pixels to 3D using depth at t=0.

  Then cross-view depth reprojection verifies which 3D points are consistent
  across views. A point is verified if at least one other camera's depth map
  agrees with its predicted depth.

  Finally, candidates are verified across ALL frames with the signed depth gap
  (see `_measure_depth_gaps`). One measurement feeds two independent verdicts:
  a point goes if its support left (some camera saw it on the surface on the
  query frame, then saw past it for `min_run_frames` consecutive frames), or if
  its visibility never settles (`flicker`). Only a *positive* gap — the camera
  seeing past the point — is evidence against it; being occluded says nothing,
  and an absolute-value test cannot tell the two apart.

  Args:
    scene_constants: Scene data dict.
    scene_state: Extrinsics dict.
    pb_renderer: PyBulletRenderer instance.
    match_radius: Max depth discrepancy (m) for cross-view agreement.
    safe_margin: Dilation kernel size for robot mask.
    tau: Metres. Surface tolerance and the size of gap that counts as seeing
        past a point; set by the thinnest object worth catching.
    min_run_frames: Consecutive frames a camera must keep seeing past a point
        before its support is called gone.
    flicker: Drop a point whose line-of-sight flag flips on more than this
        fraction of frames. None disables the test — read
        `_filter_visibility_flicker` before relying on it.

  Returns:
    static_pts_3d: (N, 3) world coordinates of static points.
    static_rgb: (N, 3) RGB colors.
  """
  camera_ids = list(scene_constants["camera"].keys())
  wrist_serial = scene_constants["meta"].get("wrist_serial")

  # Static cameras used for multi-frame verification only
  static_cams = [c for c in camera_ids if c != wrist_serial]
  if len(camera_ids) < 2:
    print("  [WARN] Need at least 2 cameras for consensus.")
    return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

  print(f"  Querying at frame 0 on all {len(camera_ids)} cameras (dense)")

  # Build robot mask at t=0
  pb_renderer.update_robot_pose(
      scene_constants["robot"]["joint_positions"][0],
      gripper_state=scene_constants["robot"]["gripper_positions"][0])

  # Unproject ALL pixels from each camera at t=0 (dense sampling)
  per_cam_pts = {}
  per_cam_rgb = {}

  for cam_id in camera_ids:
    cam_data = scene_constants["camera"][cam_id]
    ext = scene_state[cam_id]["extrinsics"][0]
    K = cam_data["K_mat"]
    h_img, w_img = cam_data["video_rgb"][0].shape[:2]

    # Robot mask (dilated)
    robot_mask = pb_renderer.render_mask(ext, K, w_img, h_img)
    kernel = np.ones((safe_margin, safe_margin), np.uint8)
    robot_mask_dilated = cv2.dilate(
        robot_mask.astype(np.uint8), kernel, iterations=1) > 0

    # Dense pixel grid (all pixels)
    depth = cam_data["raw_depth"][0]
    is_env = ~robot_mask_dilated
    has_depth = (depth > 0.05) & (depth < 5.0)
    valid_mask = is_env & has_depth
    vs, us = np.where(valid_mask)  # row, col indices

    if len(us) == 0:
      print(f"    [{cam_id}] 0 env pixels with depth")
      continue

    z = depth[vs, us]
    u_f = us.astype(np.float32)
    v_f = vs.astype(np.float32)

    # Unproject to 3D
    pts_3d = unproject_points(u_f, v_f, z, K, ext)

    # RGB colors
    rgb = cam_data["video_rgb"][0][vs, us]

    per_cam_pts[cam_id] = pts_3d
    per_cam_rgb[cam_id] = rgb
    n_robot = np.sum(robot_mask_dilated & has_depth)
    n_no_depth = np.sum(is_env & ~has_depth)
    print(f"    [{cam_id}] {len(us)} env / {n_robot} robot / "
          f"{n_no_depth} no-depth  (dense {w_img}×{h_img})")

  # Cross-view depth reprojection consensus at t=0
  # For each camera's grid points, project into every OTHER camera and
  # check depth agreement. A point is verified if at least one other
  # camera's depth map agrees with its predicted depth.
  all_verified_pts = []
  all_verified_rgb = []

  for src_cam in camera_ids:
    pts = per_cam_pts.get(src_cam)
    rgb = per_cam_rgb.get(src_cam)
    if pts is None or len(pts) == 0:
      continue

    # Check each point against all other cameras
    n_agree = np.zeros(len(pts), dtype=int)

    for dst_cam in camera_ids:
      if dst_cam == src_cam:
        continue
      dst_data = scene_constants["camera"][dst_cam]
      dst_ext = scene_state[dst_cam]["extrinsics"][0]
      dst_K = dst_data["K_mat"]
      dst_h, dst_w = dst_data["video_rgb"][0].shape[:2]

      # Project 3D points from src into dst view
      u_d, v_d, z_pred = project_points(pts, dst_K, dst_ext)
      ui_d = np.clip(np.round(u_d).astype(int), 0, dst_w - 1)
      vi_d = np.clip(np.round(v_d).astype(int), 0, dst_h - 1)

      in_bounds = ((u_d >= 0) & (u_d < dst_w) &
                   (v_d >= 0) & (v_d < dst_h) & (z_pred > 0))

      # Read dst depth and check consistency
      z_obs = dst_data["raw_depth"][0, vi_d, ui_d]
      depth_ok = (z_obs > 0.05) & (np.abs(z_pred - z_obs) < match_radius)

      n_agree += (in_bounds & depth_ok).astype(int)

    # Keep points verified by at least 1 other camera
    verified = n_agree >= 1
    n_verified = np.sum(verified)
    print(f"    [{src_cam[:8]}] {n_verified}/{len(pts)} verified by cross-view depth")

    if n_verified > 0:
      all_verified_pts.append(pts[verified])
      all_verified_rgb.append(rgb[verified])

  if not all_verified_pts:
    print("  [WARN] No cross-view verified points found.")
    return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

  # Concatenate and dedup (same 3D point may be verified from multiple cameras)
  all_pts = np.concatenate(all_verified_pts, axis=0)
  all_rgb = np.concatenate(all_verified_rgb, axis=0)
  print(f"\n  Total verified points (pre-dedup): {len(all_pts)}")

  dedup_pts, dedup_rgb = _voxel_dedup(all_pts, all_rgb,
                                       voxel_size=match_radius * 2)
  if len(dedup_pts) < len(all_pts):
    print(f"  After dedup: {len(dedup_pts)}")

  # Full-video signed-gap verification. One pass over the video measures the
  # gap in every static camera; the two filters below read it independently.
  n_before = len(dedup_pts)
  stats = _measure_depth_gaps(dedup_pts, scene_constants, scene_state,
                              static_cams, pb_renderer, tau=tau,
                              safe_margin=safe_margin)
  gone = _filter_support_left(stats, min_run_frames=min_run_frames)
  jitters = (_filter_visibility_flicker(stats, flicker=flicker)
             if flicker is not None else np.zeros(n_before, dtype=bool))
  keep = ~(gone | jitters)
  dedup_pts, dedup_rgb = dedup_pts[keep], dedup_rgb[keep]
  print(f"  Static verification ({stats['n_frames']} frames, tau "
        f"{tau * 1000:.0f}mm): {n_before} -> {len(dedup_pts)} points")
  print(f"    support left ({min_run_frames}+ consecutive frames past the "
        f"surface): {int(gone.sum())}")
  if flicker is None:
    print("    visibility flicker: disabled")
  else:
    print(f"    visibility flicker (>{flicker:.0%} of frames): "
          f"{int(jitters.sum())} ({int((jitters & ~gone).sum())} for this "
          f"reason alone)")

  # Subsample to target number of points
  if num_points is not None and len(dedup_pts) > num_points:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(dedup_pts), num_points, replace=False)
    dedup_pts = dedup_pts[idx]
    dedup_rgb = dedup_rgb[idx]
    print(f"  Subsampled to {num_points} points")

  print(f"  Found {len(dedup_pts)} static background points")
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


def _measure_depth_gaps(pts, scene_constants, scene_state, static_cams,
                        pb_renderer, tau=0.015, safe_margin=15, patch=5):
  """Measure the signed depth gap for every candidate point in every static camera.

  Project a point into a camera and compare its own depth `z` against the depth
  map at that pixel, `d`:

      gap = d - z
        gap ~ 0    the point is on the surface the camera sees   consistent
        gap < 0    something nearer is in the way                occluded
        gap > 0    the camera sees PAST the point                its support is gone

  A present, unoccluded surface point *is* the surface the camera reports, so a
  positive gap cannot happen for a correct point; occlusion only ever pushes the
  gap negative. That asymmetry is what the two filters below are built on, and it
  is why the sign is kept instead of an absolute value being taken.

  The depth is read as a median over a `patch` x `patch` window so that one bad
  pixel cannot decide anything, and pixels covered by the (dilated) robot mask are
  treated as having no reading at all — they carry the arm's depth, not the
  scene's.

  The full (S, T, N) gap array is never materialised: only the running per-camera
  statistics the filters need are carried from frame to frame.

  Args:
    pts: (N, 3) candidate world coordinates.
    static_cams: Camera ids to measure in (the wrist camera moves, so its
        extrinsics carry FK error and it is deliberately excluded).
    tau: Metres. What counts as "on the surface", and how far past the surface a
        camera must see before it is believed. Note that the gap a vanished
        support opens up is the object's local *thickness*, so tau is set by the
        thinnest object worth catching, not by the noise floor: a teapot lid gives
        ~60mm at the knob and ~20mm across the dome. Below ~10mm stereo noise
        starts coming through.
    patch: Side of the median window used to read the depth map.

  Returns:
    Dict of (S, N) arrays, S = len(static_cams), plus n_frames:
      streak:  longest run of CONSECUTIVE frames on which the camera saw past it
      onquery: the camera saw it on the surface on the query frame
      flips:   how many times the camera's clear-line-of-sight flag changed
      seen:    frames on which the camera had a usable depth reading at all
  """
  N = len(pts)
  S = len(static_cams)
  T_frames = len(scene_constants["camera"][static_cams[0]]["video_rgb"])

  streak = np.zeros((S, N), dtype=np.int32)
  run = np.zeros((S, N), dtype=np.int32)
  onquery = np.zeros((S, N), dtype=bool)
  flips = np.zeros((S, N), dtype=np.int32)
  seen = np.zeros((S, N), dtype=np.int32)
  prev_vis = np.zeros((S, N), dtype=bool)

  kernel = np.ones((safe_margin, safe_margin), np.uint8)
  rad = patch // 2
  dy, dx = np.mgrid[-rad:rad + 1, -rad:rad + 1].reshape(2, -1)
  min_valid = max(1, (patch * patch) // 6)      # enough of the window to trust it

  for t in range(T_frames):
    pb_renderer.update_robot_pose(
        scene_constants["robot"]["joint_positions"][t],
        gripper_state=scene_constants["robot"]["gripper_positions"][t])

    for s, cam_id in enumerate(static_cams):
      cam_data = scene_constants["camera"][cam_id]
      ext = scene_state[cam_id]["extrinsics"][t]
      K = cam_data["K_mat"]
      h_img, w_img = cam_data["video_rgb"][0].shape[:2]

      robot_mask = pb_renderer.render_mask(ext, K, w_img, h_img)
      robot_mask_dilated = cv2.dilate(
          robot_mask.astype(np.uint8), kernel, iterations=1) > 0

      u, v, z_pred = project_points(pts, K, ext)
      ok = np.isfinite(u) & np.isfinite(v) & (z_pred > 0)
      ui = np.round(np.where(ok, u, 0)).astype(int)   # round BEFORE bound-checking
      vi = np.round(np.where(ok, v, 0)).astype(int)
      # the whole median window has to be inside the image
      ok &= ((ui >= rad) & (ui < w_img - rad) &
             (vi >= rad) & (vi < h_img - rad))
      ok &= ~robot_mask_dilated[np.clip(vi, 0, h_img - 1),
                                np.clip(ui, 0, w_img - 1)]

      gap = np.full(N, np.nan, dtype=np.float32)
      if ok.any():
        idx = np.flatnonzero(ok)
        window = cam_data["raw_depth"][t][vi[idx, None] + dy, ui[idx, None] + dx]
        window = np.where(window > 0.05, window, np.nan)
        good = np.isfinite(window).sum(axis=1) >= min_valid
        surface = np.full(len(idx), np.nan, dtype=np.float32)
        if good.any():
          surface[good] = np.nanmedian(window[good], axis=1)
        gap[idx] = surface - z_pred[idx]

      measurable = np.isfinite(gap)
      # NaN comparisons are False, so a frame with no reading breaks the run —
      # which is what we want: a run only counts while the camera keeps looking.
      run[s] = np.where(measurable & (gap > tau), run[s] + 1, 0)
      streak[s] = np.maximum(streak[s], run[s])
      vis = measurable & (gap >= -tau)          # this camera has a clear line to it
      if t == 0:
        onquery[s] = measurable & (np.abs(gap) <= tau)
      else:
        flips[s] += vis != prev_vis[s]
      prev_vis[s] = vis
      seen[s] += measurable

  return dict(streak=streak, onquery=onquery, flips=flips, seen=seen,
              n_frames=T_frames)


def _filter_support_left(stats, min_run_frames=30):
  """Its support left: the point is still where it was, the thing under it is not.

  A camera fires if it saw the point on the surface on the query frame and then
  saw past it for `min_run_frames` **consecutive** frames. Consecutive is what
  separates a support that left from a pixel that is merely unreliable: a point
  straddling a depth edge crosses the threshold and comes back all episode, while
  a carried-away support opens a gap that stays open.

  Any one camera firing is enough — once an object is gone only some viewpoints
  have a clear line to the space it vacated, and a camera looking along the
  surface sees almost no gap at all. For the same reason the query-frame test is
  per camera and must stay that way: pooling it lets a camera that never had a
  clear view of the point vouch for one that did.

  Returns:
    (N,) bool, True = remove.
  """
  return ((stats["streak"] >= min_run_frames) & stats["onquery"]).any(axis=0)


def _filter_visibility_flicker(stats, flicker=0.10):
  """Its visibility will not settle, so its ground truth is not worth trusting.

  A static point seen by a fixed camera should go in and out of view a handful of
  times as the arm sweeps past. One whose clear-line-of-sight flag flips on more
  than `flicker` of the frames is not sitting on anything stable.

  Caveat worth knowing before you turn this on: measured over eight exported
  episodes, 51-96% of these flips have the "occluder" only 15-30mm nearer than the
  point — the depth reading wobbling across the +-tau band rather than anything
  actually passing in front. It behaves like a per-pixel noise test whose
  threshold is coupled to tau, and it removes roughly ten times as many points as
  `_filter_support_left`. Pass flicker=None in phase 1 to leave it off.

  Returns:
    (N,) bool, True = remove.
  """
  return (stats["flips"] / max(stats["n_frames"] - 1, 1) > flicker).any(axis=0)

# Phase 2: Project static 3D points to all views (static prior)

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

  print(f"\nPhase 2: Project {N} Static Points -> {len(camera_ids)} Views x {T_frames} Frames")

  per_cam_tracks = {cam: np.zeros((T_frames, N, 2), dtype=np.float32) for cam in camera_ids}
  per_cam_vis = {cam: np.zeros((T_frames, N), dtype=bool) for cam in camera_ids}
  kernel = np.ones((safe_margin, safe_margin), np.uint8)

  # Pre-render dilated robot masks once per frame across all cameras
  robot_masks_dilated = {cam: [] for cam in camera_ids}
  for t in range(T_frames):
    pb_renderer.update_robot_pose(
        scene_constants["robot"]["joint_positions"][t],
        gripper_state=scene_constants["robot"]["gripper_positions"][t])
    for cam_id in camera_ids:
      cam_data = scene_constants["camera"][cam_id]
      ext = scene_state[cam_id]["extrinsics"][t]
      K = cam_data["K_mat"]
      h_img, w_img = cam_data["video_rgb"][0].shape[:2]
      rmask = pb_renderer.render_mask(ext, K, w_img, h_img)
      rmask_dil = cv2.dilate(rmask.astype(np.uint8), kernel, iterations=1) > 0
      robot_masks_dilated[cam_id].append(rmask_dil)

  for cam_id in camera_ids:
    cam_data = scene_constants["camera"][cam_id]
    K = cam_data["K_mat"]
    h_img, w_img = cam_data["video_rgb"][0].shape[:2]

    tracks = np.zeros((T_frames, N, 2), dtype=np.float32)
    vis = np.zeros((T_frames, N), dtype=bool)

    for t in range(T_frames):
      ext = scene_state[cam_id]["extrinsics"][t]

      # Project 3D -> 2D
      u, v, z_pred = project_points(static_pts_3d, K, ext)
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
      not_robot = ~robot_masks_dilated[cam_id][t][vi, ui]

      vis[t] = in_bounds & depth_ok & not_robot

    per_cam_tracks[cam_id] = tracks
    per_cam_vis[cam_id] = vis

    vis_rate = vis.mean() * 100
    print(f"  [{cam_id}] avg visibility: {vis_rate:.1f}%")

  return per_cam_tracks, per_cam_vis

# Phase 3: Robot tracks via URDF FK (dense at t=0)

def phase3_robot_tracks(scene_constants, scene_state, pb_renderer,
                        max_robot_pts_per_cam=None):
  """Extract robot surface tracks via URDF forward kinematics.

  Uses ALL pixels at frame 0 to find robot surface points (dense sampling),
  subsamples to max_robot_pts_per_cam per source camera, then propagates
  via FK across all frames.

  Args:
    scene_constants: Scene data dict.
    scene_state: Extrinsics dict.
    pb_renderer: PyBulletRenderer instance.
    max_robot_pts_per_cam: Max robot points per source camera (None=no limit).

  Returns:
    robot_traj_3d:        (T, N_robot, 3)
    robot_per_cam_tracks: {cam: (T, N_robot, 2)}
    robot_per_cam_vis:    {cam: (T, N_robot)}
    n_robot:              int
  """
  camera_ids = list(scene_constants["camera"].keys())
  T_frames = len(scene_constants["camera"][camera_ids[0]]["video_rgb"])

  print(f"\nPhase 3: URDF FK Robot Tracking (max {max_robot_pts_per_cam} pts/cam)")

  urdf_tracker = URDFKinematicsTracker(pb_renderer)
  robot_traj_3d_all = []
  robot_per_cam_tracks_all = {cam: [] for cam in camera_ids}
  robot_per_cam_vis_all = {cam: [] for cam in camera_ids}

  for src_cam in camera_ids:
    traj_3d_rob, traj_2d_rob, vis_rob, robot_indices = \
        urdf_tracker.extract_robot_tracks(
            src_cam, scene_constants, scene_state,
            max_robot_pts=max_robot_pts_per_cam)

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
    print(f"  Total robot points: {n_robot}")
  else:
    robot_traj_3d = np.zeros((T_frames, 0, 3), dtype=np.float32)
    robot_per_cam_tracks = {
        cam: np.zeros((T_frames, 0, 2), dtype=np.float32)
        for cam in camera_ids}
    robot_per_cam_vis = {
        cam: np.zeros((T_frames, 0), dtype=bool)
        for cam in camera_ids}
    n_robot = 0
    print("  [WARN] No robot points extracted.")

  return robot_traj_3d, robot_per_cam_tracks, robot_per_cam_vis, n_robot

# Phase 4: Merge static + robot tracks

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

  print("\nPhase 4: Merging Static Background + Robot Tracks")
  print(f"  Static: {n_static} | Robot: {n_robot} | Total: {n_static + n_robot}")

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

# Export (same format as compute_tracks.py)

def export_tracks(scene_constants, scene_state, final_traj_3d,
                  final_vis_global, final_per_cam_tracks, final_per_cam_vis,
                  n_static, n_robot,
                  export_root=os.path.join(OUTPUT_ROOT, "tracks")):
  """Serialize tracking results to disk.

  Returns:
    ep_dir: Absolute path to the created episode output directory.
  """
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

  print(f"  Exported {N} tracks × {T} frames to {ep_dir}")
  return ep_dir

# Full Pipeline

def process_episode(episode_id, pb_renderer, device,
                    depth_root, extrinsics_root, export_root,
                    num_static_points=300,
                    max_robot_pts_per_cam=100):
  """Full static+robot pipeline for a single episode.

  No tracker model needed — background points use static prior only.
  Dense pixel sampling at frame 0 for both robot and background.

  Args:
    num_static_points: Target number of static background points (subsampled
        from cross-view consensus candidates). Matches pipeline.ipynb default.
    max_robot_pts_per_cam: Max robot surface points per source camera.
        Matches pipeline.ipynb default.
  """
  print(f"\nProcessing Episode: {episode_id}")

  # Load data (full video, no truncation)
  scene_constants = load_depth_data(episode_id, depth_root, load_video="full")
  scene_state = load_extrinsics(scene_constants, extrinsics_root)

  camera_ids = list(scene_constants["camera"].keys())
  T_frames = len(scene_constants["camera"][camera_ids[0]]["video_rgb"])
  print(f"  {len(camera_ids)} cameras × {T_frames} frames (full video)")

  # Phase 1: Find static background points (t=0 only, dense)
  print(f"\nPhase 1: Dense at t=0 -> Cross-View Consensus -> Static Points")
  static_pts_3d, static_rgb = phase1_find_static_candidates(
      scene_constants, scene_state, pb_renderer,
      num_points=num_static_points)

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

  # Phase 3: Robot tracks (dense at t=0)
  robot_traj_3d, robot_per_cam_tracks, robot_per_cam_vis, n_robot = \
      phase3_robot_tracks(scene_constants, scene_state, pb_renderer,
                          max_robot_pts_per_cam=max_robot_pts_per_cam)

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

  print(f"\n  Episode {episode_id}: {n_static} static + {n_robot} robot "
        f"= {n_static + n_robot} tracks exported.")
  return n_static + n_robot


# Standalone CLI


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="DROID Stage 3 v2: Static Background + Robot Tracks")
  parser.add_argument("--rank", type=int, default=0,
                      help="Rank of the process (for multi-GPU sharding)")
  parser.add_argument("--world_size", type=int, default=1,
                      help="Total number of processes")
  parser.add_argument("--limit", type=int, default=-1,
                      help="Limit total number of episodes to process")
  parser.add_argument("--depth_root", type=str,
                      default=os.path.join(OUTPUT_ROOT, "depth"))
  parser.add_argument("--extrinsics_root", type=str,
                      default=os.path.join(OUTPUT_ROOT, "extrinsics"))
  parser.add_argument("--export_root", type=str,
                      default=os.path.join(OUTPUT_ROOT, "tracks"))
  parser.add_argument("--num_static_points", type=int, default=300,
                      help="Target number of static background points")
  parser.add_argument("--max_robot_pts_per_cam", type=int, default=100,
                      help="Max robot surface points per source camera")
  args = parser.parse_args()

  print("DROID Stage 3 v2: Static Background + Robot Tracks")
  device = get_accelerator()
  pb_renderer = PyBulletRenderer()

  # Discover available episodes from extrinsics output
  ext_abs = os.path.abspath(os.path.expanduser(args.extrinsics_root))
  export_abs = os.path.abspath(os.path.expanduser(args.export_root))
  available_eps = sorted([
      d for d in os.listdir(ext_abs)
      if os.path.isdir(os.path.join(ext_abs, d))
  ])

  # Deterministic shuffle for load balancing across ranks
  random.seed(42)
  random.shuffle(available_eps)

  if args.limit > 0:
    available_eps = available_eps[:args.limit]

  # Shard across ranks
  target_eps = available_eps[args.rank::args.world_size]

  # Skip episodes that already have output (resume-friendly)
  todo_eps = []
  for ep_id in target_eps:
    ep_out = os.path.join(export_abs, ep_id, "tracks_3d.npz")
    if os.path.exists(ep_out):
      continue
    todo_eps.append(ep_id)

  print(f"Rank {args.rank}/{args.world_size}: "
        f"{len(todo_eps)} episodes to process "
        f"({len(target_eps) - len(todo_eps)} already done)")

  succeeded_eps = []

  for idx, ep_id in enumerate(todo_eps):
    print(f"\n[{idx + 1}/{len(todo_eps)}] Episode: {ep_id}")
    try:
      process_episode(
          ep_id, pb_renderer, device,
          args.depth_root, args.extrinsics_root, args.export_root,
          num_static_points=args.num_static_points,
          max_robot_pts_per_cam=args.max_robot_pts_per_cam)
      succeeded_eps.append(ep_id)
    except Exception as e:
      print(f"  [FAIL] Episode {ep_id} failed: {e}")
      traceback.print_exc()

  print(f"\nStage 3 v2 complete! "
        f"{len(succeeded_eps)}/{len(todo_eps)} episodes succeeded.")

