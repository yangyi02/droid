import argparse
import os

import cv2
import numpy as np

import core.geometry
import core.io
import core.physics
import core.runner
import core.tracking


def find_static_candidates(
  scene_constants,
  scene_state,
  pb_renderer,
  match_radius=0.005,
  num_points=None,
  safe_margin=15,
  tau=0.015,
  min_run_frames=30,
  flicker=0.10,
):
  camera_ids = list(scene_constants["camera"].keys())
  wrist_serial = scene_constants["meta"].get("wrist_serial")

  static_cams = [c for c in camera_ids if c != wrist_serial]

  pb_renderer.update_robot_pose(
    scene_constants["robot"]["joint_positions"][0],
    gripper_state=scene_constants["robot"]["gripper_positions"][0],
  )

  per_cam_pts = {}
  per_cam_rgb = {}

  for cam_id in camera_ids:
    cam_data = scene_constants["camera"][cam_id]
    ext = scene_state[cam_id]["extrinsics"][0]
    K = cam_data["K_mat"]
    h_img, w_img = cam_data["video_rgb"][0].shape[:2]

    robot_mask = pb_renderer.render_mask(ext, K, w_img, h_img)
    kernel = np.ones((safe_margin, safe_margin), np.uint8)
    robot_mask_dilated = cv2.dilate(robot_mask.astype(np.uint8), kernel, iterations=1) > 0

    depth = cam_data["raw_depth"][0]
    is_env = ~robot_mask_dilated
    has_depth = (depth > 0.05) & (depth < 5.0)
    valid_mask = is_env & has_depth
    vs, us = np.where(valid_mask)

    if len(us) == 0:
      continue

    z = depth[vs, us]
    u_f = us.astype(np.float32)
    v_f = vs.astype(np.float32)

    pts_3d = core.geometry.unproject_points(u_f, v_f, z, K, ext)

    rgb = cam_data["video_rgb"][0][vs, us]

    per_cam_pts[cam_id] = pts_3d
    per_cam_rgb[cam_id] = rgb

  all_verified_pts = []
  all_verified_rgb = []

  for src_cam in camera_ids:
    pts = per_cam_pts.get(src_cam)
    rgb = per_cam_rgb.get(src_cam)
    if pts is None or len(pts) == 0:
      continue

    n_agree = np.zeros(len(pts), dtype=int)

    for dst_cam in camera_ids:
      if dst_cam == src_cam:
        continue
      dst_data = scene_constants["camera"][dst_cam]
      dst_ext = scene_state[dst_cam]["extrinsics"][0]
      dst_K = dst_data["K_mat"]
      dst_h, dst_w = dst_data["video_rgb"][0].shape[:2]

      u_d, v_d, z_pred = core.geometry.project_points(pts, dst_K, dst_ext)
      ui_d = np.clip(np.round(u_d).astype(int), 0, dst_w - 1)
      vi_d = np.clip(np.round(v_d).astype(int), 0, dst_h - 1)

      in_bounds = (u_d >= 0) & (u_d < dst_w) & (v_d >= 0) & (v_d < dst_h) & (z_pred > 0)

      z_obs = dst_data["raw_depth"][0, vi_d, ui_d]
      depth_ok = (z_obs > 0.05) & (np.abs(z_pred - z_obs) < match_radius)

      n_agree += (in_bounds & depth_ok).astype(int)

    verified = n_agree >= 1
    n_verified = np.sum(verified)

    if n_verified > 0:
      all_verified_pts.append(pts[verified])
      all_verified_rgb.append(rgb[verified])

  all_pts = np.concatenate(all_verified_pts, axis=0)
  all_rgb = np.concatenate(all_verified_rgb, axis=0)
  print(f"  Total verified points (pre-dedup): {len(all_pts)}")

  dedup_pts, dedup_rgb = _voxel_dedup(all_pts, all_rgb, voxel_size=match_radius * 2)
  print(f"  After dedup: {len(dedup_pts)}")

  n_before = len(dedup_pts)
  stats = _measure_depth_gaps(
    dedup_pts,
    scene_constants,
    scene_state,
    static_cams,
    pb_renderer,
    tau=tau,
    safe_margin=safe_margin,
  )
  gone = _filter_support_left(stats, min_run_frames=min_run_frames)
  jitters = (
    _filter_visibility_flicker(stats, flicker=flicker)
    if flicker is not None
    else np.zeros(n_before, dtype=bool)
  )
  keep = ~(gone | jitters)
  dedup_pts, dedup_rgb = dedup_pts[keep], dedup_rgb[keep]

  if num_points is not None and len(dedup_pts) > num_points:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(dedup_pts), num_points, replace=False)
    dedup_pts = dedup_pts[idx]
    dedup_rgb = dedup_rgb[idx]

  return dedup_pts, dedup_rgb


def _voxel_dedup(pts, rgb, voxel_size=0.01):
  if len(pts) == 0:
    return pts, rgb

  voxel_indices = np.floor(pts / voxel_size).astype(np.int64)
  keys = (
    voxel_indices[:, 0].astype(np.int64) * 1000000
    + voxel_indices[:, 1].astype(np.int64) * 1000
    + voxel_indices[:, 2].astype(np.int64)
  )

  unique_keys, inverse = np.unique(keys, return_inverse=True)
  N_unique = len(unique_keys)

  out_pts = np.zeros((N_unique, 3), dtype=np.float32)
  out_rgb = np.zeros((N_unique, 3), dtype=np.uint8)

  for i in range(N_unique):
    mask = inverse == i
    out_pts[i] = np.median(pts[mask], axis=0)
    out_rgb[i] = np.median(rgb[mask].astype(np.float32), axis=0).astype(np.uint8)

  return out_pts, out_rgb


def _measure_depth_gaps(
  pts, scene_constants, scene_state, static_cams, pb_renderer, tau=0.015, safe_margin=15, patch=5
):
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
  dy, dx = np.mgrid[-rad : rad + 1, -rad : rad + 1].reshape(2, -1)
  min_valid = max(1, (patch * patch) // 6)

  for t in range(T_frames):
    pb_renderer.update_robot_pose(
      scene_constants["robot"]["joint_positions"][t],
      gripper_state=scene_constants["robot"]["gripper_positions"][t],
    )

    for s, cam_id in enumerate(static_cams):
      cam_data = scene_constants["camera"][cam_id]
      ext = scene_state[cam_id]["extrinsics"][t]
      K = cam_data["K_mat"]
      h_img, w_img = cam_data["video_rgb"][0].shape[:2]

      robot_mask = pb_renderer.render_mask(ext, K, w_img, h_img)
      robot_mask_dilated = cv2.dilate(robot_mask.astype(np.uint8), kernel, iterations=1) > 0

      u, v, z_pred = core.geometry.project_points(pts, K, ext)
      ok = np.isfinite(u) & np.isfinite(v) & (z_pred > 0)
      ui = np.round(np.where(ok, u, 0)).astype(int)
      vi = np.round(np.where(ok, v, 0)).astype(int)
      ok &= (ui >= rad) & (ui < w_img - rad) & (vi >= rad) & (vi < h_img - rad)
      ok &= ~robot_mask_dilated[np.clip(vi, 0, h_img - 1), np.clip(ui, 0, w_img - 1)]

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
      run[s] = np.where(measurable & (gap > tau), run[s] + 1, 0)
      streak[s] = np.maximum(streak[s], run[s])
      vis = measurable & (gap >= -tau)
      if t == 0:
        onquery[s] = measurable & (np.abs(gap) <= tau)
      else:
        flips[s] += vis != prev_vis[s]
      prev_vis[s] = vis
      seen[s] += measurable

  return dict(streak=streak, onquery=onquery, flips=flips, seen=seen, n_frames=T_frames)


def _filter_support_left(stats, min_run_frames=30):
  return ((stats["streak"] >= min_run_frames) & stats["onquery"]).any(axis=0)


def _filter_visibility_flicker(stats, flicker=0.10):
  return (stats["flips"] / max(stats["n_frames"] - 1, 1) > flicker).any(axis=0)


def project_static_tracks(
  static_pts_3d, scene_constants, scene_state, pb_renderer, depth_tolerance=0.05, safe_margin=15
):
  camera_ids = list(scene_constants["camera"].keys())
  T_frames = len(scene_constants["camera"][camera_ids[0]]["video_rgb"])
  N = len(static_pts_3d)

  per_cam_tracks = {cam: np.zeros((T_frames, N, 2), dtype=np.float32) for cam in camera_ids}
  per_cam_vis = {cam: np.zeros((T_frames, N), dtype=bool) for cam in camera_ids}
  kernel = np.ones((safe_margin, safe_margin), np.uint8)

  robot_masks_dilated = {cam: [] for cam in camera_ids}
  for t in range(T_frames):
    pb_renderer.update_robot_pose(
      scene_constants["robot"]["joint_positions"][t],
      gripper_state=scene_constants["robot"]["gripper_positions"][t],
    )
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

      u, v, z_pred = core.geometry.project_points(static_pts_3d, K, ext)
      tracks[t, :, 0] = u
      tracks[t, :, 1] = v

      in_bounds = (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img) & (z_pred > 0)

      ui = np.clip(np.round(u).astype(int), 0, w_img - 1)
      vi = np.clip(np.round(v).astype(int), 0, h_img - 1)
      z_obs = cam_data["raw_depth"][t, vi, ui]
      depth_ok = (z_obs > 0.05) & (np.abs(z_pred - z_obs) < depth_tolerance)

      not_robot = ~robot_masks_dilated[cam_id][t][vi, ui]

      vis[t] = in_bounds & depth_ok & not_robot

    per_cam_tracks[cam_id] = tracks
    per_cam_vis[cam_id] = vis
    print(f"  [{cam_id}] avg visibility: {vis.mean() * 100:.1f}%")

  return per_cam_tracks, per_cam_vis


def compute_robot_tracks(scene_constants, scene_state, pb_renderer, max_robot_pts_per_cam=None):
  camera_ids = list(scene_constants["camera"].keys())
  T_frames = len(scene_constants["camera"][camera_ids[0]]["video_rgb"])

  urdf_tracker = core.tracking.URDFKinematicsTracker(pb_renderer)
  robot_traj_3d_all = []
  robot_per_cam_tracks_all = {cam: [] for cam in camera_ids}
  robot_per_cam_vis_all = {cam: [] for cam in camera_ids}

  for src_cam in camera_ids:
    traj_3d_rob, traj_2d_rob, vis_rob, robot_indices = urdf_tracker.extract_robot_tracks(
      src_cam, scene_constants, scene_state, max_robot_pts=max_robot_pts_per_cam
    )

    if traj_3d_rob is None or len(robot_indices) == 0:
      continue

    rob_per_cam_2d, rob_per_cam_vis = urdf_tracker.project_to_all_views(
      traj_3d_rob, scene_constants, scene_state
    )

    rob_per_cam_2d[src_cam] = traj_2d_rob
    rob_per_cam_vis[src_cam] = vis_rob

    robot_traj_3d_all.append(traj_3d_rob)
    for cam in camera_ids:
      robot_per_cam_tracks_all[cam].append(rob_per_cam_2d[cam])
      robot_per_cam_vis_all[cam].append(rob_per_cam_vis[cam])

  if robot_traj_3d_all:
    robot_traj_3d = np.concatenate(robot_traj_3d_all, axis=1)
    robot_per_cam_tracks = {
      cam: np.concatenate(robot_per_cam_tracks_all[cam], axis=1) for cam in camera_ids
    }
    robot_per_cam_vis = {
      cam: np.concatenate(robot_per_cam_vis_all[cam], axis=1) for cam in camera_ids
    }
    n_robot = robot_traj_3d.shape[1]
  else:
    robot_traj_3d = np.zeros((T_frames, 0, 3), dtype=np.float32)
    robot_per_cam_tracks = {cam: np.zeros((T_frames, 0, 2), dtype=np.float32) for cam in camera_ids}
    robot_per_cam_vis = {cam: np.zeros((T_frames, 0), dtype=bool) for cam in camera_ids}
    n_robot = 0

  return robot_traj_3d, robot_per_cam_tracks, robot_per_cam_vis, n_robot


def merge_tracks(
  static_pts_3d,
  static_per_cam_tracks,
  static_per_cam_vis,
  robot_traj_3d,
  robot_per_cam_tracks,
  robot_per_cam_vis,
  camera_ids,
  T_frames,
):
  n_static = len(static_pts_3d)
  n_robot = robot_traj_3d.shape[1]
  print(f"  Static: {n_static} | Robot: {n_robot} | Total: {n_static + n_robot}")

  if n_static > 0:
    static_traj_3d = np.broadcast_to(static_pts_3d[None, :, :], (T_frames, n_static, 3)).copy()
  else:
    static_traj_3d = np.zeros((T_frames, 0, 3), dtype=np.float32)

  if n_static > 0:
    static_vis_global = np.zeros((T_frames, n_static), dtype=bool)
    for cam in camera_ids:
      static_vis_global |= static_per_cam_vis[cam]
  else:
    static_vis_global = np.zeros((T_frames, 0), dtype=bool)

  if n_robot > 0:
    robot_vis_global = np.zeros((T_frames, n_robot), dtype=bool)
    for cam in camera_ids:
      robot_vis_global |= robot_per_cam_vis[cam]
  else:
    robot_vis_global = np.zeros((T_frames, 0), dtype=bool)

  final_traj_3d = np.concatenate([static_traj_3d, robot_traj_3d], axis=1)
  final_vis_global = np.concatenate([static_vis_global, robot_vis_global], axis=1)
  final_per_cam_tracks = {
    cam: np.concatenate([static_per_cam_tracks[cam], robot_per_cam_tracks[cam]], axis=1)
    for cam in camera_ids
  }
  final_per_cam_vis = {
    cam: np.concatenate([static_per_cam_vis[cam], robot_per_cam_vis[cam]], axis=1)
    for cam in camera_ids
  }

  return (
    final_traj_3d,
    final_vis_global,
    final_per_cam_tracks,
    final_per_cam_vis,
    n_static,
    n_robot,
  )


def export_tracks(
  scene_constants,
  scene_state,
  final_traj_3d,
  final_vis_global,
  final_per_cam_tracks,
  final_per_cam_vis,
  n_static,
  n_robot,
  export_root=os.path.join(core.io.OUTPUT_ROOT, "tracks"),
):
  ep_id = scene_constants["meta"]["episode_id"]
  camera_ids = list(scene_constants["camera"].keys())
  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(export_root, ep_id)))
  os.makedirs(ep_dir, exist_ok=True)

  np.savez_compressed(
    os.path.join(ep_dir, "tracks_3d.npz"),
    traj_3d=final_traj_3d.astype(np.float32),
    vis_global=final_vis_global,
  )

  for cam_id in camera_ids:
    cam_dir = os.path.join(ep_dir, cam_id)
    os.makedirs(cam_dir, exist_ok=True)

    traj_2d = final_per_cam_tracks[cam_id].copy()
    vis = final_per_cam_vis[cam_id].copy()
    traj_2d[~vis] = -1000.0

    np.savez_compressed(
      os.path.join(cam_dir, "tracks_2d.npz"), traj_2d=traj_2d.astype(np.float32), vis_2d=vis
    )

    cam_data = scene_constants["camera"][cam_id]
    K = cam_data["K_mat"]
    np.save(
      os.path.join(cam_dir, "intrinsics.npy"),
      np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32),
    )

    np.save(
      os.path.join(cam_dir, "extrinsics_w2c.npy"),
      np.linalg.inv(scene_state[cam_id]["extrinsics"]).astype(np.float32),
    )

  np.savez_compressed(
    os.path.join(ep_dir, "track_metadata.npz"),
    n_static=np.array(n_static),
    n_robot=np.array(n_robot),
    point_type=np.array([0] * n_static + [1] * n_robot, dtype=np.uint8),
  )

  return ep_dir


def process_episode(
  episode_id,
  pb_renderer,
  device,
  depth_root,
  extrinsics_root,
  export_root,
  num_static_points=300,
  max_robot_pts_per_cam=100,
):

  scene_constants = core.io.load_depth_data(episode_id, depth_root, load_video="full")
  scene_state = core.io.load_extrinsics(scene_constants, extrinsics_root)

  camera_ids = list(scene_constants["camera"].keys())
  T_frames = len(scene_constants["camera"][camera_ids[0]]["video_rgb"])

  static_pts_3d, _ = find_static_candidates(
    scene_constants, scene_state, pb_renderer, num_points=num_static_points
  )

  if len(static_pts_3d) > 0:
    static_per_cam_tracks, static_per_cam_vis = project_static_tracks(
      static_pts_3d, scene_constants, scene_state, pb_renderer
    )
  else:
    static_per_cam_tracks = {
      cam: np.zeros((T_frames, 0, 2), dtype=np.float32) for cam in camera_ids
    }
    static_per_cam_vis = {cam: np.zeros((T_frames, 0), dtype=bool) for cam in camera_ids}

  robot_traj_3d, robot_per_cam_tracks, robot_per_cam_vis, n_robot = compute_robot_tracks(
    scene_constants, scene_state, pb_renderer, max_robot_pts_per_cam=max_robot_pts_per_cam
  )

  (final_traj_3d, final_vis_global, final_per_cam_tracks, final_per_cam_vis, n_static, n_robot) = (
    merge_tracks(
      static_pts_3d,
      static_per_cam_tracks,
      static_per_cam_vis,
      robot_traj_3d,
      robot_per_cam_tracks,
      robot_per_cam_vis,
      camera_ids,
      T_frames,
    )
  )

  export_tracks(
    scene_constants,
    scene_state,
    final_traj_3d,
    final_vis_global,
    final_per_cam_tracks,
    final_per_cam_vis,
    n_static,
    n_robot,
    export_root,
  )

  return n_static + n_robot


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="DROID Stage 3: Static Background + Robot Tracks")
  core.runner.add_sharding_args(parser)
  parser.add_argument(
    "--depth_root",
    type=str,
    default=os.path.join(core.io.OUTPUT_ROOT, "depth"),
    help="Root directory of depth outputs",
  )
  parser.add_argument(
    "--extrinsics_root",
    type=str,
    default=os.path.join(core.io.OUTPUT_ROOT, "extrinsics"),
    help="Root directory of extrinsics outputs",
  )
  parser.add_argument(
    "--export_root",
    type=str,
    default=os.path.join(core.io.OUTPUT_ROOT, "tracks"),
    help="Root directory for tracks output",
  )
  parser.add_argument(
    "--num_static_points",
    type=int,
    default=300,
    help="Number of static background points to sample",
  )
  parser.add_argument(
    "--max_robot_pts_per_cam", type=int, default=100, help="Max robot surface points per camera"
  )
  args = parser.parse_args()

  device = core.io.get_accelerator()
  pb_renderer = core.physics.PyBulletRenderer()

  target = core.runner.shard_episodes(
    core.runner.list_episode_dirs(args.extrinsics_root), args.rank, args.world_size, args.limit
  )
  export_abs = os.path.abspath(os.path.expanduser(args.export_root))
  done = {ep for ep in target if os.path.exists(os.path.join(export_abs, ep, "tracks_3d.npz"))}

  def run_one(episode_id):
    process_episode(
      episode_id,
      pb_renderer,
      device,
      args.depth_root,
      args.extrinsics_root,
      args.export_root,
      num_static_points=args.num_static_points,
      max_robot_pts_per_cam=args.max_robot_pts_per_cam,
    )

  core.runner.run_episodes(
    target, run_one, rank=args.rank, world_size=args.world_size, done=done, stage="Stage 3"
  )
