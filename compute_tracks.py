import os

import cv2
import numpy as np
from absl import app
from ml_collections import config_flags

import config
import core.geometry
import core.io
import core.physics
import core.runner
import core.tracking


def render_robot_masks(scene_constants, scene_state, pb_renderer, safe_margin=15):
  camera_ids = list(scene_constants["camera"].keys())
  kernel = np.ones((safe_margin, safe_margin), np.uint8)
  masks = {cam: [] for cam in camera_ids}

  for t in range(len(scene_constants["camera"][camera_ids[0]]["video_rgb"])):
    pb_renderer.update_robot_pose(
      scene_constants["robot"]["joint_positions"][t],
      gripper_state=scene_constants["robot"]["gripper_positions"][t],
    )
    for cam_id in camera_ids:
      cam_data = scene_constants["camera"][cam_id]
      h_img, w_img = cam_data["video_rgb"][0].shape[:2]
      raw = pb_renderer.render_mask(
        scene_state[cam_id]["extrinsics"][t], cam_data["K_mat"], w_img, h_img
      )
      masks[cam_id].append(cv2.dilate(raw.astype(np.uint8), kernel, iterations=1) > 0)

  return {cam: np.array(m) for cam, m in masks.items()}


def find_static_candidates(
  scene_constants,
  scene_state,
  robot_masks,
  match_radius=0.005,
  num_points=None,
  tau=0.015,
  min_run_frames=30,
  flicker=0.10,
):
  camera_ids = list(scene_constants["camera"].keys())
  wrist_serial = scene_constants["meta"].get("wrist_serial")

  static_cams = [c for c in camera_ids if c != wrist_serial]

  per_cam_pts = {}
  per_cam_rgb = {}

  for cam_id in camera_ids:
    cam_data = scene_constants["camera"][cam_id]
    ext = scene_state[cam_id]["extrinsics"][0]
    K = cam_data["K_mat"]
    h_img, w_img = cam_data["video_rgb"][0].shape[:2]
    robot_mask_dilated = robot_masks[cam_id][0]

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

    pts_3d = core.geometry.unproject_pixels(u_f, v_f, z, K, ext)

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

  voxels = np.floor(all_pts / (match_radius * 2)).astype(np.int64)
  _, inverse = np.unique(voxels, axis=0, return_inverse=True)
  order = np.argsort(inverse, kind="stable")
  cuts = np.cumsum(np.bincount(inverse))[:-1]
  dedup_pts = np.array(
    [np.median(v, axis=0) for v in np.split(all_pts[order], cuts)], dtype=np.float32
  )
  dedup_rgb = np.array(
    [np.median(v, axis=0) for v in np.split(all_rgb[order].astype(np.float32), cuts)]
  ).astype(np.uint8)
  print(f"  After dedup: {len(dedup_pts)}")

  streak, onquery, flips, n_frames = _measure_depth_gaps(
    dedup_pts, scene_constants, scene_state, static_cams, robot_masks, tau=tau
  )
  gone = ((streak >= min_run_frames) & onquery).any(axis=0)
  jitters = (
    (flips / max(n_frames - 1, 1) > flicker).any(axis=0)
    if flicker is not None
    else np.zeros(len(dedup_pts), dtype=bool)
  )
  keep = ~(gone | jitters)
  dedup_pts, dedup_rgb = dedup_pts[keep], dedup_rgb[keep]

  if num_points is not None and len(dedup_pts) > num_points:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(dedup_pts), num_points, replace=False)
    dedup_pts = dedup_pts[idx]
    dedup_rgb = dedup_rgb[idx]

  return dedup_pts, dedup_rgb


def _measure_depth_gaps(
  pts, scene_constants, scene_state, static_cams, robot_masks, tau=0.015, patch=5
):
  N = len(pts)
  S = len(static_cams)
  T_frames = len(scene_constants["camera"][static_cams[0]]["video_rgb"])

  streak = np.zeros((S, N), dtype=np.int32)
  run = np.zeros((S, N), dtype=np.int32)
  onquery = np.zeros((S, N), dtype=bool)
  flips = np.zeros((S, N), dtype=np.int32)
  prev_vis = np.zeros((S, N), dtype=bool)

  rad = patch // 2
  dy, dx = np.mgrid[-rad : rad + 1, -rad : rad + 1].reshape(2, -1)
  min_valid = max(1, (patch * patch) // 6)

  for t in range(T_frames):
    for s, cam_id in enumerate(static_cams):
      cam_data = scene_constants["camera"][cam_id]
      ext = scene_state[cam_id]["extrinsics"][t]
      K = cam_data["K_mat"]
      h_img, w_img = cam_data["video_rgb"][0].shape[:2]
      robot_mask_dilated = robot_masks[cam_id][t]

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

  return streak, onquery, flips, T_frames


def project_static_tracks(
  static_pts_3d, scene_constants, scene_state, robot_masks, depth_tolerance=0.05
):
  camera_ids = list(scene_constants["camera"].keys())
  T_frames = len(scene_constants["camera"][camera_ids[0]]["video_rgb"])
  N = len(static_pts_3d)

  per_cam_tracks = {cam: np.zeros((T_frames, N, 2), dtype=np.float32) for cam in camera_ids}
  per_cam_vis = {cam: np.zeros((T_frames, N), dtype=bool) for cam in camera_ids}

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

      not_robot = ~robot_masks[cam_id][t][vi, ui]

      vis[t] = in_bounds & depth_ok & not_robot

    per_cam_tracks[cam_id] = tracks
    per_cam_vis[cam_id] = vis
    print(f"  [{cam_id}] avg visibility: {vis.mean() * 100:.1f}%")

  return per_cam_tracks, per_cam_vis


def compute_robot_tracks(
  scene_constants, scene_state, pb_renderer, max_robot_pts_per_cam=None, safe_margin=7
):
  camera_ids = list(scene_constants["camera"].keys())
  T_frames = len(scene_constants["camera"][camera_ids[0]]["video_rgb"])

  urdf_tracker = core.tracking.URDFKinematicsTracker(pb_renderer)
  robot_traj_3d_all = []
  robot_per_cam_tracks_all = {cam: [] for cam in camera_ids}
  robot_per_cam_vis_all = {cam: [] for cam in camera_ids}

  for src_cam in camera_ids:
    traj_3d_rob, traj_2d_rob, vis_rob, robot_indices = urdf_tracker.extract_robot_tracks(
      src_cam,
      scene_constants,
      scene_state,
      safe_margin=safe_margin,
      max_robot_pts=max_robot_pts_per_cam,
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
  export_root,
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


def process_episode(episode_id, pb_renderer, device, config):
  depth_root, extrinsics_root, export_root = (
    config.paths.depth,
    config.paths.extrinsics,
    config.paths.tracks,
  )

  scene_constants = core.io.load_depth_data(episode_id, depth_root, load_video="full")
  scene_state = core.io.load_extrinsics(scene_constants, extrinsics_root)

  camera_ids = list(scene_constants["camera"].keys())
  T_frames = len(scene_constants["camera"][camera_ids[0]]["video_rgb"])

  robot_masks = render_robot_masks(
    scene_constants, scene_state, pb_renderer, safe_margin=config.tracks.safe_margin
  )

  static_pts_3d, _ = find_static_candidates(
    scene_constants,
    scene_state,
    robot_masks,
    match_radius=config.tracks.match_radius,
    num_points=config.tracks.num_static_points,
    tau=config.tracks.tau,
    min_run_frames=config.tracks.min_run_frames,
    flicker=config.tracks.flicker,
  )

  if len(static_pts_3d) > 0:
    static_per_cam_tracks, static_per_cam_vis = project_static_tracks(
      static_pts_3d,
      scene_constants,
      scene_state,
      robot_masks,
      depth_tolerance=config.tracks.depth_tolerance,
    )
  else:
    static_per_cam_tracks = {
      cam: np.zeros((T_frames, 0, 2), dtype=np.float32) for cam in camera_ids
    }
    static_per_cam_vis = {cam: np.zeros((T_frames, 0), dtype=bool) for cam in camera_ids}

  robot_traj_3d, robot_per_cam_tracks, robot_per_cam_vis, n_robot = compute_robot_tracks(
    scene_constants,
    scene_state,
    pb_renderer,
    max_robot_pts_per_cam=config.tracks.max_robot_pts_per_cam,
    safe_margin=config.tracks.robot_safe_margin,
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


def main(_):
  config = config_flag.value
  device = core.io.get_accelerator()
  pb_renderer = core.physics.PyBulletRenderer(config.paths.urdf, gpu=config.render.gpu)

  target = core.runner.shard_episodes(
    core.runner.list_episode_dirs(config.paths.extrinsics),
    config.runner.rank,
    config.runner.world_size,
    config.runner.limit,
  )
  export_abs = os.path.abspath(os.path.expanduser(config.paths.tracks))
  done = {ep for ep in target if os.path.exists(os.path.join(export_abs, ep, "tracks_3d.npz"))}

  def run_one(episode_id):
    process_episode(episode_id, pb_renderer, device, config)

  core.runner.run_episodes(
    target,
    run_one,
    rank=config.runner.rank,
    world_size=config.runner.world_size,
    done=done,
    stage="Stage 3",
  )


if __name__ == "__main__":
  config_flag = config_flags.DEFINE_config_file("config", config.__file__)
  app.run(main)
