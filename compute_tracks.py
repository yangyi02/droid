import os

import cv2
import numpy as np
import pybullet
from absl import app
from ml_collections import config_flags
from scipy.spatial.transform import Rotation

import config
import core.geometry
import core.io
import core.physics
import core.runner


def sample_depth(depth_img, u, v, z_pred):
  h_img, w_img = depth_img.shape
  ui = np.clip(np.round(u).astype(int), 0, w_img - 1)
  vi = np.clip(np.round(v).astype(int), 0, h_img - 1)
  in_frame = (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img) & (z_pred > 0)
  return np.where(in_frame, depth_img[vi, ui], np.nan)


def depth_gap(cam_data, pts_3d, extrinsic, t):
  u, v, z_pred = core.geometry.project_points(pts_3d, cam_data["K_mat"], extrinsic)
  return u, v, sample_depth(cam_data["raw_depth"][t], u, v, z_pred) - z_pred


def sample_indices(indices, num_points=None, seed=42):
  if num_points is None or len(indices) <= num_points:
    return indices
  return np.sort(np.random.default_rng(seed).choice(indices, num_points, replace=False))


def keep_tracks(per_cam, keep):
  return {cam: arr[:, keep] for cam, arr in per_cam.items()}


def link_transform(obj_id, link_id):
  if link_id == -1:
    pos, orn = pybullet.getBasePositionAndOrientation(obj_id)
  else:
    pos, orn = pybullet.getLinkState(obj_id, link_id)[:2]

  T = np.eye(4)
  T[:3, :3] = Rotation.from_quat(orn).as_matrix()
  T[:3, 3] = pos
  return T


def find_static_candidates(
  scene_constants, scene_state, pb_renderer, safe_margin=15, match_radius=0.005
):
  camera_ids = list(scene_constants["camera"])
  robot = scene_constants["robot"]
  kernel = np.ones((safe_margin, safe_margin), np.uint8)

  pb_renderer.update_robot_pose(
    robot["joint_positions"][0], gripper_state=robot["gripper_positions"][0]
  )

  verified = []
  for src_cam in camera_ids:
    cam_data = scene_constants["camera"][src_cam]
    depth = cam_data["raw_depth"][0]
    h_img, w_img = depth.shape

    robot_mask = pb_renderer.render_mask(
      scene_state[src_cam]["extrinsics"][0], cam_data["K_mat"], w_img, h_img
    )
    near_robot = cv2.dilate(robot_mask.astype(np.uint8), kernel, iterations=1) > 0
    on_env = ~near_robot & (depth > 0.05) & (depth < 5.0)
    vs, us = np.where(on_env)

    pts = core.geometry.unproject_pixels(
      us.astype(np.float32),
      vs.astype(np.float32),
      depth[vs, us],
      cam_data["K_mat"],
      scene_state[src_cam]["extrinsics"][0],
    )

    n_agree = np.zeros(len(pts), dtype=int)
    for dst_cam in camera_ids:
      if dst_cam == src_cam:
        continue
      _, _, gap = depth_gap(
        scene_constants["camera"][dst_cam], pts, scene_state[dst_cam]["extrinsics"][0], 0
      )
      n_agree += np.abs(gap) < match_radius

    verified.append(pts[n_agree >= 1])

  all_pts = np.concatenate(verified, axis=0)
  voxels = np.floor(all_pts / (match_radius * 2)).astype(np.int64)
  _, inverse = np.unique(voxels, axis=0, return_inverse=True)
  order = np.argsort(inverse, kind="stable")
  cuts = np.cumsum(np.bincount(inverse))[:-1]

  return np.array([np.median(g, axis=0) for g in np.split(all_pts[order], cuts)], dtype=np.float32)


def project_static_tracks(static_pts_3d, scene_constants, scene_state, depth_tolerance=0.05):
  per_cam_tracks = {}
  per_cam_vis = {}
  per_cam_gap = {}

  for cam_id, cam_data in scene_constants["camera"].items():
    T_frames = len(cam_data["raw_depth"])
    tracks = np.zeros((T_frames, len(static_pts_3d), 2), dtype=np.float32)
    gaps = np.zeros((T_frames, len(static_pts_3d)), dtype=np.float32)

    for t in range(T_frames):
      u, v, gap = depth_gap(cam_data, static_pts_3d, scene_state[cam_id]["extrinsics"][t], t)
      tracks[t, :, 0] = u
      tracks[t, :, 1] = v
      gaps[t] = gap

    per_cam_tracks[cam_id] = tracks
    per_cam_gap[cam_id] = gaps
    per_cam_vis[cam_id] = np.abs(gaps) < depth_tolerance

  return per_cam_tracks, per_cam_vis, per_cam_gap


def filter_static_tracks(
  static_pts_3d,
  per_cam_tracks,
  per_cam_vis,
  per_cam_gap,
  tau=0.015,
  min_run_frames=30,
  flicker=0.10,
):
  vis = np.stack(list(per_cam_vis.values()))
  gap = np.stack(list(per_cam_gap.values()))
  n_cams, T_frames, n_points = vis.shape

  run = np.zeros((n_cams, n_points), dtype=np.int32)
  streak = np.zeros((n_cams, n_points), dtype=np.int32)
  for t in range(T_frames):
    run = np.where(gap[:, t] > tau, run + 1, 0)
    streak = np.maximum(streak, run)

  gone = (streak >= min_run_frames) & vis[:, 0]
  flips = (vis[:, 1:] != vis[:, :-1]).sum(axis=1)
  jitters = flips / max(T_frames - 1, 1) > flicker

  keep = np.flatnonzero(~(gone | jitters).any(axis=0))
  print(f"  Static: {len(keep)} of {n_points} candidates survive gone/jitter")
  return static_pts_3d[keep], keep_tracks(per_cam_tracks, keep), keep_tracks(per_cam_vis, keep)


def sample_tracks(static_pts_3d, per_cam_tracks, per_cam_vis, num_points=None, seed=42):
  keep = sample_indices(np.arange(len(static_pts_3d)), num_points, seed)
  return static_pts_3d[keep], keep_tracks(per_cam_tracks, keep), keep_tracks(per_cam_vis, keep)


def find_robot_candidates(
  scene_constants, scene_state, pb_renderer, safe_margin=7, max_robot_pts_per_cam=None
):
  robot = scene_constants["robot"]
  T_frames = len(robot["joint_positions"])
  kernel = np.ones((safe_margin, safe_margin), np.uint8)

  pb_renderer.update_robot_pose(
    robot["joint_positions"][0], gripper_state=robot["gripper_positions"][0]
  )

  seeds = []
  parts = []
  for src_cam, cam_data in scene_constants["camera"].items():
    K = cam_data["K_mat"]
    h_img, w_img = cam_data["raw_depth"][0].shape
    extrinsic = scene_state[src_cam]["extrinsics"][0]

    obj_ids, link_ids, urdf_depth = pb_renderer.render_segmentation(extrinsic, K, w_img, h_img)
    is_robot = (obj_ids == pb_renderer.robot_id).astype(np.uint8)
    on_robot = cv2.erode(is_robot, kernel, iterations=1) > 0

    flat = sample_indices(np.flatnonzero(on_robot), max_robot_pts_per_cam)
    vs, us = np.unravel_index(flat, on_robot.shape)

    seeds.append(
      core.geometry.unproject_pixels(
        us.astype(np.float32), vs.astype(np.float32), urdf_depth[vs, us], K, extrinsic
      )
    )
    parts.append(np.stack([obj_ids[vs, us], link_ids[vs, us]], axis=1))

  pts_world = np.concatenate(seeds)
  parts = np.concatenate(parts)

  local_pts = {}
  for part in {tuple(p) for p in parts}:
    on_part = (parts == part).all(axis=1)
    homogeneous = np.hstack([pts_world[on_part], np.ones((on_part.sum(), 1))]).T
    local_pts[part] = (on_part, np.linalg.inv(link_transform(*part)) @ homogeneous)

  traj_3d = np.zeros((T_frames, len(pts_world), 3), dtype=np.float32)
  for t in range(T_frames):
    pb_renderer.update_robot_pose(
      robot["joint_positions"][t], gripper_state=robot["gripper_positions"][t]
    )
    for part, (on_part, homogeneous) in local_pts.items():
      traj_3d[t, on_part] = (link_transform(*part) @ homogeneous)[:3].T

  return traj_3d


def project_robot_tracks(robot_traj_3d, scene_constants, scene_state, pb_renderer):
  robot = scene_constants["robot"]
  T_frames, n_points, _ = robot_traj_3d.shape

  per_cam_tracks = {}
  per_cam_vis = {}

  for cam_id, cam_data in scene_constants["camera"].items():
    K = cam_data["K_mat"]
    h_img, w_img = cam_data["raw_depth"][0].shape
    tracks = np.zeros((T_frames, n_points, 2), dtype=np.float32)
    vis = np.zeros((T_frames, n_points), dtype=bool)

    for t in range(T_frames):
      extrinsic = scene_state[cam_id]["extrinsics"][t]
      pb_renderer.update_robot_pose(
        robot["joint_positions"][t], gripper_state=robot["gripper_positions"][t]
      )
      urdf_depth = pb_renderer.render_depth(extrinsic, K, w_img, h_img)

      u, v, z_pred = core.geometry.project_points(robot_traj_3d[t], K, extrinsic)
      tracks[t, :, 0] = u
      tracks[t, :, 1] = v

      z_urdf = sample_depth(urdf_depth, u, v, z_pred)
      z_sensor = sample_depth(cam_data["raw_depth"][t], u, v, z_pred)
      facing_camera = (z_urdf > 0) & (z_pred <= z_urdf + 0.015)
      behind_scene = (z_sensor > 0) & (z_pred > z_sensor + 0.02)
      vis[t] = facing_camera & ~behind_scene

    per_cam_tracks[cam_id] = tracks
    per_cam_vis[cam_id] = vis

  return per_cam_tracks, per_cam_vis


def filter_robot_tracks(robot_traj_3d, per_cam_tracks, per_cam_vis):
  vis = np.stack(list(per_cam_vis.values()))
  keep = np.flatnonzero(vis.any(axis=(0, 1)))
  print(f"  Robot: {len(keep)} of {vis.shape[2]} candidates are visible somewhere")
  return robot_traj_3d[:, keep], keep_tracks(per_cam_tracks, keep), keep_tracks(per_cam_vis, keep)


def merge_tracks(static_pts_3d, static_tracks, static_vis, robot_traj_3d, robot_tracks, robot_vis):
  T_frames, n_robot, _ = robot_traj_3d.shape
  n_static = len(static_pts_3d)
  print(f"  Static: {n_static} | Robot: {n_robot} | Total: {n_static + n_robot}")

  static_traj_3d = np.broadcast_to(static_pts_3d[None], (T_frames, n_static, 3))
  return (
    np.concatenate([static_traj_3d, robot_traj_3d], axis=1),
    {cam: np.concatenate([static_tracks[cam], robot_tracks[cam]], axis=1) for cam in static_tracks},
    {cam: np.concatenate([static_vis[cam], robot_vis[cam]], axis=1) for cam in static_vis},
  )


def export_tracks(
  scene_constants, scene_state, traj_3d, per_cam_tracks, per_cam_vis, n_static, export_root
):
  ep_id = scene_constants["meta"]["episode_id"]
  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(export_root, ep_id)))
  os.makedirs(ep_dir, exist_ok=True)

  np.savez_compressed(
    os.path.join(ep_dir, "tracks_3d.npz"),
    traj_3d=traj_3d.astype(np.float32),
    vis_global=np.logical_or.reduce(list(per_cam_vis.values())),
  )

  for cam_id, cam_data in scene_constants["camera"].items():
    cam_dir = os.path.join(ep_dir, cam_id)
    os.makedirs(cam_dir, exist_ok=True)

    vis = per_cam_vis[cam_id]
    traj_2d = np.where(vis[:, :, None], per_cam_tracks[cam_id], -1000.0)
    np.savez_compressed(
      os.path.join(cam_dir, "tracks_2d.npz"), traj_2d=traj_2d.astype(np.float32), vis_2d=vis
    )

    K = cam_data["K_mat"]
    np.save(
      os.path.join(cam_dir, "intrinsics.npy"),
      np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32),
    )
    np.save(
      os.path.join(cam_dir, "extrinsics_w2c.npy"),
      np.linalg.inv(scene_state[cam_id]["extrinsics"]).astype(np.float32),
    )

  n_robot = traj_3d.shape[1] - n_static
  np.savez_compressed(
    os.path.join(ep_dir, "track_metadata.npz"),
    n_static=np.array(n_static),
    n_robot=np.array(n_robot),
    point_type=np.array([0] * n_static + [1] * n_robot, dtype=np.uint8),
  )


def process_episode(episode_id, pb_renderer, config):
  scene_constants = core.io.load_depth_data(episode_id, config.paths.depth, load_video=None)
  scene_state = core.io.load_extrinsics(scene_constants, config.paths.extrinsics)

  static_pts_3d = find_static_candidates(
    scene_constants,
    scene_state,
    pb_renderer,
    safe_margin=config.tracks.safe_margin,
    match_radius=config.tracks.match_radius,
  )
  static_tracks, static_vis, static_gap = project_static_tracks(
    static_pts_3d, scene_constants, scene_state, depth_tolerance=config.tracks.depth_tolerance
  )
  static_pts_3d, static_tracks, static_vis = filter_static_tracks(
    static_pts_3d,
    static_tracks,
    static_vis,
    static_gap,
    tau=config.tracks.tau,
    min_run_frames=config.tracks.min_run_frames,
    flicker=config.tracks.flicker,
  )
  static_pts_3d, static_tracks, static_vis = sample_tracks(
    static_pts_3d, static_tracks, static_vis, num_points=config.tracks.num_static_points
  )

  robot_traj_3d = find_robot_candidates(
    scene_constants,
    scene_state,
    pb_renderer,
    safe_margin=config.tracks.robot_safe_margin,
    max_robot_pts_per_cam=config.tracks.max_robot_pts_per_cam,
  )
  robot_tracks, robot_vis = project_robot_tracks(
    robot_traj_3d, scene_constants, scene_state, pb_renderer
  )
  robot_traj_3d, robot_tracks, robot_vis = filter_robot_tracks(
    robot_traj_3d, robot_tracks, robot_vis
  )

  traj_3d, per_cam_tracks, per_cam_vis = merge_tracks(
    static_pts_3d, static_tracks, static_vis, robot_traj_3d, robot_tracks, robot_vis
  )
  export_tracks(
    scene_constants,
    scene_state,
    traj_3d,
    per_cam_tracks,
    per_cam_vis,
    len(static_pts_3d),
    config.paths.tracks,
  )


def main(_):
  config = config_flag.value
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
    process_episode(episode_id, pb_renderer, config)

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
