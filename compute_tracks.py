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


def resize_mask(mask, margin):
  """Grow the mask by margin pixels, or shrink it when margin is negative."""
  kernel = np.ones((abs(margin), abs(margin)), np.uint8)
  morph = cv2.dilate if margin > 0 else cv2.erode
  return morph(mask.astype(np.uint8), kernel).astype(bool)


def find_robot_candidates(episode, poses, pb_renderer, mask_margin):
  robot = episode["robot"]
  n_frames = len(robot["joint_positions"])

  pb_renderer.update_robot_pose(robot["joint_positions"][0], gripper_state=robot["gripper_positions"][0])

  seeds = []
  parts = []
  query_view = []
  for view, (src_cam, cam_data) in enumerate(episode["camera"].items()):
    K = cam_data["K"]
    height, width = cam_data["raw_depth"][0].shape
    T_cam2world = poses[src_cam]["extrinsics"][0]

    obj_ids, link_ids, urdf_depth = pb_renderer.render_segmentation(T_cam2world, K, width, height)
    vs, us = np.where(resize_mask(obj_ids == pb_renderer.robot_id, -mask_margin))

    seeds.append(
      core.geometry.unproject_pixels(us.astype(np.float32), vs.astype(np.float32), urdf_depth[vs, us], K, T_cam2world)
    )
    parts.append(np.stack([obj_ids[vs, us], link_ids[vs, us]], axis=1))
    query_view.append(np.full(len(vs), view, dtype=np.int8))

  points_world = np.concatenate(seeds)
  parts = np.concatenate(parts)

  local_points = {}
  for part in map(tuple, np.unique(parts, axis=0).tolist()):
    on_part = (parts == part).all(axis=1)
    homogeneous = np.hstack([points_world[on_part], np.ones((on_part.sum(), 1))]).T
    local_points[part] = (on_part, np.linalg.inv(core.physics.link_transform(*part)) @ homogeneous)

  tracks_3d = np.zeros((n_frames, len(points_world), 3), dtype=np.float32)
  for t in range(n_frames):
    pb_renderer.update_robot_pose(robot["joint_positions"][t], gripper_state=robot["gripper_positions"][t])
    for part, (on_part, homogeneous) in local_points.items():
      tracks_3d[t, on_part] = (core.physics.link_transform(*part) @ homogeneous)[:3].T

  return tracks_3d, np.concatenate(query_view)


def project_robot_tracks(robot_tracks_3d, episode, poses, pb_renderer, depth_tolerance):
  robot = episode["robot"]
  n_frames, n_points, _ = robot_tracks_3d.shape
  n_views = len(episode["camera"])

  uv = np.zeros((n_views, n_frames, n_points, 2), dtype=np.float32)
  vis = np.zeros((n_views, n_frames, n_points), dtype=bool)

  for t in range(n_frames):
    pb_renderer.update_robot_pose(robot["joint_positions"][t], gripper_state=robot["gripper_positions"][t])

    for view, (cam_id, cam_data) in enumerate(episode["camera"].items()):
      K = cam_data["K"]
      height, width = cam_data["raw_depth"][t].shape
      T_cam2world = poses[cam_id]["extrinsics"][t]
      urdf_depth = pb_renderer.render_depth(T_cam2world, K, width, height)

      u, v, z_pred = core.geometry.project_points(robot_tracks_3d[t], K, T_cam2world)
      uv[view, t] = np.stack([u, v], axis=1)

      z_urdf = core.geometry.sample_depth(urdf_depth, u, v, z_pred)
      urdf_gap = np.where(z_urdf == 0, np.inf, z_urdf) - z_pred
      vis[view, t] = urdf_gap >= -depth_tolerance

  return uv, vis


def filter_robot_tracks(vis, flicker):
  n_points = vis.shape[2]

  jitters = (vis[:, 1:] != vis[:, :-1]).mean(axis=1) > flicker

  keep = ~jitters.any(axis=0)
  print(f"  Robot: {keep.sum()} of {n_points} candidates survive jitter")
  return keep


def find_static_candidates(episode, poses, pb_renderer, match_radius, mask_margin):
  robot = episode["robot"]
  pb_renderer.update_robot_pose(robot["joint_positions"][0], gripper_state=robot["gripper_positions"][0])

  verified = []
  query_view = []
  for view, (src_cam, cam_data) in enumerate(episode["camera"].items()):
    depth = cam_data["raw_depth"][0]
    height, width = depth.shape

    robot_mask = pb_renderer.render_mask(poses[src_cam]["extrinsics"][0], cam_data["K"], width, height)
    on_env = ~resize_mask(robot_mask, mask_margin) & (depth > 0)
    vs, us = np.where(on_env)

    points = core.geometry.unproject_pixels(
      us.astype(np.float32),
      vs.astype(np.float32),
      depth[vs, us],
      cam_data["K"],
      poses[src_cam]["extrinsics"][0],
    )

    matched = np.zeros(len(points), dtype=bool)
    for other_cam, other_data in episode["camera"].items():
      if other_cam == src_cam:
        continue
      u, v, z = core.geometry.project_points(points, other_data["K"], poses[other_cam]["extrinsics"][0])
      matched |= np.abs(core.geometry.sample_depth(other_data["raw_depth"][0], u, v, z) - z) < match_radius

    verified.append(points[matched])
    query_view.append(np.full(matched.sum(), view, dtype=np.int8))

  return np.concatenate(verified).astype(np.float32), np.concatenate(query_view)


def project_static_tracks(static_points_3d, episode, poses, pb_renderer, depth_tolerance):
  robot = episode["robot"]
  n_frames = len(robot["joint_positions"])
  n_views, n_points = len(episode["camera"]), len(static_points_3d)

  uv = np.zeros((n_views, n_frames, n_points, 2), dtype=np.float32)
  vis = np.zeros((n_views, n_frames, n_points), dtype=bool)
  gap = np.zeros((n_views, n_frames, n_points), dtype=np.float32)

  for t in range(n_frames):
    pb_renderer.update_robot_pose(robot["joint_positions"][t], gripper_state=robot["gripper_positions"][t])

    for view, (cam_id, cam_data) in enumerate(episode["camera"].items()):
      K = cam_data["K"]
      height, width = cam_data["raw_depth"][t].shape
      T_cam2world = poses[cam_id]["extrinsics"][t]
      urdf_depth = pb_renderer.render_depth(T_cam2world, K, width, height)

      u, v, z_pred = core.geometry.project_points(static_points_3d, K, T_cam2world)
      uv[view, t] = np.stack([u, v], axis=1)

      z_urdf = core.geometry.sample_depth(urdf_depth, u, v, z_pred)
      z_sensor = core.geometry.sample_depth(cam_data["raw_depth"][t], u, v, z_pred)
      measured = np.stack([z_urdf, z_sensor])
      urdf_gap, sensor_gap = np.where(measured == 0, np.inf, measured) - z_pred
      vis[view, t] = np.minimum(urdf_gap, sensor_gap) >= -depth_tolerance
      gap[view, t] = sensor_gap

  return uv, vis, gap


def filter_static_tracks(vis, gap, depth_tolerance, min_run_fraction, flicker):
  _, n_frames, n_points = vis.shape
  min_frames = int(min_run_fraction * n_frames)

  seen_through = np.isfinite(gap) & (gap > depth_tolerance)
  windows = np.lib.stride_tricks.sliding_window_view(seen_through, min_frames, axis=1)

  gone = windows.all(axis=-1).any(axis=1) & vis[:, 0]
  jitters = (vis[:, 1:] != vis[:, :-1]).mean(axis=1) > flicker

  keep = ~(gone | jitters).any(axis=0)
  print(f"  Static: {keep.sum()} of {n_points} candidates survive gone/jitter")
  return keep


def sample_tracks(keep, xyz, uv, vis, query_view, n_points):
  per_view = []
  for view, view_vis in enumerate(vis):
    own = np.flatnonzero(keep & (query_view == view) & view_vis[0])
    per_view.append(np.random.permutation(own)[:n_points])

  idx = np.sort(np.concatenate(per_view))
  return xyz[..., idx, :], uv[:, :, idx], vis[:, :, idx], query_view[idx]


def merge_tracks(robot, static):
  robot_xyz, robot_uv, robot_vis, robot_view = robot
  static_xyz, static_uv, static_vis, static_view = static

  n_views = len(static_vis)
  n_frames, n_robot, _ = robot_xyz.shape
  n_static = len(static_xyz)

  robot_per_view = np.bincount(robot_view, minlength=n_views)
  static_per_view = np.bincount(static_view, minlength=n_views)
  print(f"  Robot: {robot_per_view} | Static: {static_per_view} | Total: {n_robot + n_static}")

  return (
    np.concatenate([robot_xyz, np.broadcast_to(static_xyz, (n_frames, n_static, 3))], axis=1),
    np.concatenate([robot_uv, static_uv], axis=2),
    np.concatenate([robot_vis, static_vis], axis=2),
    np.concatenate([robot_view, static_view]),
    n_robot,
    n_static,
  )


def export_tracks(episode, tracks_3d, uv, vis, query_view, n_robot, n_static, export_root):
  episode_id = episode["meta"]["episode_id"]
  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(export_root, episode_id)))
  os.makedirs(ep_dir, exist_ok=True)

  np.savez_compressed(os.path.join(ep_dir, "tracks_3d.npz"), tracks_3d=tracks_3d.astype(np.float32))

  for view, cam_id in enumerate(episode["camera"]):
    cam_dir = os.path.join(ep_dir, cam_id)
    os.makedirs(cam_dir, exist_ok=True)

    np.savez_compressed(os.path.join(cam_dir, "tracks_2d.npz"), tracks_2d=uv[view], vis_2d=vis[view])

  np.savez_compressed(
    os.path.join(ep_dir, "track_metadata.npz"),
    n_robot=np.array(n_robot),
    n_static=np.array(n_static),
    query_view=query_view,
  )


def process_episode(episode_id, pb_renderer, config):
  episode = core.io.load_depth_data(episode_id, config.paths.depth)
  poses = core.io.load_extrinsics(episode, config.paths.extrinsics)

  robot_xyz, robot_view = find_robot_candidates(episode, poses, pb_renderer, config.tracks.mask_margin)
  robot_uv, robot_vis = project_robot_tracks(
    robot_xyz, episode, poses, pb_renderer, config.tracks.robot_depth_tolerance
  )
  robot_keep = filter_robot_tracks(robot_vis, config.tracks.flicker)
  robot = sample_tracks(
    robot_keep, robot_xyz, robot_uv, robot_vis, robot_view, config.tracks.num_robot_points_per_view
  )

  static_xyz, static_view = find_static_candidates(
    episode, poses, pb_renderer, config.tracks.match_radius, config.tracks.mask_margin
  )
  static_uv, static_vis, static_gap = project_static_tracks(
    static_xyz, episode, poses, pb_renderer, config.tracks.static_depth_tolerance
  )
  static_keep = filter_static_tracks(
    static_vis,
    static_gap,
    config.tracks.static_depth_tolerance,
    config.tracks.min_run_fraction,
    config.tracks.flicker,
  )
  static = sample_tracks(
    static_keep, static_xyz, static_uv, static_vis, static_view, config.tracks.num_static_points_per_view
  )

  tracks_3d, uv, vis, query_view, n_robot, n_static = merge_tracks(robot, static)
  export_tracks(episode, tracks_3d, uv, vis, query_view, n_robot, n_static, config.paths.tracks)


def main(_):
  config = config_flag.value
  pb_renderer = core.physics.PyBulletRenderer(config.paths.urdf, gpu=config.render.gpu)

  target = core.runner.shard_episodes(
    core.runner.list_episode_dirs(config.paths.extrinsics),
    config.runner.rank,
    config.runner.world_size,
    config.runner.limit,
  )
  export_root = os.path.abspath(os.path.expanduser(config.paths.tracks))
  done = {e for e in target if os.path.exists(os.path.join(export_root, e, "tracks_3d.npz"))}

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
