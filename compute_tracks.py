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
  kernel = np.ones((abs(margin), abs(margin)), np.uint8)
  morph = cv2.dilate if margin > 0 else cv2.erode
  return morph(mask.astype(np.uint8), kernel).astype(bool)


def query_frames(episode, poses, pb_renderer, config):
  robot = episode["robot"]
  reach = len(robot["joint_positions"])

  seen = np.zeros((len(episode["camera"]), reach))
  for frame in range(reach):
    pb_renderer.update_robot_pose(robot["joint_positions"][frame], gripper_state=robot["gripper_positions"][frame])
    for view, (cam_id, cam_data) in enumerate(episode["camera"].items()):
      height, width = cam_data["raw_depth"][frame].shape
      drawn = pb_renderer.render_depth(poses[cam_id]["extrinsics"][frame], cam_data["K"], width, height)
      seen[view, frame] = resize_mask(drawn > 0, -config.tracks.mask_margin).sum()

  worst = (seen / np.maximum(seen.max(axis=1, keepdims=True), 1)).min(axis=0)
  edges = np.linspace(0, reach, config.tracks.num_query_frames + 1).astype(int)
  return np.array([0] + [lo + int(np.argmax(worst[lo:hi])) for lo, hi in zip(edges[1:-1], edges[2:])])


def spread_cells(points, min_gap):
  order = np.random.permutation(len(points))
  _, first = np.unique(np.round(points[order] / min_gap).astype(np.int64), axis=0, return_index=True)
  return order[first]


def sensor_slack(depth, u, v, z_pred, config):
  z = core.geometry.sample_depth(depth, u, v, z_pred)
  gap = np.where(z == 0, np.inf, z) - z_pred
  return gap / (config.tracks.sensor_tolerance_base + config.tracks.sensor_tolerance_slope * z_pred)


def urdf_gap(depth, u, v, z_pred):
  z = core.geometry.sample_depth(depth, u, v, z_pred)
  return np.where(z == 0, np.inf, z) - z_pred


def part_masks(parts):
  return [(part, (parts == part).all(axis=1)) for part in map(tuple, np.unique(parts, axis=0).tolist())]


def link_local(points_world, parts):
  homogeneous = np.hstack([points_world, np.ones((len(points_world), 1))]).T
  local = np.zeros_like(homogeneous)
  for part, on_part in part_masks(parts):
    local[:, on_part] = np.linalg.inv(core.physics.link_transform(*part)) @ homogeneous[:, on_part]
  return local


def find_robot_candidates(episode, poses, pb_renderer, queries, config):
  robot = episode["robot"]

  local, parts, query_view, query_frame = [], [], [], []
  for view, (src_cam, cam_data) in enumerate(episode["camera"].items()):
    for frame in queries:
      pb_renderer.update_robot_pose(robot["joint_positions"][frame], gripper_state=robot["gripper_positions"][frame])

      K = cam_data["K"]
      height, width = cam_data["raw_depth"][frame].shape
      T_cam2world = poses[src_cam]["extrinsics"][frame]

      obj_ids, link_ids, urdf_depth = pb_renderer.render_segmentation(T_cam2world, K, width, height)
      vs, us = np.where(resize_mask(obj_ids == pb_renderer.robot_id, -config.tracks.mask_margin))

      u, v, z = us.astype(np.float32), vs.astype(np.float32), urdf_depth[vs, us]
      sensor = sensor_slack(cam_data["raw_depth"][frame], u, v, z, config)
      lit = np.isinf(sensor) | (sensor >= -1)

      surface = core.geometry.unproject_pixels(u[lit], v[lit], z[lit], K, T_cam2world)
      cell = spread_cells(surface, config.tracks.min_gap)
      on_parts = np.stack([obj_ids[vs, us][lit], link_ids[vs, us][lit]], axis=1)[cell]

      local.append(link_local(surface[cell], on_parts))
      parts.append(on_parts)
      query_view.append(np.full(len(cell), view, dtype=np.int8))
      query_frame.append(np.full(len(cell), frame, dtype=np.int32))

  return (
    np.concatenate(local, axis=1),
    np.concatenate(parts),
    np.concatenate(query_view),
    np.concatenate(query_frame),
  )


def carry_robot(local, parts, robot, pb_renderer, frames):
  carried = np.zeros((len(frames), local.shape[1], 3), dtype=np.float32)
  on_links = part_masks(parts)
  for step, t in enumerate(frames):
    pb_renderer.update_robot_pose(robot["joint_positions"][t], gripper_state=robot["gripper_positions"][t])
    for part, on_part in on_links:
      carried[step, on_part] = (core.physics.link_transform(*part) @ local[:, on_part])[:3].T
  return carried


def depth_steps(depth, window=5):
  kernel = np.ones((window, window), np.uint8)
  far = cv2.dilate(depth, kernel)
  near = -cv2.dilate(np.where(depth > 0, -depth, -np.inf).astype(np.float32), kernel)
  return np.where(far > 0, far - near, np.inf)


def find_static_candidates(episode, poses, pb_renderer, queries, config):
  robot = episode["robot"]
  match_radius, max_depth = config.tracks.match_radius, config.tracks.max_depth
  max_step = config.tracks.max_edge_step

  points_3d, query_view, query_frame = [], [], []
  for frame in queries:
    pb_renderer.update_robot_pose(robot["joint_positions"][frame], gripper_state=robot["gripper_positions"][frame])

    drawn, steps = {}, {}
    for cam_id, cam_data in episode["camera"].items():
      height, width = cam_data["raw_depth"][frame].shape
      drawn[cam_id] = pb_renderer.render_depth(poses[cam_id]["extrinsics"][frame], cam_data["K"], width, height)
      steps[cam_id] = depth_steps(cam_data["raw_depth"][frame])

    for view, (src_cam, cam_data) in enumerate(episode["camera"].items()):
      depth = cam_data["raw_depth"][frame]
      K, T_cam2world = cam_data["K"], poses[src_cam]["extrinsics"][frame]

      on_env = ~resize_mask(drawn[src_cam] > 0, config.tracks.mask_margin) & (depth > 0) & (depth <= max_depth)
      vs, us = np.where(on_env & (steps[src_cam] <= max_step))

      points = core.geometry.unproject_pixels(
        us.astype(np.float32), vs.astype(np.float32), depth[vs, us], K, T_cam2world
      )

      confirmed = np.zeros(len(points), dtype=bool)
      doubted = np.zeros(len(points), dtype=bool)
      for other_cam, other_data in episode["camera"].items():
        if other_cam == src_cam:
          continue
        u, v, z = core.geometry.project_points(points, other_data["K"], poses[other_cam]["extrinsics"][frame])
        z_other = core.geometry.sample_depth(other_data["raw_depth"][frame], u, v, z)
        step = core.geometry.sample_depth(steps[other_cam], u, v, z)
        behind_arm = core.geometry.sample_depth(drawn[other_cam], u, v, z) > 0

        speaks = np.isfinite(z_other) & (z_other > 0) & ~behind_arm
        agrees = np.abs(z_other - z) < match_radius
        confirmed |= speaks & agrees & (z_other <= max_depth)
        doubted |= speaks & (~agrees | ~(step <= max_step))

      cell = spread_cells(points[confirmed & ~doubted], config.tracks.min_gap)
      points_3d.append(points[confirmed & ~doubted][cell])
      query_view.append(np.full(len(cell), view, dtype=np.int8))
      query_frame.append(np.full(len(cell), frame, dtype=np.int32))

  return np.concatenate(points_3d).astype(np.float32), np.concatenate(query_view), np.concatenate(query_frame)


def project_tracks(tracks_3d, n_robot, episode, poses, pb_renderer, config):
  robot = episode["robot"]
  n_frames, n_points, _ = tracks_3d.shape
  n_views = len(episode["camera"])

  uv = np.zeros((n_views, n_frames, n_points, 2), dtype=np.float32)
  margin = np.zeros((n_views, n_frames, n_points), dtype=np.float32)
  inside = np.zeros((n_views, n_frames, n_points), dtype=bool)
  slack = np.zeros((n_views, n_frames, n_points), dtype=np.float32)

  for t in range(n_frames):
    pb_renderer.update_robot_pose(robot["joint_positions"][t], gripper_state=robot["gripper_positions"][t])

    drawn = [
      pb_renderer.render_depth(poses[cam_id]["extrinsics"][t], cam_data["K"], *cam_data["raw_depth"][t].shape[::-1])
      for cam_id, cam_data in episode["camera"].items()
    ]

    for view, (cam_id, cam_data) in enumerate(episode["camera"].items()):
      K = cam_data["K"]
      T_cam2world = poses[cam_id]["extrinsics"][t]
      urdf_depth = drawn[view]

      height, width = cam_data["raw_depth"][t].shape
      u, v, z_pred = core.geometry.project_points(tracks_3d[t], K, T_cam2world)
      uv[view, t] = np.stack([u, v], axis=1)
      inside[view, t] = core.geometry.in_frame(u, v, width, height) & (z_pred > 0)

      urdf = urdf_gap(urdf_depth, u, v, z_pred)
      sensor = sensor_slack(cam_data["raw_depth"][t], u, v, z_pred, config)

      margin[view, t] = np.fmin(urdf / config.tracks.urdf_tolerance, np.where(np.isinf(sensor), np.nan, sensor))
      slack[view, t] = sensor

  read = np.concatenate(
    [latch(margin[:, :, :n_robot], config.tracks.hysteresis), latch(margin[:, :, n_robot:], 0.0)], axis=2
  )
  return uv, settle(read, inside), slack


def latch(margin, band):
  hidden, shown = margin < -1 - band, margin > -1 + band

  vis = np.empty(margin.shape, dtype=bool)
  vis[:, 0] = ~(margin[:, 0] < -1)
  for t in range(1, margin.shape[1]):
    vis[:, t] = np.where(hidden[:, t], False, np.where(shown[:, t], True, vis[:, t - 1]))
  return vis


def settle(vis, inside):
  settled = vis
  for _ in range(4):
    middle = settled[:, 1:-1]
    alone = (middle != settled[:, :-2]) & (middle != settled[:, 2:])
    if not alone.any():
      break
    settled = settled.copy()
    settled[:, 1:-1] = np.where(alone, ~middle, middle)
  return settled & inside


def never_seen_through(slack, max_seen_through):
  seen_through = np.isfinite(slack) & (slack > 1)
  clear_line = np.isfinite(slack) & (slack >= -1)
  return ~(seen_through.sum(axis=1) / np.maximum(clear_line.sum(axis=1), 1) > max_seen_through).any(axis=0)


def out_of_reach(points_3d, episode, pb_renderer, clearance):
  robot = episode["robot"]

  closest = np.full(len(points_3d), np.inf, dtype=np.float32)
  for t in range(len(robot["joint_positions"])):
    pb_renderer.update_robot_pose(robot["joint_positions"][t], gripper_state=robot["gripper_positions"][t])
    links = np.array(
      [core.physics.link_transform(pb_renderer.robot_id, link)[:3, 3] for link in pb_renderer.gripper_links]
    )
    closest = np.minimum(closest, np.linalg.norm(links[:, None] - points_3d[None], axis=-1).min(axis=0))

  return closest > clearance


def take(picked, home, group, quota):
  if quota <= 0 or not len(group):
    return picked
  return np.concatenate([picked, group[core.geometry.farthest_points(home[group], quota, seeds=home[picked])]])


def sample_tracks(keep, home, query_view, query_frame, is_robot, queries, config):
  on_arm = in_scene = np.empty(0, dtype=int)
  for frame in queries:
    for view in range(int(query_view.max()) + 1):
      born_here = keep & (query_view == view) & (query_frame == frame)

      on_arm = take(on_arm, home, np.flatnonzero(born_here & is_robot), config.tracks.points_per_class)
      in_scene = take(in_scene, home, np.flatnonzero(born_here & ~is_robot), config.tracks.points_per_class)

  return np.sort(np.concatenate([on_arm, in_scene]))


def export_tracks(episode, tracks_3d, uv, vis, query_view, query_frame, n_robot, export_root):
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
    n_static=np.array(uv.shape[2] - n_robot),
    query_view=query_view,
    query_frame=query_frame,
  )


def process_episode(episode_id, pb_renderer, config):
  episode = core.io.load_depth_data(episode_id, config.paths.depth)
  poses = core.io.load_extrinsics(episode, config.paths.extrinsics)
  robot = episode["robot"]
  n_frames = len(robot["joint_positions"])

  queries = query_frames(episode, poses, pb_renderer, config)
  local, parts, robot_view, robot_frame = find_robot_candidates(episode, poses, pb_renderer, queries, config)
  static_3d, static_view, static_frame = find_static_candidates(episode, poses, pb_renderer, queries, config)

  query_view = np.concatenate([robot_view, static_view])
  query_frame = np.concatenate([robot_frame, static_frame])
  is_robot = np.arange(len(query_view)) < local.shape[1]
  home = np.concatenate([carry_robot(local, parts, robot, pb_renderer, [0])[0], static_3d])

  keep = np.ones(len(query_view), dtype=bool)
  keep[~is_robot] &= out_of_reach(static_3d, episode, pb_renderer, config.tracks.gripper_clearance)

  idx = sample_tracks(keep, home, query_view, query_frame, is_robot, queries, config)
  on_arm, in_scene = idx[is_robot[idx]], idx[~is_robot[idx]] - local.shape[1]

  tracks_3d = np.concatenate(
    [
      carry_robot(local[:, on_arm], parts[on_arm], robot, pb_renderer, range(n_frames)),
      np.broadcast_to(static_3d[in_scene], (n_frames, len(in_scene), 3)),
    ],
    axis=1,
  )
  query_view, query_frame = query_view[idx], query_frame[idx]
  n_robot = len(on_arm)

  uv, vis, slack = project_tracks(tracks_3d, n_robot, episode, poses, pb_renderer, config)

  keep = vis[query_view, query_frame, np.arange(len(query_view))]
  keep[n_robot:] &= never_seen_through(slack[:, :, n_robot:], config.tracks.max_seen_through)

  n_robot = int(keep[:n_robot].sum())
  print(
    f"  {int(keep.sum())} points: {n_robot} robot, {int(keep.sum()) - n_robot} static"
    f" | per view {np.bincount(query_view[keep])}"
  )

  export_tracks(
    episode,
    tracks_3d[:, keep],
    uv[:, :, keep],
    vis[:, :, keep],
    query_view[keep],
    query_frame[keep],
    n_robot,
    config.paths.tracks,
  )


def main(_):
  config = config_flag.value
  pb_renderer = core.physics.PyBulletRenderer(config.paths.urdf, gpu=config.render.gpu)

  available = core.runner.list_episode_dirs(config.paths.extrinsics)
  if config.paths.episode_list:
    available &= core.io.read_episode_list(config.paths.episode_list)

  target = core.runner.shard_episodes(
    available,
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
