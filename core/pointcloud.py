import numpy as np
import torch
import torch.nn.functional as F

import core.geometry


def sample_camera_points(mask, z, K, n_points):
  v, u = np.where(mask)
  if len(u) < n_points:
    return None

  idx = np.random.choice(len(u), n_points, replace=False)
  v, u = v[idx], u[idx]
  return core.geometry.unproject_camera_frame(u, v, z[v, u], K)


def foreground_points(T_cam2world, K, height, width, pb_renderer, n_points, links=None):
  _, link_ids, metric = pb_renderer.render_segmentation(T_cam2world, K, width, height)

  visible = metric > 0
  if links is not None:
    visible &= np.isin(link_ids, links)

  return sample_camera_points(visible, metric, K, n_points)


def extract_robot_clouds(cam_id, episode, pb_renderer, base_extrinsic, device, depth_batch, n_points):
  is_wrist = cam_id == episode["meta"]["wrist_serial"]
  T_ee_base_all = episode["robot"]["T_ee_base_all"]
  cam_data = episode["camera"][cam_id]
  K = cam_data["K"]
  height, width = cam_data["raw_depth"][0].shape

  cache_X, kept = [], []
  n_frames = len(episode["robot"]["joint_positions"])
  for t in range(n_frames):
    pb_renderer.update_robot_pose(episode["robot"]["joint_positions"][t], episode["robot"]["gripper_positions"][t])

    T_cam2world = T_ee_base_all[t] @ base_extrinsic if is_wrist else base_extrinsic
    links = pb_renderer.gripper_links if is_wrist else None
    points_cam = foreground_points(T_cam2world, K, height, width, pb_renderer, n_points, links)
    if points_cam is None:
      continue

    cache_X.append(torch.tensor((base_extrinsic @ points_cam)[:3, :].T, dtype=torch.float32, device=device))
    kept.append(t)

  return torch.stack(cache_X), depth_batch[kept]


def robot_clouds(episode, poses, pb_renderer, device, n_points):
  robot_points, depth_batch, K = {}, {}, {}
  for cam_id, cam_data in episode["camera"].items():
    base_extrinsic = poses[cam_id]["base_extrinsic"]
    observed = torch.tensor(np.asarray(cam_data["raw_depth"], dtype=np.float32), device=device).unsqueeze(1)
    robot_points[cam_id], depth_batch[cam_id] = extract_robot_clouds(
      cam_id, episode, pb_renderer, base_extrinsic, device, observed, n_points
    )
    K[cam_id] = torch.tensor(cam_data["K"], dtype=torch.float32, device=device)

  return robot_points, depth_batch, K


def camera_frame_points(t, cam_data, n_points, max_depth):
  depth = cam_data["raw_depth"][t].astype(np.float32)
  mask = (depth > 0.0) & (depth < max_depth)
  return sample_camera_points(mask, depth, cam_data["K"], n_points)


def scene_clouds(episode, device, n_points, max_depth):
  cameras = episode["camera"]
  T_ee2base = episode["robot"]["T_ee_base_all"]

  cache = {cam_id: [] for cam_id in cameras}
  cache_ee = []
  for t in range(len(episode["robot"]["joint_positions"])):
    frame = {cam_id: camera_frame_points(t, cam_data, n_points, max_depth) for cam_id, cam_data in cameras.items()}
    if all(points is not None for points in frame.values()):
      for cam_id, points in frame.items():
        cache[cam_id].append(torch.tensor(points, dtype=torch.float32, device=device))
      cache_ee.append(torch.tensor(T_ee2base[t], dtype=torch.float32, device=device))

  return {cam_id: torch.stack(clouds) for cam_id, clouds in cache.items()}, torch.stack(cache_ee)


def batched_chamfer_distance(p1, p2, match_radius):
  dist = torch.cdist(p1, p2)
  near_12 = dist.min(dim=2)[0]
  near_21 = dist.min(dim=1)[0]

  valid_12 = near_12 < match_radius
  valid_21 = near_21 < match_radius
  loss = (near_12 * valid_12).sum() / valid_12.sum().clamp(min=1)
  loss = loss + (near_21 * valid_21).sum() / valid_21.sum().clamp(min=1)

  overlap = (valid_12.sum() + valid_21.sum()) / (p1.shape[0] * (p1.shape[1] + p2.shape[1]))
  return loss, overlap


def depth_loss_batched(points, T_cam2world, K, depth_batch, max_depth):
  _, _, height, width = depth_batch.shape

  P_c = (points - T_cam2world[:3, 3]) @ T_cam2world[:3, :3]
  z_pred = P_c[..., 2]

  u = K[0, 0] * P_c[..., 0] / z_pred + K[0, 2]
  v = K[1, 1] * P_c[..., 1] / z_pred + K[1, 2]

  grid = torch.stack([(u / (width - 1)) * 2 - 1, (v / (height - 1)) * 2 - 1], dim=-1).unsqueeze(1)

  z_obs = (
    F.grid_sample(depth_batch, grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(1).squeeze(1)
  )

  depth_in_range = (z_pred > 0.0) & (z_pred < max_depth) & (z_obs > 0.0) & (z_obs < max_depth)
  uv_in_frame = (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
  valid = depth_in_range & uv_in_frame

  return torch.abs(z_obs[valid] - z_pred[valid]).mean()


def chamfer_overlap(env, ee_poses, pose, wrist_cam_id, pairs, match_radius):
  world = {}
  for cam_id, cloud in env.items():
    to_world = ee_poses @ pose[cam_id] if cam_id == wrist_cam_id else pose[cam_id]
    world[cam_id] = (to_world @ cloud)[:, :3, :].transpose(1, 2)

  chamfer, overlap = {}, {}
  for a, b in pairs:
    chamfer[a, b], overlap[a, b] = batched_chamfer_distance(world[a], world[b], match_radius)

  return chamfer, overlap


def robot_depth_loss(robot_points, depth_batch, K, pose, max_depth):
  return {
    cam_id: depth_loss_batched(points, pose[cam_id], K[cam_id], depth_batch[cam_id], max_depth)
    for cam_id, points in robot_points.items()
  }
