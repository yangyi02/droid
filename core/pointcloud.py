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


def extract_robot_clouds(cam_id, episode, rendered, gripper_links, base_extrinsic, device, depth_batch, n_points):
  is_wrist = cam_id == episode["meta"]["wrist_serial"]
  K = episode["camera"][cam_id]["K"]
  link_ids, rendered_depth = rendered

  cache_X, kept = [], []
  for t in range(len(rendered_depth)):
    visible = rendered_depth[t] > 0
    if is_wrist:
      visible &= np.isin(link_ids[t], gripper_links)
    points_cam = sample_camera_points(visible, rendered_depth[t], K, n_points)
    if points_cam is None:
      continue

    cache_X.append(torch.tensor((base_extrinsic @ points_cam)[:3, :].T, dtype=torch.float32, device=device))
    kept.append(t)

  return torch.stack(cache_X), depth_batch[kept]


def robot_clouds(episode, poses, renders, gripper_links, device, n_points):
  robot_points, depth_batch, K = {}, {}, {}
  for cam_id, cam_data in episode["camera"].items():
    observed = torch.tensor(np.asarray(cam_data["raw_depth"], dtype=np.float32), device=device).unsqueeze(1)
    robot_points[cam_id], depth_batch[cam_id] = extract_robot_clouds(
      cam_id, episode, renders[cam_id], gripper_links, poses[cam_id]["base_extrinsic"], device, observed, n_points
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
  cache_ee, frames = [], []
  for t in range(len(episode["robot"]["joint_positions"])):
    frame = {cam_id: camera_frame_points(t, cam_data, n_points, max_depth) for cam_id, cam_data in cameras.items()}
    if all(points is not None for points in frame.values()):
      for cam_id, points in frame.items():
        cache[cam_id].append(torch.tensor(points, dtype=torch.float32, device=device))
      cache_ee.append(torch.tensor(T_ee2base[t], dtype=torch.float32, device=device))
      frames.append(t)

  return {cam_id: torch.stack(clouds) for cam_id, clouds in cache.items()}, torch.stack(cache_ee), np.array(frames)


def disparity_maps(episode, frames, device):
  maps = {}
  for cam_id, cam_data in episode["camera"].items():
    depth = torch.tensor(np.asarray(cam_data["raw_depth"][frames], dtype=np.float32), device=device)
    maps[cam_id] = torch.where(depth > 0, cam_data["K"][0, 0] * cam_data["baseline"] / depth.clamp(min=1e-6), 0.0)
  return maps


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


def sample_at_projection(points, T_cam2world, K, images):
  _, _, height, width = images.shape

  P_c = (points - T_cam2world[:3, 3]) @ T_cam2world[:3, :3]
  z_pred = P_c[..., 2]

  u = K[0, 0] * P_c[..., 0] / z_pred + K[0, 2]
  v = K[1, 1] * P_c[..., 1] / z_pred + K[1, 2]

  grid = torch.stack([(u / (width - 1)) * 2 - 1, (v / (height - 1)) * 2 - 1], dim=-1).unsqueeze(1)
  sampled = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(1).squeeze(1)

  uv_in_frame = (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
  return sampled, z_pred, uv_in_frame & (z_pred > 0.0)


def depth_loss_batched(points, T_cam2world, K, depth_batch, max_depth):
  z_obs, z_pred, in_frame = sample_at_projection(points, T_cam2world, K, depth_batch)
  valid = in_frame & (z_pred < max_depth) & (z_obs > 0.0) & (z_obs < max_depth)
  return torch.abs(z_obs[valid] - z_pred[valid]).mean()


def disparity_loss_batched(points, T_cam2world, K, depth_batch, baseline, truncation):
  z_obs, z_pred, in_frame = sample_at_projection(points, T_cam2world, K, depth_batch)
  measured, _, _ = sample_at_projection(points, T_cam2world, K, (depth_batch > 0.0).float())
  residual = K[0, 0] * baseline * torch.abs(1.0 / z_obs.clamp(min=1e-6) - 1.0 / z_pred.clamp(min=1e-6))
  residual = torch.where(in_frame, residual.clamp(max=truncation), torch.full_like(residual, truncation))
  counted = ~in_frame | (measured > 0.999)
  return residual[counted].mean()


def world_clouds(env, ee_poses, pose, wrist_cam_id):
  to_world, world = {}, {}
  for cam_id, cloud in env.items():
    to_world[cam_id] = ee_poses @ pose[cam_id] if cam_id == wrist_cam_id else pose[cam_id]
    world[cam_id] = (to_world[cam_id] @ cloud)[:, :3, :].transpose(1, 2)
  return to_world, world


def chamfer_overlap(env, ee_poses, pose, wrist_cam_id, pairs, match_radius):
  _, world = world_clouds(env, ee_poses, pose, wrist_cam_id)
  chamfer, overlap = {}, {}
  for a, b in pairs:
    chamfer[a, b], overlap[a, b] = batched_chamfer_distance(world[a], world[b], match_radius)

  return chamfer, overlap


def sparse_to_dense(points_world, T_cam2world, K, baseline, disparity, radius):
  _, height, width = disparity.shape
  P_c = (points_world - T_cam2world[..., :3, 3].unsqueeze(-2)) @ T_cam2world[..., :3, :3]
  z = P_c[..., 2].clamp(min=1e-6)
  u = K[0, 0] * P_c[..., 0] / z + K[0, 2]
  v = K[1, 1] * P_c[..., 1] / z + K[1, 2]
  d = K[0, 0] * baseline / z

  steps = torch.arange(-int(np.ceil(radius)), int(np.ceil(radius)) + 1, device=points_world.device)
  du, dv = torch.meshgrid(steps, steps, indexing="xy")
  pu = u.detach().round().unsqueeze(-1) + du.flatten()
  pv = v.detach().round().unsqueeze(-1) + dv.flatten()
  inside = (pu >= 0) & (pu < width) & (pv >= 0) & (pv < height) & (P_c[..., 2] > 0).unsqueeze(-1)
  index = (pv.clamp(0, height - 1) * width + pu.clamp(0, width - 1)).long()
  observed = torch.gather(disparity.flatten(1), 1, index.flatten(1)).view(index.shape)

  gap = (u.unsqueeze(-1) - pu) ** 2 + (v.unsqueeze(-1) - pv) ** 2 + (d.unsqueeze(-1) - observed) ** 2
  gap = torch.where(inside & (observed > 0), gap, torch.inf)
  nearest = torch.sqrt(gap.min(dim=-1).values + 1e-12)
  return nearest, nearest < radius


def chamfer_overlap_px(env, ee_poses, pose, wrist_cam_id, pairs, K, baseline, disparity, radius):
  to_world, world = world_clouds(env, ee_poses, pose, wrist_cam_id)
  chamfer, overlap = {}, {}
  for a, b in pairs:
    loss, hits, total = 0.0, 0, 0
    for source, target in ((b, a), (a, b)):
      nearest, matched = sparse_to_dense(
        world[source], to_world[target], K[target], baseline[target], disparity[target], radius
      )
      loss = loss + torch.where(matched, nearest, 0.0).sum() / matched.sum().clamp(min=1)
      hits, total = hits + matched.sum(), total + matched.numel()
    chamfer[a, b], overlap[a, b] = loss, hits / total

  return chamfer, overlap


def robot_depth_loss(robot_points, depth_batch, K, pose, max_depth):
  return {
    cam_id: depth_loss_batched(points, pose[cam_id], K[cam_id], depth_batch[cam_id], max_depth)
    for cam_id, points in robot_points.items()
  }
