import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def decode_disparity(disp, fx, baseline):
  z = np.zeros_like(disp)
  valid_mask = disp > 0
  z[valid_mask] = (fx * baseline) / disp[valid_mask]
  return z


def unproject_camera_frame(u, v, z, K):
  """Camera-frame points as a [4, N] homogeneous array, ready for a T_cam2world @ points."""
  return np.stack(
    [(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], axis=0
  )


def unproject_pixels(u, v, z, K, T_cam2world):
  return (T_cam2world @ unproject_camera_frame(u, v, z, K))[:3, :].T


def project_points(points_world, K, T_cam2world):
  T_world2cam = np.linalg.inv(T_cam2world)
  points_homo = np.hstack([points_world, np.ones((len(points_world), 1))]).T
  points_cam = T_world2cam @ points_homo
  z_cam = points_cam[2, :]
  u = np.zeros_like(z_cam)
  v = np.zeros_like(z_cam)
  valid = z_cam > 0
  u[valid] = (points_cam[0, valid] / z_cam[valid]) * K[0, 0] + K[0, 2]
  v[valid] = (points_cam[1, valid] / z_cam[valid]) * K[1, 1] + K[1, 2]
  return u, v, z_cam


def unproject_depth(depth, img_rgb, K, T_cam2world, max_depth=1.5):
  mask = (depth > 0) & (depth < max_depth)
  v, u = np.where(mask)
  return unproject_pixels(u, v, depth[mask], K, T_cam2world), img_rgb[mask]


def unproject_depth_torch(depth, img_rgb, K, T_cam2world, device, max_depth=1.5):
  depth = torch.as_tensor(depth, device=device)
  v, u = torch.nonzero((depth > 0) & (depth < max_depth), as_tuple=True)
  z = depth[v, u]
  K = torch.as_tensor(K, dtype=torch.float32, device=device)
  points_cam = torch.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], dim=1)
  T = torch.as_tensor(T_cam2world, dtype=torch.float32, device=device)
  points_world = points_cam @ T[:3, :3].T + T[:3, 3]
  return points_world, torch.as_tensor(img_rgb, device=device)[v, u]


def pose_from_euler(vec_6d):
  transform = np.eye(4)
  transform[:3, :3] = R.from_euler('xyz', vec_6d[3:]).as_matrix()
  transform[:3, 3] = vec_6d[:3]
  return transform


def axis_angle_to_matrix(v):
  theta2 = torch.sum(v**2)
  theta = torch.sqrt(theta2 + 1e-16)
  k = v / theta

  K = torch.zeros((3, 3), device=v.device)
  K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]

  R_exact = (
    torch.eye(3, device=v.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.mm(K, K)
  )

  Ka = torch.zeros_like(K)
  Ka[0, 1], Ka[0, 2], Ka[1, 0], Ka[1, 2], Ka[2, 0], Ka[2, 1] = -v[2], v[1], v[2], -v[0], -v[1], v[0]

  return torch.where(theta2 < 1e-8, torch.eye(3, device=v.device) + Ka, R_exact)


def pose_from_axis_angle(delta, device):
  rot = axis_angle_to_matrix(delta[3:])
  t = delta[:3].unsqueeze(1)
  top_rows = torch.cat([rot, t], dim=1)
  bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
  return torch.cat([top_rows, bottom_row], dim=0)
