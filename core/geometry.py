import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def decode_disparity(disp, fx, baseline):
  z = np.zeros_like(disp)
  valid_mask = disp > 0
  z[valid_mask] = (fx * baseline) / disp[valid_mask]
  return z


def unproject_points(u, v, z, K, T_cam2world=None):
  x_cam = (u - K[0, 2]) * z / K[0, 0]
  y_cam = (v - K[1, 2]) * z / K[1, 1]
  pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=0)
  if T_cam2world is None:
    return pts_cam[:3, :].T
  return (T_cam2world @ pts_cam)[:3, :].T


def project_points(pts_world, K, T_cam2world):
  T_world2cam = np.linalg.inv(T_cam2world)
  pts_homo = np.hstack([pts_world, np.ones((len(pts_world), 1))]).T
  pts_cam = T_world2cam @ pts_homo
  z_cam = pts_cam[2, :]
  u = np.zeros_like(z_cam)
  v = np.zeros_like(z_cam)
  valid = z_cam > 0
  u[valid] = (pts_cam[0, valid] / z_cam[valid]) * K[0, 0] + K[0, 2]
  v[valid] = (pts_cam[1, valid] / z_cam[valid]) * K[1, 1] + K[1, 2]
  return u, v, z_cam


def unproject_to_3d(depth, color_img, K_mat, T_cam2world=None, min_depth=0.0, max_depth=1.5):
  mask = (depth > min_depth) & (depth < max_depth)
  v, u = np.where(mask)
  if T_cam2world is None:
    T_cam2world = np.eye(4)
  pts_world = unproject_points(u, v, depth[mask], K_mat, T_cam2world)
  return pts_world, color_img[mask]


def make_4x4(vec_6d):
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


def make_T(delta, device):
  rot = axis_angle_to_matrix(delta[:3])
  t = delta[3:].unsqueeze(1)
  T_top = torch.cat([rot, t], dim=1)
  T_bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
  return torch.cat([T_top, T_bottom], dim=0)
