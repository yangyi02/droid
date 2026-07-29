"""3D vision geometry primitives for the DROID pipeline.

Provides disparity decoding, 3D unprojection, 2D projection, rigid
transform helpers, and differentiable rotation formulas.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


# ===========================================================================
# Disparity → Depth
# ===========================================================================

def decode_disparity(disp, fx, baseline):
  """Convert raw stereo disparity to metric depth Z (NumPy)."""
  z = np.zeros_like(disp)
  valid_mask = disp > 0
  z[valid_mask] = (fx * baseline) / disp[valid_mask]
  return z


# ===========================================================================
# 3D Unprojection (pixel + depth → world)
# ===========================================================================

def unproject_points(u, v, z, K, T_cam2world=None):
  """Back-project 2D pixel coordinates to 3D world points (NumPy).

  Args:
    u, v: pixel x, y coordinates (N,)
    z: depth values (N,)
    K: 3x3 intrinsic matrix
    T_cam2world: optional 4x4 camera-to-world (extrinsic) matrix

  Returns:
    pts: (N, 3) world coordinates (or camera coords if T_cam2world is None)
  """
  x_cam = (u - K[0, 2]) * z / K[0, 0]
  y_cam = (v - K[1, 2]) * z / K[1, 1]
  pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=0)
  if T_cam2world is None:
    return pts_cam[:3, :].T
  return (T_cam2world @ pts_cam)[:3, :].T


# ===========================================================================
# 2D Projection (world → pixel)
# ===========================================================================

def project_points(pts_world, K, T_cam2world):
  """Project 3D world points to 2D pixel coordinates (NumPy).

  Args:
    pts_world: (N, 3) world coordinates
    K: 3x3 intrinsic matrix
    T_cam2world: 4x4 camera-to-world matrix

  Returns:
    u, v: pixel coordinates (N,)
    z_cam: depth in camera frame (N,)
  """
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



# ===========================================================================
# Convenience: depth map → colored 3D point cloud
# ===========================================================================

def unproject_to_3d(depth, color_img, K_mat, T_cam2world=None,
                    min_depth=0., max_depth=1.5):
  """Unproject a depth image to a colored 3D point cloud."""
  mask = (depth > min_depth) & (depth < max_depth)
  v, u = np.where(mask)
  if T_cam2world is None:
    T_cam2world = np.eye(4)
  pts_world = unproject_points(u, v, depth[mask], K_mat, T_cam2world)
  return pts_world, color_img[mask]


# ===========================================================================
# Rigid Transform Helpers
# ===========================================================================

def make_4x4(vec_6d):
  """Convert a 6DoF vector [x, y, z, rx, ry, rz] to a 4x4 homogeneous matrix."""
  transform = np.eye(4)
  transform[:3, :3] = R.from_euler('xyz', vec_6d[3:]).as_matrix()
  transform[:3, 3] = vec_6d[:3]
  return transform


def axis_angle_to_matrix(rot_vec):
  """Differentiable Rodrigues rotation formula with small-angle Taylor fallback."""
  theta2 = torch.sum(rot_vec ** 2)
  theta = torch.sqrt(theta2 + 1e-16)
  k = rot_vec / theta

  K = torch.zeros((3, 3), device=rot_vec.device)
  K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]

  R_exact = torch.eye(3, device=rot_vec.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.mm(K, K)

  K_approx = torch.zeros_like(K)
  K_approx[0, 1], K_approx[0, 2], K_approx[1, 0], K_approx[1, 2], K_approx[2, 0], K_approx[2, 1] = -rot_vec[2], rot_vec[1], rot_vec[2], -rot_vec[0], -rot_vec[1], rot_vec[0]

  return torch.where(theta2 < 1e-8, torch.eye(3, device=rot_vec.device) + K_approx, R_exact)


def make_T(delta, device):
  """Build incremental 4x4 from a 6D parameter vector."""
  rot = axis_angle_to_matrix(delta[:3])
  t = delta[3:].unsqueeze(1)
  T_top = torch.cat([rot, t], dim=1)
  T_bottom = torch.tensor([[0., 0., 0., 1.]], device=device, dtype=torch.float32)
  return torch.cat([T_top, T_bottom], dim=0)
