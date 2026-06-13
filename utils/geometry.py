"""3D vision geometry primitives for the DROID pipeline.

Provides disparity decoding, 3D unprojection, 2D projection, and rigid
transform helpers in both NumPy and PyTorch variants.

Collected from pipeline.ipynb and 2026_06_11_DROID_dataset_v60.ipynb.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


# ===========================================================================
# Disparity ↔ Depth
# ===========================================================================

def decode_disparity_np(disp, fx, baseline):
  """Convert raw stereo disparity to metric depth Z (NumPy)."""
  z = np.zeros_like(disp)
  valid_mask = disp > 0
  z[valid_mask] = (fx * baseline) / disp[valid_mask]
  return z


def decode_disparity_pt(disp, fx, baseline):
  """Convert raw stereo disparity to metric depth Z (PyTorch)."""
  z = torch.zeros_like(disp)
  valid_mask = disp > 0
  z[valid_mask] = (fx * baseline) / disp[valid_mask]
  return z


# ===========================================================================
# 3D Unprojection (pixel + depth → world)
# ===========================================================================

def unproject_points_np(u, v, z, K, T_cam2world=None):
  """Unproject 2D pixel coords + depth to 3D points (NumPy).

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


def unproject_points_pt(u, v, z, K, T_cam2world=None):
  """Unproject 2D pixel coords + depth to 3D points (PyTorch, differentiable).

  Same interface as `unproject_points_np` but preserves gradient flow.
  """
  x_cam = (u - K[0, 2]) * z / K[0, 0]
  y_cam = (v - K[1, 2]) * z / K[1, 1]
  pts_cam = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=0)
  if T_cam2world is None:
    return pts_cam[:3, :].T
  return (T_cam2world @ pts_cam)[:3, :].T


# ===========================================================================
# 2D Projection (world → pixel)
# ===========================================================================

def project_points_np(pts_world, K, T_cam2world):
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
  u = np.zeros_like(pts_cam[0, :])
  v = np.zeros_like(pts_cam[1, :])
  valid_mask = z_cam > 0
  u[valid_mask] = (pts_cam[0, valid_mask] / z_cam[valid_mask]) * K[0, 0] + K[0, 2]
  v[valid_mask] = (pts_cam[1, valid_mask] / z_cam[valid_mask]) * K[1, 1] + K[1, 2]
  return u, v, z_cam


def project_points_pt(pts_world, K, T_cam2world):
  """Project 3D world points to 2D pixel coordinates (PyTorch, differentiable).

  Same interface as `project_points_np` but preserves gradient flow.
  """
  T_world2cam = torch.linalg.inv(T_cam2world)
  pts_homo = torch.cat(
      [pts_world, torch.ones((len(pts_world), 1), device=pts_world.device)],
      dim=1).T
  pts_cam = T_world2cam @ pts_homo
  z_cam = pts_cam[2, :]
  u = torch.zeros_like(pts_cam[0, :])
  v = torch.zeros_like(pts_cam[1, :])
  valid_mask = z_cam > 0
  u[valid_mask] = (pts_cam[0, valid_mask] / z_cam[valid_mask]) * K[0, 0] + K[0, 2]
  v[valid_mask] = (pts_cam[1, valid_mask] / z_cam[valid_mask]) * K[1, 1] + K[1, 2]
  return u, v, z_cam


# ===========================================================================
# Convenience: depth map → colored 3D point cloud
# ===========================================================================

def unproject_to_3d(depth, color_img, K_mat, T_cam2world=None,
                    min_depth=0., max_depth=1.5):
  """Unproject a depth image to a colored 3D point cloud.

  Args:
    depth: (H, W) metric depth map
    color_img: (H, W, 3) RGB image
    K_mat: 3x3 intrinsic matrix
    T_cam2world: optional 4x4 extrinsic matrix
    min_depth, max_depth: physical depth range filter

  Returns:
    pts_world: (M, 3) valid world points
    colors: (M, 3) corresponding RGB values
  """
  mask = (depth > min_depth) & (depth < max_depth)
  v, u = np.where(mask)
  if T_cam2world is None:
    T_cam2world = np.eye(4)
  pts_world = unproject_points_np(u, v, depth[mask], K_mat, T_cam2world)
  return pts_world, color_img[mask]


# ===========================================================================
# Rigid Transform Helpers
# ===========================================================================

def make_4x4(vec_6d):
  """Convert a 6DoF vector [x, y, z, rx, ry, rz] to a 4x4 homogeneous matrix.

  Uses XYZ Euler angle convention via scipy.
  """
  transform = np.eye(4)
  transform[:3, :3] = R.from_euler('xyz', vec_6d[3:]).as_matrix()
  transform[:3, 3] = vec_6d[:3]
  return transform
