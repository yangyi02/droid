import importlib.util
import os

import numpy as np
import pybullet
import torch
import torch.nn.functional as F

NEAR_PLANE = 0.01
FAR_PLANE = 10.0
GRIPPER_ANGLE_SCALE = 0.8028
GRIPPER_WIDTH_OFFSET = 0.08


def metric_depth(depth_buf, height, width):
  buf = np.reshape(depth_buf, (height, width))
  return (FAR_PLANE * NEAR_PLANE) / (FAR_PLANE - buf * (FAR_PLANE - NEAR_PLANE))


def _load_egl():
  cuda_pin = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
  if "EGL_VISIBLE_DEVICES" not in os.environ and cuda_pin.isdigit():
    os.environ["EGL_VISIBLE_DEVICES"] = cuda_pin

  spec = importlib.util.find_spec("eglRenderer")
  if spec is None:
    return False
  return pybullet.loadPlugin(spec.origin, "_eglRendererPlugin") >= 0


class PyBulletRenderer:
  def __init__(self, urdf, gpu=False):
    if pybullet.isConnected():
      pybullet.disconnect()
    pybullet.connect(pybullet.DIRECT)

    if not pybullet.isNumpyEnabled():
      raise RuntimeError(
        "pybullet was built without NumPy: getCameraImage is 2x slower. Rerun bash setup.sh"
      )

    self.gpu = bool(gpu) and _load_egl()
    if gpu and not self.gpu:
      raise RuntimeError("EGL requested but the plugin did not load. --config.render.gpu=False")
    self.render_mode = pybullet.ER_BULLET_HARDWARE_OPENGL if self.gpu else pybullet.ER_TINY_RENDERER

    self.robot_id = pybullet.loadURDF(
      urdf, useFixedBase=True, flags=pybullet.URDF_IGNORE_COLLISION_SHAPES
    )

    self.arm_joints = []
    self.gripper_joints = []
    self.gripper_signs = []
    self.gripper_links = []
    for i in range(pybullet.getNumJoints(self.robot_id)):
      info = pybullet.getJointInfo(self.robot_id, i)
      joint_name = info[1].decode()
      if "panda_link" not in info[12].decode():
        self.gripper_links.append(i)
      if info[2] == pybullet.JOINT_FIXED:
        continue
      if "panda_joint" in joint_name:
        self.arm_joints.append(i)
        continue
      self.gripper_joints.append(i)
      sign = -1 if "right" in joint_name else 1
      if any(k in joint_name for k in ["inner_finger", "follower", "finger_tip"]):
        sign = -sign
      self.gripper_signs.append(sign)

  def update_robot_pose(self, joint_angles, gripper_state=None):
    for i, angle in zip(self.arm_joints, joint_angles):
      pybullet.resetJointState(self.robot_id, i, angle)

    if gripper_state is not None and self.gripper_joints:
      raw_val = gripper_state[0] if isinstance(gripper_state, (list, np.ndarray)) else gripper_state
      raw_val = np.clip(raw_val, 0.0, 1.0)
      angle = (raw_val * GRIPPER_ANGLE_SCALE) - GRIPPER_WIDTH_OFFSET
      for i, sign in zip(self.gripper_joints, self.gripper_signs):
        pybullet.resetJointState(self.robot_id, i, angle * sign)

  def _get_projection_matrix(self, K, width, height):
    near, far = NEAR_PLANE, FAR_PLANE
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return [
      2.0 * fx / width,
      0.0,
      0.0,
      0.0,
      0.0,
      2.0 * fy / height,
      0.0,
      0.0,
      1.0 - 2.0 * cx / width,
      2.0 * cy / height - 1.0,
      (far + near) / (near - far),
      -1.0,
      0.0,
      0.0,
      2.0 * far * near / (near - far),
      0.0,
    ]

  def _render_raw(self, T_cam2world, K, width, height):
    cam_pos = T_cam2world[:3, 3]
    view_matrix = pybullet.computeViewMatrix(
      cam_pos.tolist(), (cam_pos + T_cam2world[:3, 2]).tolist(), (-T_cam2world[:3, 1]).tolist()
    )
    proj_matrix = self._get_projection_matrix(K, width, height)
    _, _, _, depth_buf, seg_buf = pybullet.getCameraImage(
      width,
      height,
      viewMatrix=view_matrix,
      projectionMatrix=proj_matrix,
      renderer=self.render_mode,
      flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    )
    return depth_buf, seg_buf

  def render_depth(self, T_cam2world, K, width, height):
    depth_buf, _ = self._render_raw(T_cam2world, K, width, height)
    metric = metric_depth(depth_buf, height, width)
    return np.where(metric < FAR_PLANE * 0.99, metric, 0.0)

  def render_mask(self, T_cam2world, K, width, height):
    _, seg_buf = self._render_raw(T_cam2world, K, width, height)
    seg_array = np.reshape(seg_buf, (height, width)).astype(np.int32)
    return (seg_array & 0xFFFFFF) == self.robot_id

  def render_segmentation(self, T_cam2world, K, width, height):
    depth_buf, seg_buf = self._render_raw(T_cam2world, K, width, height)
    metric = metric_depth(depth_buf, height, width)
    metric = np.where(metric < FAR_PLANE * 0.99, metric, 0.0)
    seg_array = np.reshape(seg_buf, (height, width)).astype(np.int32)
    obj_ids = seg_array & 0xFFFFFF
    link_ids = (seg_array >> 24) - 1
    return obj_ids, link_ids, metric


def get_foreground_robot_points(T_cam2world, K, depth, pb_renderer, device, n_points=2000):
  height, width = depth.shape
  render_d = pb_renderer.render_depth(T_cam2world, K, width, height)

  v_r, u_r = np.where(render_d > 0)
  if len(u_r) < n_points:
    return None

  idx = np.random.choice(len(u_r), n_points, replace=False)
  v_r, u_r = v_r[idx], u_r[idx]
  z_r = render_d[v_r, u_r]

  points_cam = np.stack(
    [(u_r - K[0, 2]) * z_r / K[0, 0], (v_r - K[1, 2]) * z_r / K[1, 1], z_r, np.ones_like(z_r)]
  )
  return torch.tensor((T_cam2world @ points_cam)[:3, :].T, dtype=torch.float32, device=device)


def get_foreground_gripper_points(T_cam2world, K, depth, pb_renderer, device, n_points=2000):
  height, width = depth.shape

  cam_pos = T_cam2world[:3, 3]
  target_pos = T_cam2world[:3, 3] + T_cam2world[:3, 2]
  view_matrix = pybullet.computeViewMatrix(
    cam_pos.tolist(), target_pos.tolist(), (-T_cam2world[:3, 1]).tolist()
  )
  proj_matrix = pb_renderer._get_projection_matrix(K, width, height)

  _, _, _, depth_buffer, seg_buffer = pybullet.getCameraImage(
    width,
    height,
    viewMatrix=view_matrix,
    projectionMatrix=proj_matrix,
    renderer=pb_renderer.render_mode,
    flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
  )

  metric = metric_depth(depth_buffer, height, width)
  seg_array = np.reshape(seg_buffer, (height, width)).astype(np.int32)
  link_ids = (seg_array >> 24) - 1
  valid_gripper = np.isin(link_ids, pb_renderer.gripper_links)

  v_r, u_r = np.where((metric < FAR_PLANE * 0.99) & valid_gripper)
  z_r = metric[v_r, u_r]
  if len(z_r) < 100:
    return None

  points_cam = np.stack(
    [(u_r - K[0, 2]) * z_r / K[0, 0], (v_r - K[1, 2]) * z_r / K[1, 1], z_r, np.ones_like(z_r)]
  )

  idx = np.random.choice(len(z_r), n_points, replace=(len(z_r) < n_points))
  return points_cam[:, idx]


def depth_loss_batched(points, T_cam2world, K, depth_batch, max_depth=1.5):
  _, _, height, width = depth_batch.shape

  P_c = (points - T_cam2world[:3, 3]) @ T_cam2world[:3, :3]
  z_pred = P_c[..., 2]

  u = K[0, 0] * P_c[..., 0] / z_pred + K[0, 2]
  v = K[1, 1] * P_c[..., 1] / z_pred + K[1, 2]

  grid = torch.stack([(u / (width - 1)) * 2 - 1, (v / (height - 1)) * 2 - 1], dim=-1).unsqueeze(1)

  z_obs = (
    F.grid_sample(depth_batch, grid, mode='bilinear', padding_mode='border', align_corners=True)
    .squeeze(1)
    .squeeze(1)
  )

  valid = (
    (z_pred > 0.0)
    & (z_pred < max_depth)
    & (z_obs > 0.0)
    & (z_obs < max_depth)
    & (u >= 0)
    & (u < width - 1)
    & (v >= 0)
    & (v < height - 1)
  )

  return torch.nan_to_num(torch.abs(z_obs[valid] - z_pred[valid]).mean(), nan=0.0)
