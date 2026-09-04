import hashlib
import importlib.util
import os
import xml.etree.ElementTree as ET

import numpy as np
import pybullet
import pybullet_data
import torch
import torch.nn.functional as F


def _hidden_on_arm(name):
  return "hand" in name or "finger" in name


def _hidden_on_ghost(name):
  return "panda_link" in name


def _load_egl():
  cuda_pin = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
  if "EGL_VISIBLE_DEVICES" not in os.environ and cuda_pin.isdigit():
    os.environ["EGL_VISIBLE_DEVICES"] = cuda_pin

  spec = importlib.util.find_spec("eglRenderer")
  if spec is None:
    return False
  return pybullet.loadPlugin(spec.origin, "_eglRendererPlugin") >= 0


def _trimmed_urdf(src, hidden):
  tree = ET.parse(src)
  stripped = []
  for link in tree.getroot().findall("link"):
    if hidden(link.get("name", "")):
      gone = link.findall("visual") + link.findall("collision")
      for element in gone:
        link.remove(element)
      if gone:
        stripped.append(link.get("name"))
  tag = hashlib.md5((src + repr(stripped)).encode()).hexdigest()[:8]
  out = os.path.join(os.path.dirname(src), f"_trimmed_{tag}.urdf")
  if not os.path.exists(out):
    tree.write(out)
  return out


class PyBulletRenderer:
  def __init__(self, ghost_urdf, gpu=False):

    if pybullet.isConnected():
      pybullet.disconnect()
    pybullet.connect(pybullet.DIRECT)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

    self.gpu = bool(gpu) and _load_egl()
    self.renderer = pybullet.ER_BULLET_HARDWARE_OPENGL if self.gpu else pybullet.ER_TINY_RENDERER

    robot_urdf = os.path.join(pybullet_data.getDataPath(), "franka_panda", "panda.urdf")
    if self.gpu:
      robot_urdf = _trimmed_urdf(robot_urdf, _hidden_on_arm)
    self.robot_id = pybullet.loadURDF(robot_urdf, useFixedBase=True)
    self.arm_joints = [
      i
      for i in range(pybullet.getNumJoints(self.robot_id))
      if "panda_joint" in pybullet.getJointInfo(self.robot_id, i)[1].decode()
      and pybullet.getJointInfo(self.robot_id, i)[2] != pybullet.JOINT_FIXED
    ]

    self.hidden_robot_links = []
    for i in range(-1, pybullet.getNumJoints(self.robot_id)):
      name = (
        pybullet.getBodyInfo(self.robot_id)[0].decode()
        if i == -1
        else pybullet.getJointInfo(self.robot_id, i)[12].decode()
      )
      if _hidden_on_arm(name):
        if not self.gpu:
          pybullet.changeVisualShape(self.robot_id, i, rgbaColor=[0, 0, 0, 0])
        self.hidden_robot_links.append(i)

    if self.gpu:
      ghost_urdf = _trimmed_urdf(ghost_urdf, _hidden_on_ghost)
    self.ghost_id = pybullet.loadURDF(ghost_urdf, useFixedBase=True)
    self.ghost_arm_joints = [
      i
      for i in range(pybullet.getNumJoints(self.ghost_id))
      if "panda_joint" in pybullet.getJointInfo(self.ghost_id, i)[1].decode()
      and pybullet.getJointInfo(self.ghost_id, i)[2] != pybullet.JOINT_FIXED
    ]

    self.gripper_joints = []
    self.gripper_signs = []
    for i in range(pybullet.getNumJoints(self.ghost_id)):
      info = pybullet.getJointInfo(self.ghost_id, i)
      jname = info[1].decode()
      if info[2] != pybullet.JOINT_FIXED and "panda_joint" not in jname:
        self.gripper_joints.append(i)
        base_sign = -1 if "right" in jname else 1
        if any(k in jname for k in ["inner_finger", "follower", "finger_tip"]):
          self.gripper_signs.append(base_sign * -1)
        else:
          self.gripper_signs.append(base_sign)

    self.hidden_ghost_links = []
    for i in range(-1, pybullet.getNumJoints(self.ghost_id)):
      name = (
        pybullet.getBodyInfo(self.ghost_id)[0].decode()
        if i == -1
        else pybullet.getJointInfo(self.ghost_id, i)[12].decode()
      )
      if _hidden_on_ghost(name):
        if not self.gpu:
          pybullet.changeVisualShape(self.ghost_id, i, rgbaColor=[0, 0, 0, 0])
        self.hidden_ghost_links.append(i)

  def update_robot_pose(self, joint_angles, gripper_state=None, gripper_width_offset=0.08):
    for i, angle in zip(self.arm_joints, joint_angles):
      pybullet.resetJointState(self.robot_id, i, angle)
    for i, angle in zip(self.ghost_arm_joints, joint_angles):
      pybullet.resetJointState(self.ghost_id, i, angle)

    if gripper_state is not None and self.gripper_joints:
      raw_val = gripper_state[0] if isinstance(gripper_state, (list, np.ndarray)) else gripper_state
      raw_val = np.clip(raw_val, 0.0, 1.0)
      angle = (raw_val * 0.8028) - gripper_width_offset
      for i, sign in zip(self.gripper_joints, self.gripper_signs):
        pybullet.resetJointState(self.ghost_id, i, angle * sign)

  def _get_projection_matrix(self, K, w, h, near=0.01, far=10.0):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return [
      2.0 * fx / w,
      0.0,
      0.0,
      0.0,
      0.0,
      2.0 * fy / h,
      0.0,
      0.0,
      1.0 - 2.0 * cx / w,
      2.0 * cy / h - 1.0,
      (far + near) / (near - far),
      -1.0,
      0.0,
      0.0,
      2.0 * far * near / (near - far),
      0.0,
    ]

  def _render_raw(self, extrinsic, K, w, h):
    cam_pos = extrinsic[:3, 3]
    view_matrix = pybullet.computeViewMatrix(
      cam_pos.tolist(), (cam_pos + extrinsic[:3, 2]).tolist(), (-extrinsic[:3, 1]).tolist()
    )
    proj_matrix = self._get_projection_matrix(K, w, h)
    _, _, _, depth_buf, seg_buf = pybullet.getCameraImage(
      w,
      h,
      viewMatrix=view_matrix,
      projectionMatrix=proj_matrix,
      renderer=self.renderer,
      flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    )
    return depth_buf, seg_buf

  def render_depth(self, extrinsic, K, w, h):
    depth_buf, _ = self._render_raw(extrinsic, K, w, h)
    metric = 0.1 / (10.0 - 9.99 * np.reshape(depth_buf, (h, w)))
    return np.where(metric < 9.9, metric, 0.0)

  def render_mask(self, extrinsic, K, w, h):
    _, seg_buf = self._render_raw(extrinsic, K, w, h)
    seg_array = np.reshape(seg_buf, (h, w)).astype(np.int32)
    obj_ids = seg_array & 0xFFFFFF
    link_ids = (seg_array >> 24) - 1
    valid_robot = (obj_ids == self.robot_id) & ~np.isin(link_ids, self.hidden_robot_links)
    valid_ghost = (obj_ids == self.ghost_id) & ~np.isin(link_ids, self.hidden_ghost_links)
    return valid_robot | valid_ghost

  def render_segmentation(self, extrinsic, K, w, h):
    depth_buf, seg_buf = self._render_raw(extrinsic, K, w, h)
    metric = 0.1 / (10.0 - 9.99 * np.reshape(depth_buf, (h, w)))
    metric = np.where(metric < 9.9, metric, 0.0)
    seg_array = np.reshape(seg_buf, (h, w)).astype(np.int32)
    obj_ids = seg_array & 0xFFFFFF
    link_ids = (seg_array >> 24) - 1
    return obj_ids, link_ids, metric


def get_foreground_robot_points(
  T_init, K, obs_depth, pb_renderer, device, max_pts=2000, min_pts=100
):
  h_img, w_img = obs_depth.shape
  render_d = pb_renderer.render_depth(T_init, K, w_img, h_img)

  v_r, u_r = np.where(render_d > 0)
  z_r = render_d[v_r, u_r]
  if len(z_r) < min_pts:
    return None

  P_cam_r = np.stack(
    [(u_r - K[0, 2]) * z_r / K[0, 0], (v_r - K[1, 2]) * z_r / K[1, 1], z_r, np.ones_like(z_r)]
  )
  pts_robot_world = (T_init @ P_cam_r)[:3, :].T

  idx = np.random.choice(len(pts_robot_world), max_pts, replace=(len(pts_robot_world) < max_pts))
  return torch.tensor(pts_robot_world[idx], dtype=torch.float32, device=device)


def get_foreground_gripper_points(T_cam_world, K, obs_depth, pb_renderer, device, max_pts=2000):
  h_img, w_img = obs_depth.shape

  cam_pos = T_cam_world[:3, 3]
  target_pos = T_cam_world[:3, 3] + T_cam_world[:3, 2]
  view_matrix = pybullet.computeViewMatrix(
    cam_pos.tolist(), target_pos.tolist(), (-T_cam_world[:3, 1]).tolist()
  )
  proj_matrix = pb_renderer._get_projection_matrix(K, w_img, h_img)

  _, _, _, depth_buffer, seg_buffer = pybullet.getCameraImage(
    w_img,
    h_img,
    viewMatrix=view_matrix,
    projectionMatrix=proj_matrix,
    renderer=pb_renderer.renderer,
    flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
  )

  metric_depth = 0.1 / (10.0 - 9.99 * np.reshape(depth_buffer, (h_img, w_img)))
  seg_array = np.reshape(seg_buffer, (h_img, w_img)).astype(np.int32)
  obj_ids = seg_array & 0xFFFFFF
  valid_ghost = obj_ids == pb_renderer.ghost_id

  v_r, u_r = np.where((metric_depth < 9.9) & valid_ghost)
  z_r = metric_depth[v_r, u_r]
  if len(z_r) < 100:
    return None

  P_cam_r = np.stack(
    [(u_r - K[0, 2]) * z_r / K[0, 0], (v_r - K[1, 2]) * z_r / K[1, 1], z_r, np.ones_like(z_r)]
  )

  idx = np.random.choice(len(z_r), max_pts, replace=(len(z_r) < max_pts))
  return P_cam_r[:, idx]


def compute_robot_loss_batched(batch_X, T_opt, K, batch_obs):
  B, _, h_img, w_img = batch_obs.shape

  P_c = (batch_X - T_opt[:3, 3]) @ T_opt[:3, :3]
  Z_pred = P_c[..., 2]

  u = K[0, 0] * P_c[..., 0] / Z_pred + K[0, 2]
  v = K[1, 1] * P_c[..., 1] / Z_pred + K[1, 2]

  grid = torch.stack([(u / (w_img - 1)) * 2 - 1, (v / (h_img - 1)) * 2 - 1], dim=-1).unsqueeze(1)

  Z_obs_raw = (
    F.grid_sample(batch_obs, grid, mode='bilinear', padding_mode='border', align_corners=True)
    .squeeze(1)
    .squeeze(1)
  )

  valid_mask = (
    (Z_pred > 0.0)
    & (Z_pred < 1.5)
    & (Z_obs_raw > 0.0)
    & (Z_obs_raw < 1.5)
    & (u >= 0)
    & (u < w_img - 1)
    & (v >= 0)
    & (v < h_img - 1)
  )

  diff = torch.abs(Z_obs_raw[valid_mask] - Z_pred[valid_mask])
  return torch.nan_to_num(diff.mean(), nan=0.0)


def compute_wrist_loss_batched(batch_P_ee, T_cam_ee_opt, K, batch_obs):
  B, _, h_img, w_img = batch_obs.shape

  T_ee_cam = torch.linalg.inv(T_cam_ee_opt)
  P_c = batch_P_ee @ T_ee_cam[:3, :3].T + T_ee_cam[:3, 3]
  Z_pred = P_c[..., 2]

  u = K[0, 0] * P_c[..., 0] / Z_pred + K[0, 2]
  v = K[1, 1] * P_c[..., 1] / Z_pred + K[1, 2]

  grid = torch.stack([(u / (w_img - 1)) * 2 - 1, (v / (h_img - 1)) * 2 - 1], dim=-1).unsqueeze(1)

  Z_obs_raw = (
    F.grid_sample(batch_obs, grid, mode='bilinear', padding_mode='border', align_corners=True)
    .squeeze(1)
    .squeeze(1)
  )

  valid_mask = (
    (Z_pred > 0.0)
    & (Z_pred < 1.5)
    & (Z_obs_raw > 0.0)
    & (Z_obs_raw < 1.5)
    & (u >= 0)
    & (u < w_img - 1)
    & (v >= 0)
    & (v < h_img - 1)
  )

  diff = torch.abs(Z_obs_raw[valid_mask] - Z_pred[valid_mask])
  return torch.nan_to_num(diff.mean(), nan=0.0)
