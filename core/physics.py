"""Robot rendering for the DROID pipeline.

PyBulletRenderer (pybullet): depth + segmentation renderer for the robot.
  compute_extrinsics optimises camera poses against the point clouds it
  produces, compute_tracks masks with them, compute_metrics scores with them.

It replaced a yourdfpy renderer that sampled the CAD meshes and approximated
visibility with front-face culling; the rasteriser gets visibility right for
free, which is worth most on the wrist camera where the gripper occludes
itself. notebooks/pybullet_gpu_pipeline_validation.ipynb has the comparison.
"""

import hashlib
import importlib.util
import os
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p

import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_URDF = os.path.join(
    _SCRIPT_DIR, "assets", "franka_description",
    "franka_panda_robotiq_2f85_og.urdf",
)


# ===========================================================================
# PyBulletRenderer — for compute_tracks (segmentation + masking)
# ===========================================================================


def _hidden_on_arm(name):
  """Arm-body links the ghost body draws instead (the Robotiq hand)."""
  return "hand" in name or "finger" in name


def _hidden_on_ghost(name):
  """Ghost-body links the arm body draws instead (the whole Panda arm)."""
  return "panda_link" in name


def _load_egl():
  """Load pybullet's GPU rasteriser into the current connection.

  Two traps. The module pybullet ships is `eglRenderer`; `_eglRendererPlugin`
  is only the name it registers under, so asking find_spec for the latter
  silently returns None. And the plugin only sees geometry registered after it
  loads, so this has to run before the first loadURDF -- otherwise the renders
  come back empty, very fast, which reads like a speedup.
  """
  # The plugin takes its device from EGL_VISIBLE_DEVICES and ignores
  # CUDA_VISIBLE_DEVICES, so under run_parallel.sh -- which pins each worker
  # with CUDA_VISIBLE_DEVICES -- every worker's EGL context would open on GPU 0
  # while its tensors live on the pinned card. Mirror the pin, unless the
  # caller has already chosen. Only a single-index pin is unambiguous; a list
  # or a UUID is left alone.
  cuda_pin = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
  if "EGL_VISIBLE_DEVICES" not in os.environ and cuda_pin.isdigit():
    os.environ["EGL_VISIBLE_DEVICES"] = cuda_pin

  spec = importlib.util.find_spec("eglRenderer")
  if spec is None:
    return False
  try:
    return p.loadPlugin(spec.origin, "_eglRendererPlugin") >= 0
  except p.error:
    return False


def _trimmed_urdf(src, hidden):
  """Copy `src` with the geometry of `hidden(link_name)` links removed.

  Both <visual> and <collision> have to go: a link with no visual is drawn
  from its collision mesh instead, which is coarser and slightly fatter, so
  dropping one alone leaves the link on screen in the wrong shape.

  The copy is written beside the original, which keeps every relative and
  package:// mesh path resolving as it did. Gitignored as _trimmed_*.urdf.
  """
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
  """Dual-body PyBullet renderer: Franka arm + Robotiq gripper.

  The 'robot' body renders the arm links (hand/finger hidden).
  The 'ghost' body renders the gripper (arm links hidden).
  Together they form the complete visual model.

  gpu=True renders on the EGL rasteriser instead of the CPU one: ~4.4x faster
  at 1280x720 and up to ~35x at the small resolutions a point-cloud pass wants.
  It also changes how the two bodies hide their unwanted links -- alpha=0 means
  nothing to EGL, so the geometry is stripped from the URDF instead. Renders
  are not bit-identical across the two (mask IoU 0.98, agreeing to under a
  millimetre away from silhouettes), so it is off by default. See
  notebooks/pybullet_egl_mask_benchmark.ipynb.
  """

  def __init__(self, ghost_urdf=None, gpu=False):
    import pybullet_data

    # A pybullet built without NumPy support marshals every pixel of
    # getCameraImage into a Python tuple before returning it: ~336 ms per
    # 1280x720 render here instead of ~25 ms. It degrades silently, and pip
    # produces such a build by default (its isolated build env has no numpy),
    # so a venv that predates setup.sh's rebuild step still looks fine.
    if not p.isNumpyEnabled():
      raise RuntimeError(
          'pybullet was built without NumPy support -- rendering would be '
          '~15x slower. Rebuild it in place:\n'
          '  pip install --force-reinstall --no-deps --no-binary pybullet \\\n'
          '      --no-build-isolation --no-cache-dir pybullet')

    if ghost_urdf is None:
      ghost_urdf = _DEFAULT_URDF

    if p.isConnected():
      p.disconnect()
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Before the first loadURDF, or the plugin sees no geometry. Falls back to
    # the CPU rasteriser when there is no GPU, rather than rendering nothing.
    self.gpu = bool(gpu) and _load_egl()
    self.renderer = (p.ER_BULLET_HARDWARE_OPENGL if self.gpu
                     else p.ER_TINY_RENDERER)

    # Real body: thin arm (hand/finger hidden)
    robot_urdf = os.path.join(pybullet_data.getDataPath(),
                              "franka_panda", "panda.urdf")
    if self.gpu:
      robot_urdf = _trimmed_urdf(robot_urdf, _hidden_on_arm)
    self.robot_id = p.loadURDF(robot_urdf, useFixedBase=True)
    self.arm_joints = [
        i for i in range(p.getNumJoints(self.robot_id))
        if "panda_joint" in p.getJointInfo(self.robot_id, i)[1].decode()
        and p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED
    ]

    self.hidden_robot_links = []
    for i in range(-1, p.getNumJoints(self.robot_id)):
      name = (p.getBodyInfo(self.robot_id)[0].decode() if i == -1
              else p.getJointInfo(self.robot_id, i)[12].decode())
      if _hidden_on_arm(name):
        if not self.gpu:                    # on gpu the geometry is gone
          p.changeVisualShape(self.robot_id, i, rgbaColor=[0, 0, 0, 0])
        self.hidden_robot_links.append(i)

    # Ghost body: Robotiq gripper (arm links hidden)
    if self.gpu:
      ghost_urdf = _trimmed_urdf(ghost_urdf, _hidden_on_ghost)
    self.ghost_id = p.loadURDF(ghost_urdf, useFixedBase=True)
    self.ghost_arm_joints = [
        i for i in range(p.getNumJoints(self.ghost_id))
        if "panda_joint" in p.getJointInfo(self.ghost_id, i)[1].decode()
        and p.getJointInfo(self.ghost_id, i)[2] != p.JOINT_FIXED
    ]

    self.gripper_joints = []
    self.gripper_signs = []
    for i in range(p.getNumJoints(self.ghost_id)):
      info = p.getJointInfo(self.ghost_id, i)
      jname = info[1].decode()
      if info[2] != p.JOINT_FIXED and "panda_joint" not in jname:
        self.gripper_joints.append(i)
        base_sign = -1 if "right" in jname else 1
        if any(k in jname for k in ["inner_finger", "follower", "finger_tip"]):
          self.gripper_signs.append(base_sign * -1)
        else:
          self.gripper_signs.append(base_sign)

    self.hidden_ghost_links = []
    for i in range(-1, p.getNumJoints(self.ghost_id)):
      name = (p.getBodyInfo(self.ghost_id)[0].decode() if i == -1
              else p.getJointInfo(self.ghost_id, i)[12].decode())
      if _hidden_on_ghost(name):
        if not self.gpu:
          p.changeVisualShape(self.ghost_id, i, rgbaColor=[0, 0, 0, 0])
        self.hidden_ghost_links.append(i)

  def update_robot_pose(self, joint_angles, gripper_state=None,
                        gripper_width_offset=0.08):
    """Synchronize both bodies to given joint configuration."""
    for i, angle in zip(self.arm_joints, joint_angles):
      p.resetJointState(self.robot_id, i, angle)
    for i, angle in zip(self.ghost_arm_joints, joint_angles):
      p.resetJointState(self.ghost_id, i, angle)

    if gripper_state is not None and self.gripper_joints:
      raw_val = (gripper_state[0]
                 if isinstance(gripper_state, (list, np.ndarray))
                 else gripper_state)
      raw_val = np.clip(raw_val, 0.0, 1.0)
      angle = (raw_val * 0.8028) - gripper_width_offset
      for i, sign in zip(self.gripper_joints, self.gripper_signs):
        p.resetJointState(self.ghost_id, i, angle * sign)

  def _get_projection_matrix(self, K, w, h, near=0.01, far=10.0):
    """Convert intrinsic matrix to OpenGL projection matrix."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return [
        2.0 * fx / w, 0.0, 0.0, 0.0,
        0.0, 2.0 * fy / h, 0.0, 0.0,
        1.0 - 2.0 * cx / w, 2.0 * cy / h - 1.0,
        (far + near) / (near - far), -1.0,
        0.0, 0.0, 2.0 * far * near / (near - far), 0.0,
    ]

  def _render_raw(self, extrinsic, K, w, h):
    """Common rendering call returning depth_buffer + seg_buffer."""
    cam_pos = extrinsic[:3, 3]
    view_matrix = p.computeViewMatrix(
        cam_pos.tolist(), (cam_pos + extrinsic[:3, 2]).tolist(),
        (-extrinsic[:3, 1]).tolist())
    proj_matrix = self._get_projection_matrix(K, w, h)
    # Spelled out rather than left to the ER_BULLET_HARDWARE_OPENGL fallback,
    # which silently lands on the CPU rasteriser when no EGL plugin is loaded.
    # Which of the two is in use is decided once, in __init__.
    _, _, _, depth_buf, seg_buf = p.getCameraImage(
        w, h, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=self.renderer,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
    return depth_buf, seg_buf

  def render_depth(self, extrinsic, K, w, h):
    """Render physical depth map from camera pose."""
    depth_buf, _ = self._render_raw(extrinsic, K, w, h)
    metric = 0.1 / (10.0 - 9.99 * np.reshape(depth_buf, (h, w)))
    return np.where(metric < 9.9, metric, 0.0)

  def render_mask(self, extrinsic, K, w, h):
    """Render binary robot mask."""
    _, seg_buf = self._render_raw(extrinsic, K, w, h)
    seg_array = np.reshape(seg_buf, (h, w)).astype(np.int32)
    obj_ids = seg_array & 0xFFFFFF
    link_ids = (seg_array >> 24) - 1
    valid_robot = ((obj_ids == self.robot_id) &
                   ~np.isin(link_ids, self.hidden_robot_links))
    valid_ghost = ((obj_ids == self.ghost_id) &
                   ~np.isin(link_ids, self.hidden_ghost_links))
    return valid_robot | valid_ghost

  def render_segmentation(self, extrinsic, K, w, h):
    """Render full segmentation: (obj_ids, link_ids, metric_depth)."""
    depth_buf, seg_buf = self._render_raw(extrinsic, K, w, h)
    metric = 0.1 / (10.0 - 9.99 * np.reshape(depth_buf, (h, w)))
    metric = np.where(metric < 9.9, metric, 0.0)
    seg_array = np.reshape(seg_buf, (h, w)).astype(np.int32)
    obj_ids = seg_array & 0xFFFFFF
    link_ids = (seg_array >> 24) - 1
    return obj_ids, link_ids, metric


# ===========================================================================
# Robot point clouds and depth losses — shared by compute_extrinsics
# (optimization) and compute_metrics (evaluation)
# ===========================================================================


def get_foreground_robot_points(T_init, K, obs_depth, pb_renderer, device,
                                max_pts=2000):
  """Extract robot point cloud via PyBullet depth rendering and reprojection.

  Renders the full robot body at the given extrinsic pose, selects pixels
  where the rendered depth is nonzero, and lifts them to world-frame 3D
  points.

  Args:
    T_init: (4, 4) NumPy camera-to-world extrinsic matrix.
    K: (3, 3) NumPy intrinsic matrix.
    obs_depth: (H, W) NumPy observed depth map (used only for resolution).
    pb_renderer: ``PyBulletRenderer`` instance (from ``core.physics``).
    device: torch device string or object.
    max_pts: Target number of output points.

  Returns:
    (max_pts, 3) float32 tensor on *device*, or ``None`` if too few pixels.
  """
  h_img, w_img = obs_depth.shape
  render_d = pb_renderer.render_depth(T_init, K, w_img, h_img)

  v_r, u_r = np.where(render_d > 0)
  z_r = render_d[v_r, u_r]
  if len(z_r) < max_pts:
    return None

  P_cam_r = np.stack([
      (u_r - K[0, 2]) * z_r / K[0, 0],
      (v_r - K[1, 2]) * z_r / K[1, 1],
      z_r,
      np.ones_like(z_r),
  ])
  pts_robot_world = (T_init @ P_cam_r)[:3, :].T

  idx = np.random.choice(
      len(pts_robot_world), max_pts,
      replace=(len(pts_robot_world) < max_pts),
  )
  return torch.tensor(
      pts_robot_world[idx], dtype=torch.float32, device=device,
  )


def get_foreground_gripper_points(T_cam_world, K, obs_depth, pb_renderer,
                                  device, max_pts=2000):
  """Extract gripper-only point cloud using PyBullet segmentation mask.

  Renders a full camera image and filters by the ghost body's segmentation
  ID so that only gripper pixels survive.  Returns **camera-frame**
  homogeneous coordinates (4, max_pts) for hand-eye optimization.

  Args:
    T_cam_world: (4, 4) NumPy camera-to-world extrinsic matrix.
    K: (3, 3) NumPy intrinsic matrix.
    obs_depth: (H, W) NumPy observed depth map (used only for resolution).
    pb_renderer: ``PyBulletRenderer`` instance.
    device: torch device string or object.
    max_pts: Target number of output points.

  Returns:
    (4, max_pts) NumPy float64 array of camera-frame homogeneous points,
    or ``None`` if fewer than 100 valid pixels.
  """
  h_img, w_img = obs_depth.shape

  cam_pos = T_cam_world[:3, 3]
  target_pos = T_cam_world[:3, 3] + T_cam_world[:3, 2]
  view_matrix = p.computeViewMatrix(
      cam_pos.tolist(), target_pos.tolist(), (-T_cam_world[:3, 1]).tolist(),
  )
  proj_matrix = pb_renderer._get_projection_matrix(K, w_img, h_img)

  _, _, _, depth_buffer, seg_buffer = p.getCameraImage(
      w_img, h_img,
      viewMatrix=view_matrix,
      projectionMatrix=proj_matrix,
      renderer=pb_renderer.renderer,
      flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
  )

  metric_depth = 0.1 / (10.0 - 9.99 * np.reshape(depth_buffer, (h_img, w_img)))
  seg_array = np.reshape(seg_buffer, (h_img, w_img)).astype(np.int32)
  obj_ids = seg_array & 0xFFFFFF
  valid_ghost = (obj_ids == pb_renderer.ghost_id)

  v_r, u_r = np.where((metric_depth < 9.9) & valid_ghost)
  z_r = metric_depth[v_r, u_r]
  if len(z_r) < 100:
    return None

  P_cam_r = np.stack([
      (u_r - K[0, 2]) * z_r / K[0, 0],
      (v_r - K[1, 2]) * z_r / K[1, 1],
      z_r,
      np.ones_like(z_r),
  ])

  idx = np.random.choice(len(z_r), max_pts, replace=(len(z_r) < max_pts))
  return P_cam_r[:, idx]


def compute_robot_loss_batched(batch_X, T_opt, K, batch_obs):
  """Depth re-projection loss for external cameras.

  Projects world-frame robot points into the camera using *T_opt*, samples
  observed depth via differentiable ``grid_sample``, and returns the mean
  absolute depth error.

  Unlike ``compute_robot_loss`` in ``compute_extrinsics.py``, this version
  does **not** use surface normals, front-face culling, or depth tolerance.

  Args:
    batch_X: (B, N, 3) world-frame robot points.
    T_opt: (4, 4) camera-to-world extrinsic (differentiable).
    K: (3, 3) intrinsic matrix (tensor).
    batch_obs: (B, 1, H, W) observed depth maps.

  Returns:
    Scalar loss tensor.
  """
  B, _, h_img, w_img = batch_obs.shape

  P_c = (batch_X - T_opt[:3, 3]) @ T_opt[:3, :3]
  Z_pred = P_c[..., 2]

  u = K[0, 0] * P_c[..., 0] / Z_pred + K[0, 2]
  v = K[1, 1] * P_c[..., 1] / Z_pred + K[1, 2]

  grid = torch.stack([
      (u / (w_img - 1)) * 2 - 1,
      (v / (h_img - 1)) * 2 - 1,
  ], dim=-1).unsqueeze(1)

  Z_obs_raw = F.grid_sample(
      batch_obs, grid, mode='bilinear', padding_mode='border',
      align_corners=True,
  ).squeeze(1).squeeze(1)

  valid_mask = (
      (Z_pred > 0.) & (Z_pred < 1.5) &
      (Z_obs_raw > 0.) & (Z_obs_raw < 1.5) &
      (u >= 0) & (u < w_img - 1) &
      (v >= 0) & (v < h_img - 1)
  )

  diff = torch.abs(Z_obs_raw[valid_mask] - Z_pred[valid_mask])
  return torch.nan_to_num(diff.mean(), nan=0.0)


def compute_wrist_loss_batched(batch_P_ee, T_cam_ee_opt, K, batch_obs):
  """Depth re-projection loss for the wrist camera.

  Points are anchored in the end-effector frame.  The function inverts
  ``T_cam_ee_opt`` to transform them back to the camera frame before
  projection and depth comparison.

  Args:
    batch_P_ee: (B, N, 3) points in end-effector frame.
    T_cam_ee_opt: (4, 4) wrist-cam-to-EE extrinsic (differentiable).
    K: (3, 3) intrinsic matrix (tensor).
    batch_obs: (B, 1, H, W) observed depth maps.

  Returns:
    Scalar loss tensor.
  """
  B, _, h_img, w_img = batch_obs.shape

  T_ee_cam = torch.linalg.inv(T_cam_ee_opt)
  P_c = batch_P_ee @ T_ee_cam[:3, :3].T + T_ee_cam[:3, 3]
  Z_pred = P_c[..., 2]

  u = K[0, 0] * P_c[..., 0] / Z_pred + K[0, 2]
  v = K[1, 1] * P_c[..., 1] / Z_pred + K[1, 2]

  grid = torch.stack([
      (u / (w_img - 1)) * 2 - 1,
      (v / (h_img - 1)) * 2 - 1,
  ], dim=-1).unsqueeze(1)

  Z_obs_raw = F.grid_sample(
      batch_obs, grid, mode='bilinear', padding_mode='border',
      align_corners=True,
  ).squeeze(1).squeeze(1)

  valid_mask = (
      (Z_pred > 0.) & (Z_pred < 1.5) &
      (Z_obs_raw > 0.) & (Z_obs_raw < 1.5) &
      (u >= 0) & (u < w_img - 1) &
      (v >= 0) & (v < h_img - 1)
  )

  diff = torch.abs(Z_obs_raw[valid_mask] - Z_pred[valid_mask])
  return torch.nan_to_num(diff.mean(), nan=0.0)
