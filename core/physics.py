"""Robot physics renderers for the DROID pipeline.

TensorRobotRenderer (yourdfpy): differentiable point cloud renderer for
  camera-robot depth alignment in compute_extrinsics.
PyBulletRenderer (pybullet): segmentation + depth renderer for robot
  masking and URDF tracking in compute_tracks.
"""

import os

import numpy as np
import pybullet as p

import torch
import trimesh
import yourdfpy

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_URDF = os.path.join(
    _SCRIPT_DIR, "assets", "franka_description",
    "franka_panda_robotiq_2f85_og.urdf",
)


# ===========================================================================
# TensorRobotRenderer — for compute_extrinsics (differentiable alignment)
# ===========================================================================

class TensorRobotRenderer:
  """High-speed robot point cloud renderer using yourdfpy forward kinematics."""

  def __init__(self, urdf_path=None, device="cuda", total_samples=100000):
    if urdf_path is None:
      urdf_path = _DEFAULT_URDF
    self.device = device
    self.dtype = torch.float32
    self.total_samples = total_samples
    print(f"[TensorRobotRenderer] Loading yourdfpy model from {urdf_path}...")
    self.robot = yourdfpy.URDF.load(urdf_path)

    self.mesh_points = {}
    self.world_points_cache = {}

    # Parse gripper joint names and kinematic mirror signs
    self.gripper_joint_names = []
    self.gripper_signs = []
    for joint_name, joint in self.robot.joint_map.items():
      if joint.type != 'fixed' and 'panda_joint' not in joint_name:
        self.gripper_joint_names.append(joint_name)
        base_sign = -1 if 'right' in joint_name else 1
        self.gripper_signs.append(
            base_sign * (-1 if any(k in joint_name for k in ['inner_finger', 'follower', 'finger_tip']) else 1)
        )

  def _sample_mesh(self):
    """Pre-sample mesh surface points and face normals."""
    node_names, mesh_objs, mesh_areas = [], [], []
    for node_name in self.robot.scene.graph.nodes_geometry:
      _, geom_name = self.robot.scene.graph[node_name]
      mesh = self.robot.scene.geometry.get(geom_name)
      if mesh is None or mesh.area <= 0:
        continue
      area = mesh.area * (0.0001 if any(k in geom_name.lower() for k in ['hand_camera', 'camera']) else 1.0)
      node_names.append(node_name)
      mesh_objs.append(mesh)
      mesh_areas.append(area)

    total_area = sum(mesh_areas)
    for name, mesh, area in zip(node_names, mesh_objs, mesh_areas):
      count = max(100, int(self.total_samples * area / total_area))
      pts, face_idx = trimesh.sample.sample_surface(mesh, count)
      self.mesh_points[name] = torch.tensor(
          np.hstack([pts, mesh.face_normals[face_idx]]),
          dtype=self.dtype, device=self.device,
      )

  def get_world_points(self, joint_positions, gripper_state, only_gripper=False, num_points=None):
    """Return world-frame 3D points with normals as (N, 6) tensor."""
    cache_key = tuple(
        np.round(joint_positions, 4).tolist()
        + [round(float(gripper_state), 4), int(only_gripper), num_points]
    )
    if cache_key in self.world_points_cache:
      return self.world_points_cache[cache_key]

    # Assemble joint configuration
    cfg = {f'panda_joint{i+1}': float(joint_positions[i]) for i in range(7)}
    angle = (np.clip(float(gripper_state), 0.0, 1.0) * 0.8028) - 0.08
    for j_name, sign in zip(self.gripper_joint_names, self.gripper_signs):
      cfg[j_name] = angle * sign

    self.robot.update_cfg(cfg)
    if not self.mesh_points:
      self._sample_mesh()

    all_pts = []
    gripper_keywords = ['hand', 'link8', 'robotiq', 'finger', 'knuckle', 'follower', 'pad', 'inner', 'outer']

    for node_name, local_data in self.mesh_points.items():
      if only_gripper and not any(k in node_name.lower() for k in gripper_keywords):
        continue

      local_pts, local_normals = local_data[:, :3], local_data[:, 3:]
      pose = torch.tensor(self.robot.scene.graph[node_name][0], dtype=self.dtype, device=self.device)

      pts_h = torch.cat([local_pts, torch.ones((local_pts.shape[0], 1), device=self.device, dtype=self.dtype)], dim=1)
      world_pts = torch.mm(pts_h, pose.T)[:, :3]
      world_normals = torch.mm(local_normals, pose[:3, :3].T)

      all_pts.append(torch.cat([world_pts, world_normals], dim=1))

    if not all_pts:
      return None
    out_pts = torch.cat(all_pts, dim=0)

    # Random subsample if requested
    if num_points is not None and out_pts.shape[0] > num_points:
      out_pts = out_pts[torch.randperm(out_pts.shape[0], device=self.device)[:num_points]]

    self.world_points_cache[cache_key] = out_pts
    return out_pts


# ===========================================================================
# PyBulletRenderer — for compute_tracks (segmentation + masking)
# ===========================================================================

class PyBulletRenderer:
  """Dual-body PyBullet renderer: Franka arm + Robotiq gripper.

  The 'robot' body renders the arm links (hand/finger hidden).
  The 'ghost' body renders the gripper (arm links hidden).
  Together they form the complete visual model.
  """

  def __init__(self, ghost_urdf=None):
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

    # No EGL plugin on purpose -- see _render_raw.

    # Real body: thin arm (hand/finger hidden)
    self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    self.arm_joints = [
        i for i in range(p.getNumJoints(self.robot_id))
        if "panda_joint" in p.getJointInfo(self.robot_id, i)[1].decode()
        and p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED
    ]

    self.hidden_robot_links = []
    for i in range(-1, p.getNumJoints(self.robot_id)):
      name = (p.getBodyInfo(self.robot_id)[0].decode() if i == -1
              else p.getJointInfo(self.robot_id, i)[12].decode())
      if "hand" in name or "finger" in name:
        p.changeVisualShape(self.robot_id, i, rgbaColor=[0, 0, 0, 0])
        self.hidden_robot_links.append(i)

    # Ghost body: Robotiq gripper (arm links hidden)
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
      if "panda_link" in name:
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
    # ER_TINY_RENDERER, the CPU rasteriser, spelled out rather than left to the
    # ER_BULLET_HARDWARE_OPENGL fallback. The GPU path is ~2.5x faster (28 vs
    # 99 ms per 1280x720 frame on an A100) but produces a *different* image:
    # both bodies are hidden link-by-link with rgbaColor alpha=0, and the EGL
    # rasteriser draws alpha=0 links anyway. The ghost's hidden arm then
    # occludes the real one -- measured on the Panda, the robot mask collapses
    # from 4.43% to 0.36% of the frame, IoU 0.08. Switching renderers means
    # first hiding links some way that both rasterisers respect.
    _, _, _, depth_buf, seg_buf = p.getCameraImage(
        w, h, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER,
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
