"""DROID Stage 3: Multi-View CoTracker Tracking + Median 3D Fusion.

Reads Stage 1 (depth) and Stage 2 (extrinsics) outputs, runs CoTracker3
dense 2D tracking on every camera view, fuses 3D via median across views
and frames, and exports unified multi-view track data.

Pipeline:
  Phase 1: Per-view independent CoTracker dense 2D tracking
  Phase 2: Lift to 3D + URDF robot mask filtering
  Phase 3: 3D nearest-neighbor dedup → unified point set
  Phase 4: Multi-keyframe CoTracker query mode cross-view completion
  Phase 5: Multi-view multi-frame nanmedian 3D fusion + smoothing
  Phase 6: Quality filtering + export

Dual-Track Architecture:
  Track A (Environment): CoTracker dense 2D → 3D dedup → cross-view query → median fusion
  Track B (Robot):       URDF forward kinematics → per-link binding → cross-view projection
"""

import argparse
import fcntl
import json
import os
import random
import sys
import warnings
from collections import defaultdict

import cv2
import numpy as np
import pybullet as p
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Hardware Setup
# ---------------------------------------------------------------------------
def get_accelerator(force_egl=True):
  """Configure headless GPU hardware acceleration and return active device."""
  if force_egl:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Model Loading (CoTracker3 only for Stage 3)
# ---------------------------------------------------------------------------
def init_tracking_models():
  """Load CoTracker3 model for dense 2D tracking."""
  device = get_accelerator()
  print(f"🚀 Launching CoTracker3 onto {device} | CUDA_VISIBLE_DEVICES: "
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
  if not torch.cuda.is_available():
    print("⚠️ WARNING: PyTorch cannot find a valid CUDA device.")

  vendor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "third_party")
  cotracker_path = os.path.join(vendor_dir, "co-tracker")
  if cotracker_path not in sys.path:
    sys.path.append(cotracker_path)

  from cotracker.predictor import CoTrackerPredictor

  cotracker_model = CoTrackerPredictor(
      checkpoint=os.path.join(cotracker_path,
                              "weights/cotracker3_offline.pth")
  ).to(device)
  print("  ✅ CoTracker3 loaded.")
  return cotracker_model


# ---------------------------------------------------------------------------
# 2. Data Loading
# ---------------------------------------------------------------------------
def load_depth_data(episode_id, depth_root="~/droid_data/output/mv-tap/droid/depth"):
  """Load Stage 1 depth outputs (same loader as compute_extrinsics)."""
  ep_dir = os.path.abspath(
      os.path.expanduser(os.path.join(depth_root, episode_id))
  )
  if not os.path.isdir(ep_dir):
    raise FileNotFoundError(f"Depth output not found: {ep_dir}")

  # --- Robot kinematics ---
  robot_data = np.load(os.path.join(ep_dir, "robot.npz"), allow_pickle=True)
  wrist_serial = (str(robot_data["wrist_serial"])
                  if "wrist_serial" in robot_data else None)

  robot = {
      "joint_positions": robot_data["joint_positions"].astype(np.float32),
      "gripper_positions": robot_data["gripper_positions"].astype(np.float32),
  }

  # --- Per-camera data ---
  camera = {}
  cam_dirs = [
      d for d in os.listdir(ep_dir)
      if os.path.isdir(os.path.join(ep_dir, d))
  ]

  for cam_id in sorted(cam_dirs):
    cam_path = os.path.join(ep_dir, cam_id)
    cam_data = {}

    # Calibration
    calib_path = os.path.join(cam_path, "calibration.npz")
    if os.path.exists(calib_path):
      calib = np.load(calib_path)
      cam_data["K_mat"] = calib["K_calib_left"].astype(np.float32)

    # Depth
    depth_path = os.path.join(cam_path, "raw_depth.npz")
    if os.path.exists(depth_path):
      depth_uint16 = np.load(depth_path)["depth"]
      cam_data["raw_depth"] = depth_uint16.astype(np.float32) / 1000.0

    # Video
    video_path = os.path.join(cam_path, "video_left.mp4")
    if os.path.exists(video_path):
      cap = cv2.VideoCapture(video_path)
      frames = []
      while True:
        ret, frame = cap.read()
        if not ret:
          break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      cap.release()
      if frames:
        cam_data["video_rgb"] = np.array(frames, dtype=np.uint8)

    camera[cam_id] = cam_data

  scene_constants = {
      "meta": {
          "episode_id": episode_id,
          "wrist_serial": wrist_serial,
      },
      "robot": robot,
      "camera": camera,
  }

  n_frames = len(robot["joint_positions"])
  n_cams = len(camera)
  print(f"  ✅ Loaded depth data: {n_cams} cameras, {n_frames} frames.")
  return scene_constants


def load_extrinsics(scene_constants,
                    extrinsics_root="~/droid_data/output/mv-tap/droid/extrinsics"):
  """Load Stage 2 extrinsics outputs."""
  ep_id = scene_constants["meta"]["episode_id"]
  ep_dir = os.path.abspath(
      os.path.expanduser(os.path.join(extrinsics_root, ep_id))
  )

  scene_state = {}
  for cam_id in scene_constants["camera"]:
    cam_ext_path = os.path.join(ep_dir, cam_id, "extrinsics.json")
    if not os.path.exists(cam_ext_path):
      raise FileNotFoundError(f"Extrinsics not found: {cam_ext_path}")

    with open(cam_ext_path, "r") as f:
      payload = json.load(f)

    scene_state[cam_id] = {
        "base_extrinsic": np.array(payload["base_extrinsic"],
                                   dtype=np.float32),
        "extrinsics": np.array(payload["extrinsics"], dtype=np.float32),
    }

  print(f"  ✅ Loaded extrinsics for {len(scene_state)} cameras.")
  return scene_state


# ---------------------------------------------------------------------------
# 3. Geometry Utilities
# ---------------------------------------------------------------------------
def unproject_points_np(u, v, z, K, T_cam2world=None):
  """Back-project 2D pixel coordinates to 3D world points."""
  x_cam = (u - K[0, 2]) * z / K[0, 0]
  y_cam = (v - K[1, 2]) * z / K[1, 1]
  pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=0)
  if T_cam2world is None:
    return pts_cam[:3, :].T
  return (T_cam2world @ pts_cam)[:3, :].T


def project_points_np(pts_world, K, T_cam2world):
  """Project 3D world points to 2D pixel coordinates."""
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


# ---------------------------------------------------------------------------
# 4. URDF Robot Renderer (for robot masking)
# ---------------------------------------------------------------------------
class PyBulletRenderer:
  """Dual-body PyBullet renderer: Franka arm + Robotiq gripper.

  The 'robot' body renders the arm links (hand/finger hidden).
  The 'ghost' body renders the gripper (arm links hidden).
  Together they form the complete visual model.
  """

  def __init__(self, ghost_urdf=None):
    import pybullet_data
    import importlib.util

    if ghost_urdf is None:
      ghost_urdf = os.path.join(
          os.path.dirname(os.path.abspath(__file__)),
          "third_party/PointWorld/assets/franka_description/"
          "franka_panda_robotiq_2f85_og.urdf")

    if p.isConnected():
      p.disconnect()
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    egl_spec = importlib.util.find_spec('eglRendererPlugin')
    if egl_spec:
      p.loadPlugin(egl_spec.origin, "_eglRendererPlugin")

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
    p.performCollisionDetection()

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
    target = cam_pos + extrinsic[:3, 2]
    view_matrix = p.computeViewMatrix(
        cam_pos.tolist(), (cam_pos + extrinsic[:3, 2]).tolist(),
        (-extrinsic[:3, 1]).tolist())
    proj_matrix = self._get_projection_matrix(K, w, h)
    _, _, _, depth_buf, seg_buf = p.getCameraImage(
        w, h, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
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


# ---------------------------------------------------------------------------
# 5. URDF Kinematics Tracker (Track B: Robot)
# ---------------------------------------------------------------------------
class URDFKinematicsTracker:
  """Forward kinematics-based robot 3D trajectory generator.

  Binds seed 2D tracking points to URDF link frames at t=0,
  then propagates them via forward kinematics across all frames.
  """

  def __init__(self, pb_renderer):
    self.pb = pb_renderer

  def _get_link_transform(self, obj_id, link_id):
    """Get 4x4 world-frame transform for a PyBullet link."""
    if link_id == -1:
      pos, orn = p.getBasePositionAndOrientation(obj_id)
    else:
      state = p.getLinkState(obj_id, link_id)
      pos, orn = state[0], state[1]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(orn).as_matrix()
    T[:3, 3] = pos
    return T

  def extract_robot_tracks(self, src_cam, scene_constants, scene_state,
                           safe_margin=7):
    """Extract 3D robot surface trajectories via URDF forward kinematics.

    Returns:
      traj_3d: (T, N_robot, 3) world coordinates
      traj_2d: (T, N_robot, 2) pixel coordinates in src_cam
      vis_2d:  (T, N_robot) visibility mask in src_cam
      robot_indices: indices into the original tracks_2d
    """
    print(f"    🦾 URDF tracking [{src_cam}]")
    src_data = scene_constants["camera"][src_cam]
    src_state = scene_state[src_cam]
    K_mat = src_data["K_mat"]
    extrinsics = src_state["extrinsics"]
    h_img, w_img = src_data["video_rgb"][0].shape[:2]
    n_frames = len(src_data["video_rgb"])
    tracks_2d_t0 = src_data["tracks_2d"][0]

    # Frame 0: find seed points on robot
    self.pb.update_robot_pose(
        scene_constants["robot"]["joint_positions"][0],
        gripper_state=scene_constants["robot"]["gripper_positions"][0])

    obj_ids, link_ids, urdf_depth = self.pb.render_segmentation(
        extrinsics[0], K_mat, w_img, h_img)
    is_robot = ((obj_ids == self.pb.robot_id) |
                (obj_ids == self.pb.ghost_id))

    # Erode mask to avoid edge artifacts
    kernel = np.ones((safe_margin, safe_margin), np.uint8)
    is_robot_safe = cv2.erode(
        is_robot.astype(np.uint8), kernel, iterations=1) > 0

    u0 = np.clip(np.round(tracks_2d_t0[:, 0]).astype(int), 0, w_img - 1)
    v0 = np.clip(np.round(tracks_2d_t0[:, 1]).astype(int), 0, h_img - 1)
    robot_indices = np.where(is_robot_safe[v0, u0])[0]

    if len(robot_indices) == 0:
      print("      ⚠️ No robot points found at t=0.")
      return None, None, None, None

    print(f"      Found {len(robot_indices)} safe robot surface points.")

    # Bind to local link frames
    robot_objs = obj_ids[v0[robot_indices], u0[robot_indices]]
    robot_links = link_ids[v0[robot_indices], u0[robot_indices]]
    z0 = urdf_depth[v0[robot_indices], u0[robot_indices]]

    pts_world_t0 = unproject_points_np(
        tracks_2d_t0[robot_indices, 0],
        tracks_2d_t0[robot_indices, 1],
        z0, K_mat, extrinsics[0])

    unique_parts = set(zip(robot_objs, robot_links))
    local_pts_dict = {}
    for oid, lid in unique_parts:
      mask = (robot_objs == oid) & (robot_links == lid)
      pts = pts_world_t0[mask]
      T_link = self._get_link_transform(oid, lid)
      T_inv = np.linalg.inv(T_link)
      P_homo = np.hstack([pts, np.ones((len(pts), 1))]).T
      local_pts_dict[(oid, lid)] = (mask, T_inv @ P_homo)

    # Forward kinematics propagation
    traj_3d = np.zeros((n_frames, len(robot_indices), 3), dtype=np.float32)
    traj_2d = np.zeros((n_frames, len(robot_indices), 2), dtype=np.float32)
    vis_2d = np.zeros((n_frames, len(robot_indices)), dtype=bool)

    # Cache URDF depth per frame — render once, reuse across all points.
    # PyBullet render is the dominant cost; don't call it multiple times/frame.
    urdf_depth_cache = {}

    for t in range(n_frames):
      self.pb.update_robot_pose(
          scene_constants["robot"]["joint_positions"][t],
          gripper_state=scene_constants["robot"]["gripper_positions"][t])

      for oid, lid in unique_parts:
        mask, P_local = local_pts_dict[(oid, lid)]
        T_link_t = self._get_link_transform(oid, lid)
        P_world_t = T_link_t @ P_local
        traj_3d[t, mask, :] = P_world_t[:3, :].T

      u_t, v_t, z_pred = project_points_np(
          traj_3d[t], K_mat, extrinsics[t])
      traj_2d[t, :, 0] = u_t
      traj_2d[t, :, 1] = v_t

      # Render once per frame and cache for project_to_all_views reuse
      urdf_depth_t = self.pb.render_depth(extrinsics[t], K_mat, w_img, h_img)
      urdf_depth_cache[(src_cam, t)] = urdf_depth_t
      raw_depth_t = src_data["raw_depth"][t]

      ui = np.clip(np.round(u_t).astype(int), 0, w_img - 1)
      vi = np.clip(np.round(v_t).astype(int), 0, h_img - 1)

      in_bounds = ((u_t >= 0) & (u_t < w_img) &
                   (v_t >= 0) & (v_t < h_img) & (z_pred > 0))
      z_urdf = urdf_depth_t[vi, ui]
      not_self_occ = (z_urdf > 0) & (z_pred <= z_urdf + 0.015)
      z_sensor = raw_depth_t[vi, ui]
      not_env_occ = ~((z_sensor > 0) & (z_pred > z_sensor + 0.02))
      vis_2d[t] = in_bounds & not_self_occ & not_env_occ

    # Store cache on self so project_to_all_views can reuse renders
    if not hasattr(self, '_urdf_depth_cache'):
      self._urdf_depth_cache = {}
    self._urdf_depth_cache.update(urdf_depth_cache)

    return traj_3d, traj_2d, vis_2d, robot_indices

  def project_to_all_views(self, traj_3d, scene_constants, scene_state):
    """Project robot 3D tracks to all camera views with visibility.

    Returns:
      per_cam_traj_2d: {cam_id: (T, N_robot, 2)}
      per_cam_vis: {cam_id: (T, N_robot)}
    """
    camera_ids = list(scene_constants["camera"].keys())
    T, N_robot, _ = traj_3d.shape
    per_cam_traj_2d = {}
    per_cam_vis = {}

    for cam_id in camera_ids:
      cam_data = scene_constants["camera"][cam_id]
      cam_state = scene_state[cam_id]
      K = cam_data["K_mat"]
      h_img, w_img = cam_data["video_rgb"][0].shape[:2]

      cam_traj_2d = np.zeros((T, N_robot, 2), dtype=np.float32)
      cam_vis = np.zeros((T, N_robot), dtype=bool)

      cache = getattr(self, '_urdf_depth_cache', {})

      for t in range(T):
        # Only call update_robot_pose + render_depth if not already cached
        cache_key = (cam_id, t)
        if cache_key not in cache:
          self.pb.update_robot_pose(
              scene_constants["robot"]["joint_positions"][t],
              gripper_state=scene_constants["robot"]["gripper_positions"][t])
          cache[cache_key] = self.pb.render_depth(
              cam_state["extrinsics"][t], K, w_img, h_img)

        urdf_depth_t = cache[cache_key]

        u_t, v_t, z_pred = project_points_np(
            traj_3d[t], K, cam_state["extrinsics"][t])
        cam_traj_2d[t, :, 0] = u_t
        cam_traj_2d[t, :, 1] = v_t

        raw_depth_t = cam_data["raw_depth"][t]

        ui = np.clip(np.round(u_t).astype(int), 0, w_img - 1)
        vi = np.clip(np.round(v_t).astype(int), 0, h_img - 1)

        in_bounds = ((u_t >= 0) & (u_t < w_img) &
                     (v_t >= 0) & (v_t < h_img) & (z_pred > 0))
        z_urdf = urdf_depth_t[vi, ui]
        not_self_occ = (z_urdf > 0) & (z_pred <= z_urdf + 0.015)
        z_sensor = raw_depth_t[vi, ui]
        not_env_occ = ~((z_sensor > 0) & (z_pred > z_sensor + 0.02))
        cam_vis[t] = in_bounds & not_self_occ & not_env_occ

      per_cam_traj_2d[cam_id] = cam_traj_2d
      per_cam_vis[cam_id] = cam_vis

    return per_cam_traj_2d, per_cam_vis


# ---------------------------------------------------------------------------
# 6. Phase 1: Per-view Independent CoTracker
# ---------------------------------------------------------------------------
def phase1_extract_2d_tracks(cotracker_model, scene_constants, device):
  """Run CoTracker3 dense 2D tracking on every camera view."""
  print("\n" + "=" * 60)
  print("🎯 Phase 1: Per-view Independent CoTracker Dense 2D Tracking")
  print("=" * 60)

  for cam_id in scene_constants["camera"]:
    cam_data = scene_constants["camera"][cam_id]
    if "video_rgb" not in cam_data:
      print(f"  ⚠️ [{cam_id}] No video_rgb, skipping.")
      continue

    video_tensor = (
        torch.from_numpy(cam_data["video_rgb"])
        .permute(0, 3, 1, 2)[None].float().to(device)
    )
    with torch.no_grad():
      pred_tracks, pred_vis = cotracker_model(
          video_tensor, grid_size=30, grid_query_frame=0,
          backward_tracking=False)

    cam_data["tracks_2d"] = pred_tracks[0].cpu().numpy()
    cam_data["vis_2d"] = pred_vis[0].cpu().numpy() > 0.5

    T, N, _ = cam_data["tracks_2d"].shape
    print(f"  ✅ [{cam_id}] {N} tracks × {T} frames")

    # Explicit GPU cleanup
    del video_tensor, pred_tracks, pred_vis
    torch.cuda.empty_cache()

  return scene_constants


# ---------------------------------------------------------------------------
# 6. Phase 2: Lift to 3D + Robot Mask
# ---------------------------------------------------------------------------
def phase2_lift_and_filter(scene_constants, scene_state, pb_renderer):
  """Lift t=0 2D tracks to 3D and filter out robot points."""
  print("\n" + "=" * 60)
  print("🦴 Phase 2: Lift to 3D + Robot Mask Filtering")
  print("=" * 60)

  camera_ids = list(scene_constants["camera"].keys())
  per_cam_env = {}

  for cam_id in camera_ids:
    cam_data = scene_constants["camera"][cam_id]
    cam_state = scene_state[cam_id]

    if "tracks_2d" not in cam_data or "raw_depth" not in cam_data:
      per_cam_env[cam_id] = None
      continue

    tracks_2d = cam_data["tracks_2d"]
    vis_2d = cam_data["vis_2d"]
    h_img, w_img = cam_data["video_rgb"][0].shape[:2]

    # Robot mask at t=0
    pb_renderer.update_robot_pose(
        scene_constants["robot"]["joint_positions"][0],
        gripper_state=scene_constants["robot"]["gripper_positions"][0])
    robot_mask = pb_renderer.render_mask(
        cam_state["extrinsics"][0], cam_data["K_mat"], w_img, h_img)
    kernel = np.ones((15, 15), np.uint8)
    robot_mask_dilated = cv2.dilate(
        robot_mask.astype(np.uint8), kernel, iterations=1) > 0

    # Query each track at t=0
    u0 = np.clip(np.round(tracks_2d[0, :, 0]).astype(int), 0, w_img - 1)
    v0 = np.clip(np.round(tracks_2d[0, :, 1]).astype(int), 0, h_img - 1)
    z0 = cam_data["raw_depth"][0, v0, u0]

    is_env = ~robot_mask_dilated[v0, u0]
    has_depth = (z0 > 0.05) & (z0 < 5.0)
    env_indices = np.where(is_env & has_depth)[0]

    if len(env_indices) == 0:
      print(f"  ⚠️ [{cam_id}] No environment points, skipping.")
      per_cam_env[cam_id] = None
      continue

    # Lift to 3D
    pts_3d_t0 = unproject_points_np(
        tracks_2d[0, env_indices, 0],
        tracks_2d[0, env_indices, 1],
        z0[env_indices],
        cam_data["K_mat"],
        cam_state["extrinsics"][0])

    per_cam_env[cam_id] = {
        "env_indices": env_indices,
        "pts_3d_t0": pts_3d_t0,
        "tracks_2d": tracks_2d[:, env_indices, :],
        "vis_2d": vis_2d[:, env_indices],
    }

    n_env = len(env_indices)
    n_robot = np.sum(~is_env)
    print(f"  ✅ [{cam_id}] Env: {n_env} | Robot: {n_robot} | "
          f"No depth: {np.sum(~has_depth)}")

  return per_cam_env


# ---------------------------------------------------------------------------
# 7. Phase 3: 3D Nearest-Neighbor Dedup
# ---------------------------------------------------------------------------
def phase3_3d_dedup(per_cam_env, camera_ids, match_radius=0.015):
  """Deduplicate across views using 3D nearest-neighbor matching."""
  print("\n" + "=" * 60)
  print("🔗 Phase 3: 3D Nearest-Neighbor Dedup → Unified Point Set")
  print("=" * 60)

  # Collect all 3D seeds
  all_pts = []
  all_cam_labels = []

  for cam_id in camera_ids:
    if per_cam_env.get(cam_id) is None:
      continue
    pts = per_cam_env[cam_id]["pts_3d_t0"]
    for i in range(len(pts)):
      all_cam_labels.append((cam_id, i))
    all_pts.append(pts)

  if not all_pts:
    print("  ⚠️ No 3D points from any view.")
    return None, None, None

  all_pts = np.concatenate(all_pts, axis=0)
  N_all = len(all_pts)
  print(f"  📊 Total {N_all} 3D seed points from {len(per_cam_env)} views")

  # Union-Find
  parent = list(range(N_all))

  def find(x):
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x

  def union_op(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
      parent[ra] = rb

  tree = cKDTree(all_pts)
  pairs = tree.query_pairs(r=match_radius)
  print(f"  🔗 Found {len(pairs)} close-distance pairs")

  for a, b in pairs:
    union_op(a, b)

  # Build unified point set
  groups = defaultdict(list)
  for i in range(N_all):
    groups[find(i)].append(i)

  N_unified = len(groups)
  print(f"  🏆 Unified set: {N_unified} unique points "
        f"(removed {N_all - N_unified} duplicates)")

  unified_pts_3d = np.zeros((N_unified, 3), dtype=np.float32)
  unified_to_cam = [dict() for _ in range(N_unified)]

  for uid, (_, members) in enumerate(groups.items()):
    member_pts = all_pts[members]
    unified_pts_3d[uid] = np.median(member_pts, axis=0)
    for m in members:
      cam_id, local_idx = all_cam_labels[m]
      unified_to_cam[uid][cam_id] = local_idx

  # Coverage stats
  coverage = np.array([len(d) for d in unified_to_cam])
  for n_views in range(1, len(camera_ids) + 1):
    count = np.sum(coverage >= n_views)
    if count > 0:
      print(f"    Seen by ≥{n_views} views: {count} points")

  return unified_pts_3d, unified_to_cam, N_unified


# ---------------------------------------------------------------------------
# 8. Phase 4: Multi-Keyframe CoTracker Query + 2D Median Fusion
# ---------------------------------------------------------------------------
def phase4_cross_view_completion(cotracker_model, scene_constants, scene_state,
                                 per_cam_env, unified_pts_3d, unified_to_cam,
                                 N_unified, device):
  """Complete missing tracks via multi-keyframe CoTracker query mode."""
  print("\n" + "=" * 60)
  print("🔄 Phase 4: Multi-Keyframe CoTracker Query + 2D Median Completion")
  print("=" * 60)

  camera_ids = list(scene_constants["camera"].keys())
  T_frames = len(scene_constants["camera"][camera_ids[0]]["video_rgb"])
  keyframes = [0, T_frames // 4, T_frames // 2,
               3 * T_frames // 4, T_frames - 1]
  print(f"  🕐 Using {len(keyframes)} keyframes: {keyframes}")

  # Output structures
  per_cam_tracks = {
      cam: np.full((T_frames, N_unified, 2), -1000.0, np.float32)
      for cam in camera_ids}
  per_cam_vis = {
      cam: np.zeros((T_frames, N_unified), bool)
      for cam in camera_ids}

  # Step 1: Fill native tracks from Phase 1
  for uid in range(N_unified):
    for cam_id, local_idx in unified_to_cam[uid].items():
      if per_cam_env.get(cam_id) is None:
        continue
      per_cam_tracks[cam_id][:, uid, :] = (
          per_cam_env[cam_id]["tracks_2d"][:, local_idx, :])
      per_cam_vis[cam_id][:, uid] = (
          per_cam_env[cam_id]["vis_2d"][:, local_idx])

  for cam_id in camera_ids:
    n_have = sum(1 for uid in range(N_unified)
                 if cam_id in unified_to_cam[uid])
    print(f"  [{cam_id}] Native: {n_have} | Need: {N_unified - n_have}")

  # Step 2: Multi-keyframe query + 2D median fusion
  for cam_id in camera_ids:
    cam_data = scene_constants["camera"][cam_id]
    cam_state = scene_state[cam_id]
    if "video_rgb" not in cam_data:
      continue
    h_img, w_img = cam_data["video_rgb"][0].shape[:2]

    missing_uids = [uid for uid in range(N_unified)
                    if cam_id not in unified_to_cam[uid]]
    if not missing_uids:
      print(f"  ⏭️ [{cam_id}] No missing, skip.")
      continue

    M = len(missing_uids)
    pts_3d_missing = unified_pts_3d[missing_uids]

    video_tensor = (
        torch.from_numpy(cam_data["video_rgb"])
        .permute(0, 3, 1, 2)[None].float().to(device)
    )

    vote_tracks = []
    vote_vis = []

    for t_k in keyframes:
      u_seed, v_seed, z_seed = project_points_np(
          pts_3d_missing, cam_data["K_mat"],
          cam_state["extrinsics"][t_k])
      ui_seed = np.clip(np.round(u_seed).astype(int), 0, w_img - 1)
      vi_seed = np.clip(np.round(v_seed).astype(int), 0, h_img - 1)
      z_sensor = cam_data["raw_depth"][t_k, vi_seed, ui_seed]

      in_bounds = ((u_seed >= 0) & (u_seed < w_img) &
                   (v_seed >= 0) & (v_seed < h_img) & (z_seed > 0))
      not_occluded = (z_sensor > 0) & (z_sensor >= z_seed - 0.02)
      valid_seed = in_bounds & not_occluded
      valid_indices = np.where(valid_seed)[0]

      if len(valid_indices) == 0:
        vote_tracks.append(
            np.full((T_frames, M, 2), np.nan, np.float32))
        vote_vis.append(np.zeros((T_frames, M), bool))
        continue

      queries_np = np.stack([
          np.full(len(valid_indices), t_k, dtype=np.float32),
          u_seed[valid_indices],
          v_seed[valid_indices]], axis=-1)
      queries_t = torch.tensor(
          queries_np, dtype=torch.float32, device=device)[None]

      with torch.no_grad():
        pred_tracks, pred_vis = cotracker_model(
            video_tensor, queries=queries_t, backward_tracking=True)

      kf_tracks = np.full((T_frames, M, 2), np.nan, np.float32)
      kf_vis = np.zeros((T_frames, M), bool)
      kf_tracks[:, valid_indices, :] = pred_tracks[0].cpu().numpy()
      kf_vis[:, valid_indices] = pred_vis[0].cpu().numpy() > 0.5
      kf_tracks[~kf_vis] = np.nan

      vote_tracks.append(kf_tracks)
      vote_vis.append(kf_vis)

      del pred_tracks, pred_vis
      torch.cuda.empty_cache()

    # 2D Median fusion across keyframes
    stacked_tracks = np.stack(vote_tracks, axis=0)  # (K, T, M, 2)
    stacked_vis = np.stack(vote_vis, axis=0)          # (K, T, M)

    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=RuntimeWarning)
      median_tracks = np.nanmedian(stacked_tracks, axis=0)  # (T, M, 2)

    any_vis = np.any(stacked_vis, axis=0)
    has_value = ~np.isnan(median_tracks[:, :, 0])
    fused_vis = any_vis & has_value

    for j, uid in enumerate(missing_uids):
      valid_frames = fused_vis[:, j]
      per_cam_tracks[cam_id][valid_frames, uid, :] = (
          median_tracks[valid_frames, j, :])
      per_cam_vis[cam_id][:, uid] = fused_vis[:, j]

    n_any_vis = np.sum(np.any(fused_vis, axis=0))
    print(f"  ✅ [{cam_id}] Filled {n_any_vis}/{M} missing points "
          f"({len(keyframes)} keyframes)")

    del video_tensor
    torch.cuda.empty_cache()

  # Final coverage stats
  print("\n📊 Post-completion coverage:")
  for cam_id in camera_ids:
    n_vis = np.sum(per_cam_vis[cam_id][0])
    pct = n_vis / N_unified * 100 if N_unified > 0 else 0
    print(f"  [{cam_id}] t=0 visible: {n_vis}/{N_unified} ({pct:.1f}%)")

  return per_cam_tracks, per_cam_vis


# ---------------------------------------------------------------------------
# 9. Phase 5: Multi-View Multi-Frame Median 3D Fusion
# ---------------------------------------------------------------------------
def phase5_median_3d_fusion(scene_constants, scene_state, per_cam_tracks,
                            per_cam_vis, N_unified, min_views=2,
                            min_visible_frames=5):
  """Fuse 3D positions across views and frames using nanmedian."""
  print("\n" + "=" * 60)
  print("📐 Phase 5: Multi-View Multi-Frame Median 3D Fusion")
  print("=" * 60)

  camera_ids = list(scene_constants["camera"].keys())
  T_frames = len(scene_constants["camera"][camera_ids[0]]["video_rgb"])

  # Collect per-view 3D observations
  per_view_3d = np.full(
      (len(camera_ids), T_frames, N_unified, 3), np.nan, dtype=np.float32)

  for v_idx, cam_id in enumerate(camera_ids):
    cam_data = scene_constants["camera"][cam_id]
    cam_state = scene_state[cam_id]
    h_img, w_img = cam_data["video_rgb"][0].shape[:2]

    for t in range(T_frames):
      tracks_t = per_cam_tracks[cam_id][t]
      vis_t = per_cam_vis[cam_id][t]

      u = tracks_t[:, 0]
      v = tracks_t[:, 1]
      ui = np.clip(np.round(u).astype(int), 0, w_img - 1)
      vi = np.clip(np.round(v).astype(int), 0, h_img - 1)
      z = cam_data["raw_depth"][t, vi, ui]

      valid = (vis_t & (z > 0.05) & (z < 5.0) &
               (u >= 0) & (u < w_img) &
               (v >= 0) & (v < h_img))

      if valid.any():
        pts_3d = unproject_points_np(
            u[valid], v[valid], z[valid],
            cam_data["K_mat"], cam_state["extrinsics"][t])
        per_view_3d[v_idx, t, valid, :] = pts_3d

  # Median fusion
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    fused_traj_3d = np.nanmedian(per_view_3d, axis=0)

  observation_counts = np.sum(
      ~np.isnan(per_view_3d[:, :, :, 0]), axis=0)
  avg_obs = np.nanmean(observation_counts)
  print(f"  📊 Average views per point per frame: {avg_obs:.1f}")

  # Gaussian smoothing
  import pandas as pd
  smoothed_3d = fused_traj_3d.copy()
  valid_any = ~np.all(np.isnan(fused_traj_3d[:, :, 0]), axis=0)
  n_valid = np.sum(valid_any)
  print(f"  📊 Points with any valid observations: {n_valid}/{N_unified}")

  if n_valid > 0:
    df = pd.DataFrame(smoothed_3d[:, valid_any, :].reshape(T_frames, -1))
    df = df.interpolate(method="linear", limit_direction="both")
    interpolated = df.to_numpy().reshape(T_frames, n_valid, 3)
    smoothed_valid = gaussian_filter1d(interpolated, sigma=1.5, axis=0)
    smoothed_3d[:, valid_any, :] = smoothed_valid

  # Visibility: >= min_views observations at this frame
  fused_vis = observation_counts >= min_views

  # Quality filter
  total_visible = np.sum(fused_vis, axis=0)
  quality_mask = total_visible >= min_visible_frames
  n_survived = np.sum(quality_mask)
  print(f"  🏆 Quality filter (≥{min_visible_frames} frames visible): "
        f"{n_survived}/{N_unified} survived")

  # Apply filter
  final_traj_3d = smoothed_3d[:, quality_mask, :]
  final_vis_global = fused_vis[:, quality_mask]
  final_per_cam_tracks = {
      cam: per_cam_tracks[cam][:, quality_mask, :]
      for cam in camera_ids}
  final_per_cam_vis = {
      cam: per_cam_vis[cam][:, quality_mask]
      for cam in camera_ids}

  avg_visible = np.mean(np.sum(final_vis_global, axis=1))
  print(f"  📊 Avg visible points per frame: {avg_visible:.0f}")

  return (final_traj_3d, final_vis_global, final_per_cam_tracks,
          final_per_cam_vis, n_survived)


# ---------------------------------------------------------------------------
# 10. Phase 6: Export
# ---------------------------------------------------------------------------
def export_tracks(scene_constants, scene_state, final_traj_3d,
                  final_vis_global, final_per_cam_tracks, final_per_cam_vis,
                  export_root="~/droid_data/output/mv-tap/droid/tracks"):
  """Serialize tracking results to disk."""
  ep_id = scene_constants["meta"]["episode_id"]
  camera_ids = list(scene_constants["camera"].keys())
  ep_dir = os.path.abspath(
      os.path.expanduser(os.path.join(export_root, ep_id))
  )
  os.makedirs(ep_dir, exist_ok=True)

  T, N, _ = final_traj_3d.shape

  # Global 3D trajectories
  np.savez_compressed(
      os.path.join(ep_dir, "tracks_3d.npz"),
      traj_3d=final_traj_3d.astype(np.float32),
      vis_global=final_vis_global,
  )

  # Per-camera 2D tracks + visibility
  for cam_id in camera_ids:
    cam_dir = os.path.join(ep_dir, cam_id)
    os.makedirs(cam_dir, exist_ok=True)

    # Set invisible coords to -1000
    traj_2d = final_per_cam_tracks[cam_id].copy()
    vis = final_per_cam_vis[cam_id].copy()
    traj_2d[~vis] = -1000.0

    np.savez_compressed(
        os.path.join(cam_dir, "tracks_2d.npz"),
        traj_2d=traj_2d.astype(np.float32),
        vis_2d=vis,
    )

    # Intrinsics
    cam_data = scene_constants["camera"][cam_id]
    K = cam_data["K_mat"]
    np.save(os.path.join(cam_dir, "intrinsics.npy"),
            np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                     dtype=np.float32))

    # Extrinsics (w2c)
    np.save(os.path.join(cam_dir, "extrinsics_w2c.npy"),
            np.linalg.inv(
                scene_state[cam_id]["extrinsics"]).astype(np.float32))

  print(f"  💾 Exported {N} tracks × {T} frames to {ep_dir}")
  return ep_dir


# ---------------------------------------------------------------------------
# 11. Main Pipeline
# ---------------------------------------------------------------------------
def process_episode(episode_id, cotracker_model, pb_renderer, device,
                    depth_root, extrinsics_root, export_root):
  """Full dual-track pipeline for a single episode.

  Track B (Robot): URDF forward kinematics (no learning)
  Track A (Environment): CoTracker multi-view fusion
  """
  print(f"\n{'=' * 60}")
  print(f"🎬 Processing Episode: {episode_id}")
  print(f"{'=' * 60}")

  # Load data
  scene_constants = load_depth_data(episode_id, depth_root)
  scene_state = load_extrinsics(scene_constants, extrinsics_root)

  camera_ids = list(scene_constants["camera"].keys())
  T_frames = len(scene_constants["camera"][camera_ids[0]]["video_rgb"])

  # Validate data
  has_video = all("video_rgb" in scene_constants["camera"][c]
                  for c in camera_ids)
  has_depth = all("raw_depth" in scene_constants["camera"][c]
                  for c in camera_ids)
  if not has_video or not has_depth:
    raise ValueError("Missing video_rgb or raw_depth for some cameras")

  # ================================================================
  # Phase 1: Per-view CoTracker (needed for both robot seed + env)
  # ================================================================
  scene_constants = phase1_extract_2d_tracks(
      cotracker_model, scene_constants, device)

  # ================================================================
  # Track B: Robot points via URDF forward kinematics
  # ================================================================
  print("\n" + "=" * 60)
  print("🦾 Track B: URDF Forward Kinematics Robot Tracking")
  print("=" * 60)

  urdf_tracker = URDFKinematicsTracker(pb_renderer)
  robot_traj_3d_all = []
  robot_per_cam_tracks_all = {cam: [] for cam in camera_ids}
  robot_per_cam_vis_all = {cam: [] for cam in camera_ids}

  for src_cam in camera_ids:
    traj_3d_rob, traj_2d_rob, vis_rob, robot_indices = \
        urdf_tracker.extract_robot_tracks(
            src_cam, scene_constants, scene_state)

    if traj_3d_rob is None or len(robot_indices) == 0:
      continue

    # Project robot 3D to all views
    rob_per_cam_2d, rob_per_cam_vis = \
        urdf_tracker.project_to_all_views(
            traj_3d_rob, scene_constants, scene_state)

    # Use source view's native 2D for itself
    rob_per_cam_2d[src_cam] = traj_2d_rob
    rob_per_cam_vis[src_cam] = vis_rob

    robot_traj_3d_all.append(traj_3d_rob)
    for cam in camera_ids:
      robot_per_cam_tracks_all[cam].append(rob_per_cam_2d[cam])
      robot_per_cam_vis_all[cam].append(rob_per_cam_vis[cam])

  # Concatenate robot tracks from all source views
  if robot_traj_3d_all:
    robot_traj_3d = np.concatenate(robot_traj_3d_all, axis=1)
    robot_per_cam_tracks = {
        cam: np.concatenate(robot_per_cam_tracks_all[cam], axis=1)
        for cam in camera_ids}
    robot_per_cam_vis = {
        cam: np.concatenate(robot_per_cam_vis_all[cam], axis=1)
        for cam in camera_ids}
    N_robot = robot_traj_3d.shape[1]
    print(f"  🦾 Total robot points: {N_robot}")
  else:
    robot_traj_3d = np.zeros((T_frames, 0, 3), dtype=np.float32)
    robot_per_cam_tracks = {
        cam: np.zeros((T_frames, 0, 2), dtype=np.float32)
        for cam in camera_ids}
    robot_per_cam_vis = {
        cam: np.zeros((T_frames, 0), dtype=bool)
        for cam in camera_ids}
    N_robot = 0
    print("  ⚠️ No robot points extracted.")

  # ================================================================
  # Track A: Environment points via CoTracker multi-view fusion
  # ================================================================

  # Phase 2: Lift to 3D + robot mask
  per_cam_env = phase2_lift_and_filter(
      scene_constants, scene_state, pb_renderer)

  # Phase 3: 3D dedup
  unified_pts_3d, unified_to_cam, N_unified = phase3_3d_dedup(
      per_cam_env, camera_ids)

  if unified_pts_3d is not None and N_unified > 0:
    # Phase 4: Cross-view completion
    per_cam_tracks, per_cam_vis = phase4_cross_view_completion(
        cotracker_model, scene_constants, scene_state,
        per_cam_env, unified_pts_3d, unified_to_cam,
        N_unified, device)

    # Phase 5: Median 3D fusion
    (env_traj_3d, env_vis_global, env_per_cam_tracks,
     env_per_cam_vis, N_env) = phase5_median_3d_fusion(
        scene_constants, scene_state, per_cam_tracks, per_cam_vis,
        N_unified)
  else:
    env_traj_3d = np.zeros((T_frames, 0, 3), dtype=np.float32)
    env_per_cam_tracks = {
        cam: np.zeros((T_frames, 0, 2), dtype=np.float32)
        for cam in camera_ids}
    env_per_cam_vis = {
        cam: np.zeros((T_frames, 0), dtype=bool)
        for cam in camera_ids}
    env_vis_global = np.zeros((T_frames, 0), dtype=bool)
    N_env = 0

  # ================================================================
  # Merge: Robot (Track B) + Environment (Track A)
  # ================================================================
  print("\n" + "=" * 60)
  print("🔗 Merging Track A (Environment) + Track B (Robot)")
  print("=" * 60)

  # Robot visibility is already per-cam
  robot_vis_global = np.ones((T_frames, N_robot), dtype=bool)
  for cam in camera_ids:
    robot_vis_global &= robot_per_cam_vis[cam]

  final_traj_3d = np.concatenate(
      [env_traj_3d, robot_traj_3d], axis=1)
  final_vis_global = np.concatenate(
      [env_vis_global, robot_vis_global], axis=1)
  final_per_cam_tracks = {
      cam: np.concatenate(
          [env_per_cam_tracks[cam], robot_per_cam_tracks[cam]], axis=1)
      for cam in camera_ids}
  final_per_cam_vis = {
      cam: np.concatenate(
          [env_per_cam_vis[cam], robot_per_cam_vis[cam]], axis=1)
      for cam in camera_ids}

  N_final = N_env + N_robot
  print(f"  📊 Environment: {N_env} | Robot: {N_robot} | Total: {N_final}")

  if N_final == 0:
    raise ValueError("No points from either track")

  # ================================================================
  # Export
  # ================================================================
  export_tracks(scene_constants, scene_state, final_traj_3d,
                final_vis_global, final_per_cam_tracks, final_per_cam_vis,
                export_root)

  # Also save per-type metadata for downstream consumers
  ep_dir = os.path.abspath(
      os.path.expanduser(os.path.join(export_root, episode_id)))
  np.savez_compressed(
      os.path.join(ep_dir, "track_metadata.npz"),
      n_env=np.array(N_env),
      n_robot=np.array(N_robot),
      # First N_env are environment, next N_robot are robot
      point_type=np.array(
          [0] * N_env + [1] * N_robot, dtype=np.uint8),
  )

  print(f"\n  ✅ Episode {episode_id}: {N_env} env + {N_robot} robot "
        f"= {N_final} tracks exported.")
  return N_final


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="DROID Stage 3: Multi-View Tracking + 3D Fusion")
  parser.add_argument("--rank", type=int, default=0,
                      help="Rank of the process")
  parser.add_argument("--world_size", type=int, default=1,
                      help="Total number of processes")
  parser.add_argument("--limit", type=int, default=-1,
                      help="Limit total number of episodes to process")
  parser.add_argument("--depth_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/depth",
                      help="Root directory of Stage 1 depth outputs")
  parser.add_argument("--extrinsics_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/extrinsics",
                      help="Root directory of Stage 2 extrinsics outputs")
  parser.add_argument("--export_root", type=str,
                      default="~/droid_data/output/mv-tap/droid/tracks",
                      help="Root directory for Stage 3 track outputs")
  args = parser.parse_args()

  print("🚀 DROID Stage 3: Multi-View Tracking + 3D Fusion Pipeline")
  device = get_accelerator()
  cotracker_model = init_tracking_models()
  pb_renderer = PyBulletRenderer()

  # Discover available episodes from extrinsics output
  ext_abs = os.path.abspath(os.path.expanduser(args.extrinsics_root))
  available_eps = sorted([
      d for d in os.listdir(ext_abs)
      if os.path.isdir(os.path.join(ext_abs, d))
  ])
  random.seed(42)
  random.shuffle(available_eps)
  if args.limit > 0:
    available_eps = available_eps[:args.limit]
  target_eps = available_eps[args.rank::args.world_size]
  print(f"📋 Rank {args.rank}/{args.world_size}: "
        f"{len(target_eps)} episodes to process")

  succeeded_eps = []

  for idx, ep_id in enumerate(target_eps):
    try:
      n_tracks = process_episode(
          ep_id, cotracker_model, pb_renderer, device,
          args.depth_root, args.extrinsics_root, args.export_root)
      succeeded_eps.append(ep_id)
    except Exception as e:
      print(f"  ❌ Episode {ep_id} failed: {e}")
      import traceback
      traceback.print_exc()
      continue

  # Multi-process safe append
  tracks_list = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "episodes_tracks.txt")
  if succeeded_eps:
    batch = "".join(ep_id + "\n" for ep_id in succeeded_eps)
    with open(tracks_list, "a") as f:
      fcntl.flock(f, fcntl.LOCK_EX)
      f.write(batch)
      fcntl.flock(f, fcntl.LOCK_UN)
    print(f"\n📝 Appended {len(succeeded_eps)} episodes to {tracks_list}")

  print(f"\n🎉 Stage 3 complete! "
        f"{len(succeeded_eps)}/{len(target_eps)} episodes succeeded.")
