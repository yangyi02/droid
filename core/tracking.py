"""URDF-based robot tracking via forward kinematics.

URDFKinematicsTracker generates grid seed points on the robot surface,
binds them to URDF link frames at t=0, then propagates across all frames
using PyBullet FK. No external tracker model is required.
"""

import cv2
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from core.geometry import project_points, unproject_points


class URDFKinematicsTracker:
  """Forward kinematics-based robot 3D trajectory generator.

  Generates grid seed points at t=0, identifies which fall on the robot
  surface, binds them to URDF link frames, then propagates via forward
  kinematics across all frames. No external tracker model is needed.
  """

  def __init__(self, pb_renderer):
    self.pb = pb_renderer
    self._urdf_depth_cache = {}

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
                           safe_margin=7, max_robot_pts=None):
    """Extract 3D robot surface trajectories via URDF forward kinematics.

    Uses ALL pixels at frame 0 as seed candidates, identifies which fall
    on the robot surface (via eroded robot mask), and propagates them via
    FK. No grid subsampling — dense coverage.

    Args:
      src_cam: Source camera serial number.
      scene_constants: Scene data dict.
      scene_state: Extrinsics dict.
      safe_margin: Robot mask erosion kernel size.

    Returns:
      traj_3d: (T, N_robot, 3) world coordinates
      traj_2d: (T, N_robot, 2) pixel coordinates in src_cam
      vis_2d:  (T, N_robot) visibility mask in src_cam
      robot_indices: indices into the dense pixel array
    """
    print(f"    URDF tracking [{src_cam}]")
    src_data = scene_constants["camera"][src_cam]
    src_state = scene_state[src_cam]
    K_mat = src_data["K_mat"]
    extrinsics = src_state["extrinsics"]
    h_img, w_img = src_data["video_rgb"][0].shape[:2]
    n_frames = len(src_data["video_rgb"])

    # Dense pixel grid at frame 0 (all pixels)
    ys = np.arange(h_img, dtype=np.float32)
    xs = np.arange(w_img, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    seed_pts_2d = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (H*W, 2)

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

    u0 = np.clip(np.round(seed_pts_2d[:, 0]).astype(int), 0, w_img - 1)
    v0 = np.clip(np.round(seed_pts_2d[:, 1]).astype(int), 0, h_img - 1)
    robot_indices = np.where(is_robot_safe[v0, u0])[0]

    if len(robot_indices) == 0:
      print("      [WARN] No robot points found at t=0.")
      return None, None, None, None

    # Subsample if too many robot points
    if max_robot_pts is not None and len(robot_indices) > max_robot_pts:
      rng = np.random.default_rng(42)
      robot_indices = rng.choice(robot_indices, max_robot_pts, replace=False)
      robot_indices = np.sort(robot_indices)

    print(f"      Found {len(robot_indices)} robot surface points.")

    # Bind to local link frames
    robot_objs = obj_ids[v0[robot_indices], u0[robot_indices]]
    robot_links = link_ids[v0[robot_indices], u0[robot_indices]]
    z0 = urdf_depth[v0[robot_indices], u0[robot_indices]]

    pts_world_t0 = unproject_points(
        seed_pts_2d[robot_indices, 0],
        seed_pts_2d[robot_indices, 1],
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

      u_t, v_t, z_pred = project_points(
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

    # Store cache so project_to_all_views can reuse renders
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

      cache = self._urdf_depth_cache

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

        u_t, v_t, z_pred = project_points(
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
