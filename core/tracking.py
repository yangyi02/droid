import cv2
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

import core.geometry


class URDFKinematicsTracker:
  def __init__(self, pb_renderer):
    self.pb = pb_renderer
    self._urdf_depth_cache = {}

  def _get_link_transform(self, obj_id, link_id):
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
    print(f"    URDF tracking [{src_cam}]")
    src_data = scene_constants["camera"][src_cam]
    src_state = scene_state[src_cam]
    K_mat = src_data["K_mat"]
    extrinsics = src_state["extrinsics"]
    h_img, w_img = src_data["video_rgb"][0].shape[:2]
    n_frames = len(src_data["video_rgb"])

    ys = np.arange(h_img, dtype=np.float32)
    xs = np.arange(w_img, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    seed_pts_2d = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    self.pb.update_robot_pose(
        scene_constants["robot"]["joint_positions"][0],
        gripper_state=scene_constants["robot"]["gripper_positions"][0])

    obj_ids, link_ids, urdf_depth = self.pb.render_segmentation(
        extrinsics[0], K_mat, w_img, h_img)
    is_robot = ((obj_ids == self.pb.robot_id) |
                (obj_ids == self.pb.ghost_id))

    kernel = np.ones((safe_margin, safe_margin), np.uint8)
    is_robot_safe = cv2.erode(
        is_robot.astype(np.uint8), kernel, iterations=1) > 0

    u0 = np.clip(np.round(seed_pts_2d[:, 0]).astype(int), 0, w_img - 1)
    v0 = np.clip(np.round(seed_pts_2d[:, 1]).astype(int), 0, h_img - 1)
    robot_indices = np.where(is_robot_safe[v0, u0])[0]

    if max_robot_pts is not None and len(robot_indices) > max_robot_pts:
      rng = np.random.default_rng(42)
      robot_indices = rng.choice(robot_indices, max_robot_pts, replace=False)
      robot_indices = np.sort(robot_indices)

    print(f"      Found {len(robot_indices)} robot surface points.")

    robot_objs = obj_ids[v0[robot_indices], u0[robot_indices]]
    robot_links = link_ids[v0[robot_indices], u0[robot_indices]]
    z0 = urdf_depth[v0[robot_indices], u0[robot_indices]]

    pts_world_t0 = core.geometry.unproject_points(
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

    traj_3d = np.zeros((n_frames, len(robot_indices), 3), dtype=np.float32)
    traj_2d = np.zeros((n_frames, len(robot_indices), 2), dtype=np.float32)
    vis_2d = np.zeros((n_frames, len(robot_indices)), dtype=bool)

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

      u_t, v_t, z_pred = core.geometry.project_points(
          traj_3d[t], K_mat, extrinsics[t])
      traj_2d[t, :, 0] = u_t
      traj_2d[t, :, 1] = v_t

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

    self._urdf_depth_cache.update(urdf_depth_cache)

    return traj_3d, traj_2d, vis_2d, robot_indices

  def project_to_all_views(self, traj_3d, scene_constants, scene_state):
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
        cache_key = (cam_id, t)
        if cache_key not in cache:
          self.pb.update_robot_pose(
              scene_constants["robot"]["joint_positions"][t],
              gripper_state=scene_constants["robot"]["gripper_positions"][t])
          cache[cache_key] = self.pb.render_depth(
              cam_state["extrinsics"][t], K, w_img, h_img)

        urdf_depth_t = cache[cache_key]

        u_t, v_t, z_pred = core.geometry.project_points(
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
