"""2D/3D tracking — CoTracker, URDF kinematics, consensus voting.

Test:
  python -c "from droid.tracking import extract_2d_tracks; print('✅ tracking OK')"
"""

import warnings

import cv2
import numpy as np
import pybullet as p
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from core.geometry import project_points_np, unproject_points_np


# ==========================================
# CoTracker Dense 2D Tracking
# ==========================================

def extract_2d_tracks(cotracker_model, scene_constants, device):
    """Run CoTracker3 dense 2D tracking on all camera views.

    Adds 'tracks_2d' and 'vis_2d' keys to each camera's data dict.
    """
    print("  🎯 Running CoTracker3 dense tracking...")
    for cam_id in scene_constants['camera']:
        cam_data = scene_constants['camera'][cam_id]
        video_tensor = (
            torch.from_numpy(cam_data['video_rgb'])
            .permute(0, 3, 1, 2)[None].float().to(device)
        )
        with torch.no_grad():
            pred_tracks, pred_vis = cotracker_model(
                video_tensor, grid_size=30, grid_query_frame=0,
                backward_tracking=False
            )
        cam_data.update({
            'tracks_2d': pred_tracks[0].cpu().numpy(),
            'vis_2d': pred_vis[0].cpu().numpy(),
        })
    return scene_constants


# ==========================================
# URDF Kinematics Tracker (Track B)
# ==========================================

class URDFKinematicsTracker:
    """Forward kinematics-based robot 3D trajectory generator.

    Binds seed 2D tracking points to URDF link frames at t=0,
    then propagates them via forward kinematics across all frames.
    """

    def __init__(self, pb_renderer):
        self.pb_renderer = pb_renderer

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
        """Extract 3D robot surface trajectories via URDF kinematics.

        Args:
            src_cam: source camera ID
            scene_constants: data dict
            scene_state: extrinsics dict
            safe_margin: erosion kernel size for edge safety

        Returns:
            traj_3d: (T, N_robot, 3) world coordinates
            traj_2d: (T, N_robot, 2) pixel coordinates
            vis_2d: (T, N_robot) visibility mask
            robot_indices: indices into the original tracks_2d
        """
        print(f"    🦾 URDF tracking [{src_cam}]")
        src_data = scene_constants['camera'][src_cam]
        src_state = scene_state[src_cam]
        K_mat = src_data['K_mat']
        extrinsics = src_state['extrinsics']
        h_img, w_img = src_data['video_rgb'][0].shape[:2]
        n_frames = len(src_data['video_rgb'])
        tracks_2d_t0 = src_data['tracks_2d'][0]

        # ---- Frame 0: find seed points on robot ----
        self.pb_renderer.update_robot_pose(
            scene_constants['robot']['joint_positions'][0],
            gripper_state=scene_constants['robot']['gripper_positions'][0]
        )

        obj_ids, link_ids, urdf_depth = self.pb_renderer.render_segmentation(
            extrinsics[0], K_mat, w_img, h_img
        )
        is_robot = (
            (obj_ids == self.pb_renderer.robot_id) |
            (obj_ids == self.pb_renderer.ghost_id)
        )

        # Erode mask to avoid edge artifacts
        kernel = np.ones((safe_margin, safe_margin), np.uint8)
        is_robot_safe = cv2.erode(
            is_robot.astype(np.uint8), kernel, iterations=1
        ) > 0

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
            z0, K_mat, extrinsics[0]
        )

        unique_parts = set(zip(robot_objs, robot_links))
        local_pts_dict = {}
        for oid, lid in unique_parts:
            mask = (robot_objs == oid) & (robot_links == lid)
            pts = pts_world_t0[mask]
            T_link = self._get_link_transform(oid, lid)
            T_inv = np.linalg.inv(T_link)
            P_homo = np.hstack([pts, np.ones((len(pts), 1))]).T
            local_pts_dict[(oid, lid)] = (mask, T_inv @ P_homo)

        # ---- Forward kinematics propagation ----
        traj_3d = np.zeros((n_frames, len(robot_indices), 3), dtype=np.float32)
        traj_2d = np.zeros((n_frames, len(robot_indices), 2), dtype=np.float32)
        vis_2d = np.zeros((n_frames, len(robot_indices)), dtype=bool)

        for t in range(n_frames):
            self.pb_renderer.update_robot_pose(
                scene_constants['robot']['joint_positions'][t],
                gripper_state=scene_constants['robot']['gripper_positions'][t]
            )

            for oid, lid in unique_parts:
                mask, P_local = local_pts_dict[(oid, lid)]
                T_link_t = self._get_link_transform(oid, lid)
                P_world_t = T_link_t @ P_local
                traj_3d[t, mask, :] = P_world_t[:3, :].T

            u_t, v_t, z_pred = project_points_np(traj_3d[t], K_mat, extrinsics[t])
            traj_2d[t, :, 0] = u_t
            traj_2d[t, :, 1] = v_t

            # Visibility: bounds + self-occlusion + env-occlusion
            urdf_depth_t = self.pb_renderer.render_depth(extrinsics[t], K_mat, w_img, h_img)
            raw_depth_t = src_data['raw_depth'][t]

            ui = np.clip(np.round(u_t).astype(int), 0, w_img - 1)
            vi = np.clip(np.round(v_t).astype(int), 0, h_img - 1)

            in_bounds = (u_t >= 0) & (u_t < w_img) & (v_t >= 0) & (v_t < h_img) & (z_pred > 0)
            z_urdf = urdf_depth_t[vi, ui]
            not_self_occ = (z_urdf > 0) & (z_pred <= z_urdf + 0.015)
            z_sensor = raw_depth_t[vi, ui]
            not_env_occ = ~((z_sensor > 0) & (z_pred > z_sensor + 0.02))
            vis_2d[t] = in_bounds & not_self_occ & not_env_occ

        return traj_3d, traj_2d, vis_2d, robot_indices


# ==========================================
# Consensus Visual Tracker (Track A)
# ==========================================

class ConsensusVisualTracker:
    """15-vote multi-view spatio-temporal consensus tracker for environment points.

    For each source camera, tracks environment points across all cameras
    at 5 keyframes, then fuses via NaN-median voting and Gaussian smoothing.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def extract_env_tracks(self, src_cam, env_indices, scene_constants,
                           scene_state, max_env_depth=5.0, occ_margin=0.10):
        """Extract consensus-filtered 3D environment point trajectories.

        Args:
            src_cam: source camera ID
            env_indices: indices of environment points in tracks_2d
            scene_constants: data dict
            scene_state: extrinsics dict

        Returns:
            smoothed_3d: (T, N_env, 3) smoothed world coordinates
            refined_2d: (T, N_env, 3) refined pixel coords + depth
            unified_vis: (T, N_env) consensus visibility mask
        """
        print(f"    👁️ Consensus tracking [{src_cam}] ({len(env_indices)} pts)")

        src_data = scene_constants['camera'][src_cam]
        src_state = scene_state[src_cam]
        T_total = len(src_data['video_rgb'])
        N_env = len(env_indices)

        keyframes = [0, T_total // 4, T_total // 2, 3 * T_total // 4, T_total - 1]
        camera_ids = list(scene_constants['camera'].keys())

        src_tracks = src_data['tracks_2d'][:, env_indices, :]
        src_vis = src_data['vis_2d'][:, env_indices]

        vote_traj_dict, vote_vis_dict = {}, {}

        for tgt_cam in camera_ids:
            tgt_data = scene_constants['camera'][tgt_cam]
            tgt_state = scene_state[tgt_cam]
            h_img, w_img = tgt_data['video_rgb'][0].shape[:2]

            video_tensor = (
                torch.from_numpy(tgt_data['video_rgb'])
                .permute(0, 3, 1, 2)[None].float().to(self.device)
            )

            for t_k in keyframes:
                vote_name = f"{tgt_cam}_kf{t_k}"

                # Project source points to target at keyframe
                u_src = src_tracks[t_k, :, 0]
                v_src = src_tracks[t_k, :, 1]
                vis_src = src_vis[t_k]
                ui_src = np.clip(np.round(u_src).astype(int), 0, w_img - 1)
                vi_src = np.clip(np.round(v_src).astype(int), 0, h_img - 1)
                z_src = src_data['raw_depth'][t_k, vi_src, ui_src]
                valid_src = vis_src & (z_src > 0.05) & (z_src < max_env_depth)

                pts_3d = unproject_points_np(
                    u_src, v_src, z_src, src_data['K_mat'],
                    src_state['extrinsics'][t_k]
                )
                u_tgt, v_tgt, z_pred = project_points_np(
                    pts_3d, tgt_data['K_mat'], tgt_state['extrinsics'][t_k]
                )
                ui_tgt = np.clip(np.round(u_tgt).astype(int), 0, w_img - 1)
                vi_tgt = np.clip(np.round(v_tgt).astype(int), 0, h_img - 1)
                z_sensor = tgt_data['raw_depth'][t_k, vi_tgt, ui_tgt]

                in_bounds = ((u_tgt >= 0) & (u_tgt < w_img) &
                             (v_tgt >= 0) & (v_tgt < h_img))
                is_visible = (z_sensor > 0) & (z_sensor >= z_pred - occ_margin)
                valid_mask = valid_src & in_bounds & is_visible
                valid_idx = np.where(valid_mask)[0]

                if len(valid_idx) == 0:
                    vote_traj_dict[vote_name] = np.zeros((T_total, N_env, 3), dtype=np.float32)
                    vote_vis_dict[vote_name] = np.zeros((T_total, N_env), dtype=bool)
                    continue

                # Run CoTracker with bidirectional tracking
                queries_np = np.stack([
                    np.full_like(u_tgt[valid_idx], t_k),
                    u_tgt[valid_idx],
                    v_tgt[valid_idx]
                ], axis=-1)
                queries_t = torch.tensor(
                    queries_np, dtype=torch.float32, device=self.device
                )[None, ...]

                with torch.no_grad():
                    pred_tracks, pred_vis = self.model(
                        video_tensor, queries=queries_t, backward_tracking=True
                    )

                tgt_tracks_N = np.zeros((T_total, N_env, 2), dtype=np.float32)
                tgt_vis_N = np.zeros((T_total, N_env), dtype=bool)
                tgt_tracks_N[:, valid_idx, :] = pred_tracks[0].cpu().numpy()
                tgt_vis_N[:, valid_idx] = pred_vis[0].cpu().numpy()

                # Lift tracked points to 3D
                traj_3d_K = np.zeros((T_total, N_env, 3), dtype=np.float32)
                vis_3d_K = np.zeros((T_total, N_env), dtype=bool)

                for t in range(T_total):
                    u_t = tgt_tracks_N[t, :, 0]
                    v_t = tgt_tracks_N[t, :, 1]
                    vis_t = tgt_vis_N[t]
                    ui_t = np.clip(np.round(u_t).astype(int), 0, w_img - 1)
                    vi_t = np.clip(np.round(v_t).astype(int), 0, h_img - 1)
                    z_t = tgt_data['raw_depth'][t, vi_t, ui_t]
                    valid_t = (vis_t & (z_t > 0.05) & (z_t < max_env_depth) &
                               (u_t >= 0) & (u_t < w_img) &
                               (v_t >= 0) & (v_t < h_img))
                    vis_3d_K[t] = valid_t
                    if valid_t.any():
                        traj_3d_K[t, valid_t] = unproject_points_np(
                            u_t[valid_t], v_t[valid_t], z_t[valid_t],
                            tgt_data['K_mat'], tgt_state['extrinsics'][t]
                        )

                vote_traj_dict[vote_name] = traj_3d_K
                vote_vis_dict[vote_name] = vis_3d_K

        # ---- Fusion: NaN-median + Gaussian smoothing ----
        stacked_traj = np.stack(list(vote_traj_dict.values()), axis=0)
        stacked_vis = np.stack(list(vote_vis_dict.values()), axis=0)
        stacked_traj_nan = np.where(stacked_vis[..., None], stacked_traj, np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            unified_traj_3d = np.nanmedian(stacked_traj_nan, axis=0)

        view_counts = np.sum(stacked_vis, axis=0)
        unified_vis = view_counts >= 3

        # Interpolate + smooth
        import pandas as pd
        smoothed = unified_traj_3d.copy()
        df = pd.DataFrame(smoothed.reshape(T_total, -1))
        df = df.interpolate(method='linear', limit_direction='both')
        interpolated = df.to_numpy().reshape(T_total, N_env, 3)
        smoothed = gaussian_filter1d(interpolated, sigma=2.0, axis=0)
        smoothed[~unified_vis] = np.nan

        # Re-project to source camera 2D
        refined_2d = np.zeros((T_total, N_env, 3), dtype=np.float32)
        for t in range(T_total):
            u_r, v_r, z_r = project_points_np(
                smoothed[t], src_data['K_mat'], src_state['extrinsics'][t]
            )
            refined_2d[t, :, 0] = u_r
            refined_2d[t, :, 1] = v_r
            refined_2d[t, :, 2] = z_r

        return smoothed, refined_2d, unified_vis
