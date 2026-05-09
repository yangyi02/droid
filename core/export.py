"""TAPVid format export — serialize dual-track 3D ground truth.

Test:
  python -c "from droid.export import extract_and_export_tapvid; print('✅ export OK')"
"""

import os

import cv2
import numpy as np

from core.geometry import project_points_np


def extract_and_export_tapvid(export_root, scene_constants, scene_state,
                               urdf_tracker, consensus_tracker, pb_renderer):
    """Extract dual-track 3D ground truth and serialize to TAPVid format.

    Outputs per episode:
      <export_root>/<episode_id>/
        tracks_xyz.npy       — (T, N, 3) world coordinates
        queries_xytv.npy     — (N, 4) [u, v, t, view_idx]
        <view_idx>/
          images_jpeg_bytes.npy  — (T,) object array of JPEG bytes
          visibility.npy         — (T, N) bool
          intrinsics.npy         — (4,) [fx, fy, cx, cy]
          extrinsics_w2c.npy     — (T, 4, 4) world-to-camera
    """
    print("  📦 Extracting and exporting TAPVid data...")
    episode_id = scene_constants['meta']['episode_id']
    camera_ids = list(scene_constants['camera'].keys())
    T_frames = len(scene_constants['camera'][camera_ids[0]]['video_rgb'])

    all_final_tracks = []
    all_queries = []
    all_visibility = {cam: [] for cam in camera_ids}
    total_survivors = 0

    for v_source_idx, src_cam in enumerate(camera_ids):
        cam_data = scene_constants['camera'][src_cam]
        cam_state = scene_state[src_cam]
        h_img, w_img = cam_data['video_rgb'][0].shape[:2]
        tracks_2d = cam_data['tracks_2d']
        raw_depth = cam_data['raw_depth']

        orig_2d_all, traj_3d_all, vis_all = [], [], []

        # ---- Track B: URDF robot points ----
        traj_3d_rob, proj_2d_rob, vis_rob, robot_indices = \
            urdf_tracker.extract_robot_tracks(src_cam, scene_constants, scene_state)

        if proj_2d_rob is not None and len(robot_indices) > 0:
            orig_2d_all.append(tracks_2d[:, robot_indices, :])
            traj_3d_all.append(traj_3d_rob)
            vis_all.append(vis_rob)

        # ---- Track A: Environment points with consensus ----
        joint_angles_t0 = scene_constants['robot']['joint_positions'][0]
        gripper_state_t0 = scene_constants['robot']['gripper_positions'][0]
        pb_renderer.update_robot_pose(joint_angles_t0, gripper_state=gripper_state_t0)

        robot_mask = pb_renderer.render_mask(
            cam_state['extrinsics'][0], cam_data['K_mat'], w_img, h_img
        ) > 0
        kernel = np.ones((15, 15), np.uint8)
        robot_mask_dilated = cv2.dilate(
            robot_mask.astype(np.uint8), kernel, iterations=1
        ) > 0

        u0 = np.clip(np.round(tracks_2d[0, :, 0]).astype(int), 0, w_img - 1)
        v0 = np.clip(np.round(tracks_2d[0, :, 1]).astype(int), 0, h_img - 1)
        is_env = ~robot_mask_dilated[v0, u0]
        has_depth = (raw_depth[0, v0, u0] > 0.05) & (raw_depth[0, v0, u0] < 5.0)
        env_indices = np.where(is_env & has_depth)[0]

        if len(env_indices) > 0:
            smoothed_3d, refined_2d, unified_vis = \
                consensus_tracker.extract_env_tracks(
                    src_cam, env_indices, scene_constants, scene_state
                )
            orig_2d_env = tracks_2d[:, env_indices, :]

            # Quality filtering
            valid_counts = np.sum(unified_vis, axis=0)
            D_2D = np.linalg.norm(
                orig_2d_env - refined_2d[:, :, :2], axis=-1
            )
            max_drift = np.max(np.where(unified_vis, D_2D, 0), axis=0)
            flicker_counts = np.sum(
                unified_vis[:-1] != unified_vis[1:], axis=0
            )

            survivor_mask = (
                (valid_counts >= 5) &
                (max_drift < 10.0) &
                (flicker_counts <= T_frames * 0.10)
            )
            golden_idx = np.where(survivor_mask)[0]

            if len(golden_idx) > 0:
                traj_3d_env = smoothed_3d[:, golden_idx, :]
                vis_env = unified_vis[:, golden_idx].copy()
                orig_2d_golden = orig_2d_env[:, golden_idx, :]
                proj_2d_env = refined_2d[:, golden_idx, :2]
                z_pred_env = refined_2d[:, golden_idx, 2]

                # X-ray occlusion test
                for t in range(T_frames):
                    ui = np.clip(np.round(proj_2d_env[t, :, 0]).astype(int), 0, w_img - 1)
                    vi = np.clip(np.round(proj_2d_env[t, :, 1]).astype(int), 0, h_img - 1)
                    z_sensor = raw_depth[t, vi, ui]
                    occluded = (z_sensor > 0) & (z_sensor < z_pred_env[t] - 0.05)
                    vis_env[t] = vis_env[t] & (~occluded)

                orig_2d_all.append(orig_2d_golden)
                traj_3d_all.append(traj_3d_env)
                vis_all.append(vis_env)

        if not orig_2d_all:
            continue

        combined_orig = np.concatenate(orig_2d_all, axis=1)
        combined_3d = np.concatenate(traj_3d_all, axis=1)
        combined_vis = np.concatenate(vis_all, axis=1)

        N_survivors = combined_3d.shape[1]
        total_survivors += N_survivors
        all_final_tracks.append(combined_3d)

        queries = np.zeros((N_survivors, 4), dtype=np.float32)
        queries[:, 0] = combined_orig[0, :, 0]  # u
        queries[:, 1] = combined_orig[0, :, 1]  # v
        queries[:, 2] = 0                         # t
        queries[:, 3] = v_source_idx              # view
        all_queries.append(queries)

        # Cross-view visibility
        for tgt_cam in camera_ids:
            if tgt_cam == src_cam:
                all_visibility[tgt_cam].append(combined_vis)
                continue

            tgt_data = scene_constants['camera'][tgt_cam]
            tgt_state = scene_state[tgt_cam]
            tgt_vis = np.zeros((T_frames, N_survivors), dtype=bool)

            for t in range(T_frames):
                valid_3d = ~np.isnan(combined_3d[t, :, 2])
                if valid_3d.any():
                    u_v, v_v, z_p = project_points_np(
                        combined_3d[t, valid_3d],
                        tgt_data['K_mat'],
                        tgt_state['extrinsics'][t]
                    )
                    ui = np.clip(np.round(u_v).astype(int), 0, w_img - 1)
                    vi = np.clip(np.round(v_v).astype(int), 0, h_img - 1)
                    in_bounds = ((u_v >= 0) & (u_v < w_img) &
                                 (v_v >= 0) & (v_v < h_img) & (z_p > 0))
                    z_s = tgt_data['raw_depth'][t, vi, ui]
                    occluded = (z_s > 0) & (z_s < z_p - 0.05)
                    tgt_vis_t = np.zeros(N_survivors, dtype=bool)
                    tgt_vis_t[valid_3d] = in_bounds & (~occluded)
                    tgt_vis[t] = tgt_vis_t

            all_visibility[tgt_cam].append(tgt_vis)

    if total_survivors == 0:
        print(f"  ⚠️ No valid points! Skipping export.")
        return

    # ---- Serialize ----
    final_tracks = np.concatenate(all_final_tracks, axis=1)
    queries_xytv = np.concatenate(all_queries, axis=0)

    seq_dir = os.path.join(export_root, episode_id)
    os.makedirs(seq_dir, exist_ok=True)
    np.save(os.path.join(seq_dir, "tracks_xyz.npy"),
            final_tracks.astype(np.float32))
    np.save(os.path.join(seq_dir, "queries_xytv.npy"), queries_xytv)

    for v_idx, cam in enumerate(camera_ids):
        cam_dir = os.path.join(seq_dir, str(v_idx))
        os.makedirs(cam_dir, exist_ok=True)
        cam_data = scene_constants['camera'][cam]

        jpeg_list = [
            cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()
            for img in cam_data['video_rgb']
        ]
        np.save(os.path.join(cam_dir, "images_jpeg_bytes.npy"),
                np.array(jpeg_list, dtype=object))

        final_vis = np.concatenate(all_visibility[cam], axis=1)
        np.save(os.path.join(cam_dir, "visibility.npy"), final_vis)

        K = cam_data['K_mat']
        np.save(os.path.join(cam_dir, "intrinsics.npy"),
                np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32))
        np.save(os.path.join(cam_dir, "extrinsics_w2c.npy"),
                np.linalg.inv(scene_state[cam]['extrinsics']).astype(np.float32))

    print(f"  🎉 Exported {total_survivors} points to {seq_dir}")
