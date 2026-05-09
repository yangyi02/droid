"""Camera calibration — VGGT, Stage 2 robot alignment, Stage 3 joint alignment.

Test:
  python -c "from droid.calibration import init_camera_states; print('✅ calibration OK')"
"""

import copy
import os

import cv2
import numpy as np
import pybullet as p
import torch
import torch.nn.functional as F
import torch.optim as optim

from core.geometry import axis_angle_to_matrix, make_4x4, unproject_points_np


# ==========================================
# Camera State Initialization
# ==========================================

def init_camera_states(scene_constants, extrinsics_db):
    """Initialize multi-camera extrinsic states from metadata.

    Returns:
        scene_state: dict mapping cam_id -> {base_extrinsic, extrinsics}
    """
    print("  🌐 Initializing camera extrinsic states...")
    wrist_serial = scene_constants['meta']['wrist_serial']
    robot_data = scene_constants['robot']
    n_frames = len(robot_data['T_ee_base_all'])
    episode_id = scene_constants['meta']['episode_id']
    episode_extrinsics = extrinsics_db.get(episode_id, {})

    scene_state = {}
    for cam_id in scene_constants['camera'].keys():
        if cam_id == wrist_serial:
            base_ext = robot_data['T_cam_ee_init']
            cam_trajectory = robot_data['T_ee_base_all'] @ base_ext
        elif cam_id in episode_extrinsics:
            ext_data = episode_extrinsics[cam_id]
            ext_vec = (ext_data.get('extrinsics', ext_data)
                       if isinstance(ext_data, dict) else ext_data)
            base_ext = make_4x4(ext_vec)
            cam_trajectory = np.tile(base_ext, (n_frames, 1, 1))
        else:
            print(f"    ⚠️ No extrinsics for camera [{cam_id}], setting to None.")
            base_ext = None
            cam_trajectory = None

        scene_state[cam_id] = {
            'base_extrinsic': base_ext,
            'extrinsics': cam_trajectory,
        }
    return scene_state


# ==========================================
# VGGT Visual Extrinsics
# ==========================================

def _estimate_multi_camera_vggt(vggt_model, img_list, device):
    """Use VGGT to estimate relative camera poses from images."""
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)

    filenames = []
    for i, img in enumerate(img_list):
        fname = f"/tmp/tmp_vggt_{i}.png"
        cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        filenames.append(fname)

    images = load_and_preprocess_images(filenames).to(device)
    images_input = images.unsqueeze(0)

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=dtype):
            aggregated_tokens_list, ps_idx = vggt_model.aggregator(images_input)
            pose_enc = vggt_model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images_input.shape[-2:]
            )

    T_ref_to_tgts = []
    for i in range(1, len(img_list)):
        ext_mat = extrinsic[0, i].cpu().numpy()
        T = np.eye(4)
        T[:3, :] = ext_mat
        T_ref_to_tgts.append(T)

    # Cleanup
    for fname in filenames:
        if os.path.exists(fname):
            os.remove(fname)

    return T_ref_to_tgts


def vggt_warmup_extrinsics(scene_constants, vggt_model, device):
    """Use VGGT to estimate camera extrinsics from first frame.

    Used as fallback when pre-calibrated extrinsics are missing.
    """
    print("  🌍 Running VGGT visual extrinsic estimation...")
    wrist_serial = scene_constants['meta']['wrist_serial']
    ext_cams = [cam for cam in scene_constants['camera'].keys()
                if cam != wrist_serial]
    ref_cam = ext_cams[0]
    other_cams = ext_cams[1:]

    robot_data = scene_constants['robot']
    n_frames = len(scene_constants['camera'][ref_cam]['video_rgb'])

    new_scene_state = {}
    new_scene_state[wrist_serial] = {
        'base_extrinsic': robot_data['T_cam_ee_init'],
        'extrinsics': robot_data['T_ee_base_all'] @ robot_data['T_cam_ee_init'],
    }

    img_list = (
        [scene_constants['camera'][ref_cam]['video_rgb'][0]]
        + [scene_constants['camera'][cam]['video_rgb'][0] for cam in other_cams]
        + [scene_constants['camera'][wrist_serial]['video_rgb'][0]]
    )

    T_rel_list = _estimate_multi_camera_vggt(vggt_model, img_list, device)
    T_ref_to_others = T_rel_list[:-1]
    T_ref_to_wrist = T_rel_list[-1]

    T_wrist_to_base_first = (robot_data['T_ee_base_all'][0]
                             @ robot_data['T_cam_ee_init'])
    T_ref_to_base = T_wrist_to_base_first @ T_ref_to_wrist

    new_scene_state[ref_cam] = {
        'base_extrinsic': T_ref_to_base,
        'extrinsics': np.tile(T_ref_to_base, (n_frames, 1, 1)),
    }

    for tgt_cam, T_ref_to_tgt in zip(other_cams, T_ref_to_others):
        T_tgt_to_base = T_ref_to_base @ np.linalg.inv(T_ref_to_tgt)
        new_scene_state[tgt_cam] = {
            'base_extrinsic': T_tgt_to_base,
            'extrinsics': np.tile(T_tgt_to_base, (n_frames, 1, 1)),
        }

    return new_scene_state


# ==========================================
# GPU Loss Functions
# ==========================================

def _make_T_from_delta(delta, device):
    """Build 4x4 transform from 6D delta (rot3 + trans3)."""
    T = torch.eye(4, device=device)
    T[:3, :3] = axis_angle_to_matrix(delta[:3])
    T[:3, 3] = delta[3:]
    return T


def _get_foreground_robot_points(T_init, K, obs_depth, pb_renderer, max_pts, device):
    """Extract robot surface point cloud from rendered depth."""
    h_img, w_img = obs_depth.shape
    render_d = pb_renderer.render_depth(T_init, K, w_img, h_img)
    v_r, u_r = np.where(render_d > 0)
    z_r = render_d[v_r, u_r]
    if len(z_r) < max_pts:
        return None
    P_cam_r = np.stack([
        (u_r - K[0, 2]) * z_r / K[0, 0],
        (v_r - K[1, 2]) * z_r / K[1, 1],
        z_r, np.ones_like(z_r)
    ])
    pts = (T_init @ P_cam_r)[:3, :].T
    idx = np.random.choice(len(pts), max_pts, replace=(len(pts) < max_pts))
    return torch.tensor(pts[idx], dtype=torch.float32, device=device)


def _get_foreground_gripper_points(T_cam_world, K, obs_depth, pb_renderer, max_pts):
    """Extract only gripper surface points (not arm links)."""
    h_img, w_img = obs_depth.shape
    cam_pos = T_cam_world[:3, 3]
    target_pos = cam_pos + T_cam_world[:3, 2]
    view_matrix = p.computeViewMatrix(cam_pos, target_pos, -T_cam_world[:3, 1])
    proj_matrix = pb_renderer._get_projection_matrix(K, w_img, h_img)
    _, _, _, depth_buffer, seg_buffer = p.getCameraImage(
        w_img, h_img, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
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
        z_r, np.ones_like(z_r)
    ])
    idx = np.random.choice(P_cam_r.shape[1], max_pts,
                           replace=(P_cam_r.shape[1] < max_pts))
    return P_cam_r[:, idx]


def _compute_robot_loss_batched(batch_X, T_opt, K, batch_obs, device):
    """GPU batched depth alignment loss for robot points."""
    B, _, h_img, w_img = batch_obs.shape
    P_c = (batch_X - T_opt[:3, 3]) @ T_opt[:3, :3]
    Z_pred = P_c[..., 2]
    u = K[0, 0] * P_c[..., 0] / Z_pred + K[0, 2]
    v = K[1, 1] * P_c[..., 1] / Z_pred + K[1, 2]
    grid = torch.stack([
        (u / (w_img - 1)) * 2 - 1, (v / (h_img - 1)) * 2 - 1
    ], dim=-1).unsqueeze(1)
    Z_obs = F.grid_sample(
        batch_obs, grid, mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(1).squeeze(1)
    valid = (
        (Z_pred > 0.) & (Z_pred < 1.5) & (Z_obs > 0.) & (Z_obs < 1.5) &
        (u >= 0) & (u < w_img - 1) & (v >= 0) & (v < h_img - 1)
    )
    diff = torch.abs(Z_obs[valid] - Z_pred[valid])
    return torch.nan_to_num(diff.mean(), nan=0.0)


def _compute_wrist_loss_batched(batch_P_ee, T_cam_ee_opt, K, batch_obs, device):
    """GPU batched depth alignment loss for wrist gripper points."""
    B, _, h_img, w_img = batch_obs.shape
    T_ee_cam = torch.linalg.inv(T_cam_ee_opt)
    P_c = batch_P_ee @ T_ee_cam[:3, :3].T + T_ee_cam[:3, 3]
    Z_pred = P_c[..., 2]
    u = K[0, 0] * P_c[..., 0] / Z_pred + K[0, 2]
    v = K[1, 1] * P_c[..., 1] / Z_pred + K[1, 2]
    grid = torch.stack([
        (u / (w_img - 1)) * 2 - 1, (v / (h_img - 1)) * 2 - 1
    ], dim=-1).unsqueeze(1)
    Z_obs = F.grid_sample(
        batch_obs, grid, mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(1).squeeze(1)
    valid = (
        (Z_pred > 0.) & (Z_pred < 1.5) & (Z_obs > 0.) & (Z_obs < 1.5) &
        (u >= 0) & (u < w_img - 1) & (v >= 0) & (v < h_img - 1)
    )
    diff = torch.abs(Z_obs[valid] - Z_pred[valid])
    return torch.nan_to_num(diff.mean(), nan=0.0)


# ==========================================
# Stage 2: External Camera - Robot Alignment
# ==========================================

def run_stage2_robot_alignment(scene_constants, init_scene_state, pb_renderer, device):
    """Align external cameras to robot body via depth matching."""
    OUTER_LOOPS, INNER_LOOPS, MAX_ROBOT_PTS = 5, 100, 2000
    print("  🦾 Stage 2: External camera-robot alignment...")

    ext_cams = [c for c in scene_constants['camera'].keys()
                if c != scene_constants['meta']['wrist_serial']]
    n_frames = len(scene_constants['camera'][ext_cams[0]]['video_rgb'])

    pybullet_scene_state = copy.deepcopy(init_scene_state)

    for cam in ext_cams:
        print(f"    📷 Optimizing [{cam}]...")
        T_init_t = torch.tensor(
            init_scene_state[cam]['base_extrinsic'],
            dtype=torch.float32, device=device
        )
        K_t = torch.tensor(
            scene_constants['camera'][cam]['K_mat'],
            dtype=torch.float32, device=device
        )
        K_np = scene_constants['camera'][cam]['K_mat']

        d_ext = torch.zeros(6, requires_grad=True, device=device)
        optimizer = optim.Adam([d_ext], lr=0.001)

        for outer_step in range(OUTER_LOOPS):
            with torch.no_grad():
                T_cur_np = (T_init_t @ _make_T_from_delta(d_ext, device)).cpu().numpy()

            cache_X, cache_obs = [], []
            for t in range(n_frames):
                pb_renderer.update_robot_pose(
                    scene_constants['robot']['joint_positions'][t]
                )
                d_obs = scene_constants['camera'][cam]['raw_depth'][t].astype(np.float32)
                r_pts = _get_foreground_robot_points(
                    T_cur_np, K_np, d_obs, pb_renderer, MAX_ROBOT_PTS, device
                )
                if r_pts is not None:
                    cache_X.append(r_pts)
                    cache_obs.append(
                        torch.tensor(d_obs, dtype=torch.float32, device=device)[None, ...]
                    )

            if not cache_X:
                continue

            batch_X = torch.stack(cache_X)
            batch_obs = torch.stack(cache_obs)

            for _ in range(INNER_LOOPS):
                optimizer.zero_grad()
                loss = _compute_robot_loss_batched(
                    batch_X, T_init_t @ _make_T_from_delta(d_ext, device),
                    K_t, batch_obs, device
                )
                loss.backward()
                optimizer.step()

            print(f"      Outer {outer_step+1}/{OUTER_LOOPS} | Loss: {loss.item():.4f}")

        with torch.no_grad():
            T_final = (T_init_t @ _make_T_from_delta(d_ext, device)).cpu().numpy()
            pybullet_scene_state[cam]['base_extrinsic'] = T_final
            pybullet_scene_state[cam]['extrinsics'] = np.tile(T_final, (n_frames, 1, 1))

    return pybullet_scene_state


# ==========================================
# Stage 3: Joint Unified Alignment
# ==========================================

def run_stage3_joint_alignment(scene_constants, stage2_scene_state, pb_renderer, device):
    """Joint environment stitching + robot + wrist alignment."""
    print("  🌍 Stage 3: Joint unified alignment...")

    camera_ids = list(scene_constants['camera'].keys())
    wrist_cam = scene_constants['meta']['wrist_serial']
    ext_cams = [c for c in camera_ids if c != wrist_cam]
    cam1, cam2 = ext_cams[0], ext_cams[1]
    n_frames = len(scene_constants['camera'][cam1]['video_rgb'])

    def to_t(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    T_ee_all = scene_constants['robot']['T_ee_base_all']
    T_cam_ee_init = stage2_scene_state[wrist_cam]['base_extrinsic']
    init_p1 = stage2_scene_state[cam1]['base_extrinsic']
    init_p2 = stage2_scene_state[cam2]['base_extrinsic']

    K_np1 = scene_constants['camera'][cam1]['K_mat']
    K_np2 = scene_constants['camera'][cam2]['K_mat']
    K_np_w = scene_constants['camera'][wrist_cam]['K_mat']
    K_t1, K_t2, K_t_w = to_t(K_np1), to_t(K_np2), to_t(K_np_w)

    # ---- Pre-compute point clouds ----
    def get_cam_points_t(t, cam_data):
        depth = cam_data['raw_depth'][t].astype(np.float32)
        K = cam_data['K_mat']
        valid = (depth > 0.) & (depth < 1.5)
        vs, us = np.where(valid)
        if len(us) < 100:
            return None
        zs = depth[vs, us]
        P = np.stack([
            (us - K[0, 2]) * zs / K[0, 0],
            (vs - K[1, 2]) * zs / K[1, 1],
            zs, np.ones_like(zs)
        ], axis=0)
        n_sample = min(P.shape[1], 2000)
        idx = np.random.choice(P.shape[1], n_sample, replace=(P.shape[1] < n_sample))
        return torch.tensor(P[:, idx], dtype=torch.float32, device=device)

    cache_P1, cache_P2, cache_Pw, cache_Tee = [], [], [], []
    cache_X1, cache_obs1, cache_X2, cache_obs2 = [], [], [], []
    cache_P_ee, cache_obs_w = [], []

    for t in range(n_frames):
        p1 = get_cam_points_t(t, scene_constants['camera'][cam1])
        p2 = get_cam_points_t(t, scene_constants['camera'][cam2])
        pw = get_cam_points_t(t, scene_constants['camera'][wrist_cam])
        if p1 is not None and p2 is not None and pw is not None:
            cache_P1.append(p1)
            cache_P2.append(p2)
            cache_Pw.append(pw)
            cache_Tee.append(to_t(T_ee_all[t]))

        pb_renderer.update_robot_pose(
            scene_constants['robot']['joint_positions'][t],
            gripper_state=scene_constants['robot']['gripper_positions'][t]
        )

        d1 = scene_constants['camera'][cam1]['raw_depth'][t].astype(np.float32)
        r1 = _get_foreground_robot_points(init_p1, K_np1, d1, pb_renderer, 2000, device)
        if r1 is not None:
            cache_X1.append(r1)
            cache_obs1.append(to_t(d1)[None, ...])

        d2 = scene_constants['camera'][cam2]['raw_depth'][t].astype(np.float32)
        r2 = _get_foreground_robot_points(init_p2, K_np2, d2, pb_renderer, 2000, device)
        if r2 is not None:
            cache_X2.append(r2)
            cache_obs2.append(to_t(d2)[None, ...])

        T_cam_world_np = T_ee_all[t] @ T_cam_ee_init
        dw = scene_constants['camera'][wrist_cam]['raw_depth'][t].astype(np.float32)
        P_r = _get_foreground_gripper_points(T_cam_world_np, K_np_w, dw, pb_renderer, 2000)
        if P_r is not None:
            P_ee = (T_cam_ee_init @ P_r)[:3, :].T
            cache_P_ee.append(to_t(P_ee))
            cache_obs_w.append(to_t(dw)[None, ...])

    batch_P1 = torch.stack(cache_P1)
    batch_P2 = torch.stack(cache_P2)
    batch_Pw = torch.stack(cache_Pw)
    batch_Tee = torch.stack(cache_Tee)
    batch_X1 = torch.stack(cache_X1)
    batch_obs1 = torch.stack(cache_obs1)
    batch_X2 = torch.stack(cache_X2)
    batch_obs2 = torch.stack(cache_obs2)
    batch_P_ee = torch.stack(cache_P_ee) if cache_P_ee else None
    batch_obs_w = torch.stack(cache_obs_w) if cache_obs_w else None

    # ---- Optimization ----
    d1_param = torch.zeros(6, requires_grad=True, device=device)
    d2_param = torch.zeros(6, requires_grad=True, device=device)
    dhe_param = torch.zeros(6, requires_grad=True, device=device)
    optimizer = optim.Adam([d1_param, d2_param, dhe_param], lr=0.001)

    T1_init_t, T2_init_t, Tee_init_t = to_t(init_p1), to_t(init_p2), to_t(T_cam_ee_init)

    def batched_chamfer(p1, p2):
        dist = torch.cdist(p1, p2)
        m12 = torch.min(dist, dim=2)[0]
        m21 = torch.min(dist, dim=1)[0]
        v12 = m12 < 0.05
        v21 = m21 < 0.05
        loss = torch.tensor(0.0, device=device)
        if v12.any():
            loss += m12[v12].mean()
        if v21.any():
            loss += m21[v21].mean()
        return loss

    for step in range(500):
        optimizer.zero_grad()

        T1_opt = T1_init_t @ _make_T_from_delta(d1_param, device)
        T2_opt = T2_init_t @ _make_T_from_delta(d2_param, device)
        T_wrist = batch_Tee @ (Tee_init_t @ _make_T_from_delta(dhe_param, device))

        bc1 = (T1_opt @ batch_P1)[:, :3, :].transpose(1, 2)
        bc2 = (T2_opt @ batch_P2)[:, :3, :].transpose(1, 2)
        bcw = torch.bmm(T_wrist, batch_Pw)[:, :3, :].transpose(1, 2)

        l_ch = batched_chamfer(bc1, bc2) + batched_chamfer(bc1, bcw) + batched_chamfer(bc2, bcw)
        l_r1 = _compute_robot_loss_batched(batch_X1, T1_opt, K_t1, batch_obs1, device)
        l_r2 = _compute_robot_loss_batched(batch_X2, T2_opt, K_t2, batch_obs2, device)
        l_w = torch.tensor(0.0, device=device)
        if batch_P_ee is not None:
            l_w = _compute_wrist_loss_batched(
                batch_P_ee, Tee_init_t @ _make_T_from_delta(dhe_param, device),
                K_t_w, batch_obs_w, device
            )

        loss = l_ch + 1.0 * (l_r1 + l_r2 + l_w)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 499:
            print(f"    Step {step:03d} | Chamfer: {l_ch.item():.4f} | "
                  f"Rob1: {l_r1.item():.4f} | Rob2: {l_r2.item():.4f} | "
                  f"Wrist: {l_w.item():.4f}")

    with torch.no_grad():
        final_p1 = (T1_init_t @ _make_T_from_delta(d1_param, device)).cpu().numpy()
        final_p2 = (T2_init_t @ _make_T_from_delta(d2_param, device)).cpu().numpy()
        final_cam_ee = (Tee_init_t @ _make_T_from_delta(dhe_param, device)).cpu().numpy()

    ultimate = {}
    ultimate[cam1] = {'base_extrinsic': final_p1, 'extrinsics': np.tile(final_p1, (n_frames, 1, 1))}
    ultimate[cam2] = {'base_extrinsic': final_p2, 'extrinsics': np.tile(final_p2, (n_frames, 1, 1))}
    ultimate[wrist_cam] = {'base_extrinsic': final_cam_ee, 'extrinsics': T_ee_all @ final_cam_ee}

    return ultimate
