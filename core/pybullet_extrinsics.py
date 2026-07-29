"""PyBullet-based camera extrinsics calibration for the DROID pipeline.

Alternative to the yourdfpy-based method in compute_extrinsics.py.  Instead of
sampling mesh surfaces with yourdfpy (``TensorRobotRenderer``), this module
drives ``PyBulletRenderer`` (``core.physics``) to render depth images for each
joint configuration and reprojects them into 3D point clouds.

Key differences from the yourdfpy path:
  - ``get_foreground_robot_points`` renders a depth map via PyBullet and
    reprojects; the yourdfpy path samples CAD meshes via forward kinematics.
  - ``get_foreground_gripper_points`` uses PyBullet segmentation masks to
    isolate gripper-only pixels for the wrist camera.
  - ``compute_robot_loss_batched`` does NOT use surface normals, front-face
    culling, or depth tolerance.
  - The Stage 2 loop uses 5 outer × 100 inner restart iterations per camera
    instead of a single 500-step sweep.

Both paths produce the same output format:
  ``{cam_id: {'base_extrinsic': np.array(4,4), 'extrinsics': np.array(N,4,4)}}``
so they can be swapped transparently in the main pipeline.
"""

import copy

import numpy as np
import pybullet as p
import torch
import torch.nn.functional as F
import torch.optim as optim

from core.geometry import make_T


# ===========================================================================
# Point cloud extraction from PyBullet renders
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
      renderer=p.ER_BULLET_HARDWARE_OPENGL,
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


# ===========================================================================
# Differentiable loss functions (no normals / no tolerance)
# ===========================================================================

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


# ===========================================================================
# Stage 2: Per-camera independent alignment (PyBullet)
# ===========================================================================

def run_stage2_alignment_pybullet(scene_constants, pb_renderer,
                                  init_scene_state, device):
  """Stage 2 per-camera independent optimization using PyBullet point clouds.

  Each camera is optimised independently with 5 outer restarts × 100 inner
  gradient descent steps.  External cameras use ``get_foreground_robot_points``
  (full body depth); the wrist camera uses ``get_foreground_gripper_points``
  (segmentation-masked gripper only).

  Args:
    scene_constants: Dict of metadata, camera data, and robot kinematics
        (same format as ``compute_extrinsics.py``).
    pb_renderer: ``PyBulletRenderer`` instance from ``core.physics``.
    init_scene_state: Dict ``{cam_id: {'base_extrinsic': ..., 'extrinsics': ...}}``
        produced by Stage 1 (VGGT or dataset init).
    device: Torch device string or object.

  Returns:
    Updated ``scene_state`` dict with the same structure.
  """
  print("\nStage 2 (PyBullet): Per-camera independent alignment...")
  wrist_cam = scene_constants['meta']['wrist_serial']
  stage2_scene_state = copy.deepcopy(init_scene_state)
  T_ee_base_all = scene_constants['robot']['T_ee_base_all']
  n_frames = len(scene_constants['robot']['joint_positions'])

  for cam in scene_constants['camera'].keys():
    is_wrist = (cam == wrist_cam)
    mode = "wrist (gripper-only)" if is_wrist else "external (full body)"
    print(f"\n  Optimizing [{mode}] camera: [{cam}] ...")

    K_np = scene_constants['camera'][cam]['K_mat']
    K_t = torch.tensor(K_np, dtype=torch.float32, device=device)
    T_base_np = stage2_scene_state[cam]['base_extrinsic']
    if T_base_np is None:
      print(f"    ⚠️ No initial extrinsic available! Skipping.")
      continue
    T_init_t = torch.tensor(T_base_np, dtype=torch.float32, device=device)

    # -- Pre-extract point clouds and observed depth tensors ---
    cache_X, cache_obs = [], []
    for t in range(n_frames):
      joints = scene_constants['robot']['joint_positions'][t]
      gripper = scene_constants['robot']['gripper_positions'][t]
      pb_renderer.update_robot_pose(joints, gripper)

      d_obs = scene_constants['camera'][cam]['raw_depth'][t].astype(np.float32)
      T_cam_np = stage2_scene_state[cam]['extrinsics'][t]

      if is_wrist:
        pts = get_foreground_gripper_points(
            T_cam_np, K_np, d_obs, pb_renderer, device,
        )
        if pts is None:
          continue
        # Convert camera-frame homogeneous (4, N) → EE-frame (N, 3)
        T_world_to_ee = np.linalg.inv(T_ee_base_all[t])
        pts_world = (T_cam_np @ pts)[:3, :].T  # → (N, 3)
        pts_ee = (T_world_to_ee[:3, :3] @ pts_world.T + T_world_to_ee[:3, 3:4]).T
        cache_X.append(
            torch.tensor(pts_ee, dtype=torch.float32, device=device),
        )
      else:
        pts = get_foreground_robot_points(
            T_cam_np, K_np, d_obs, pb_renderer, device,
        )
        if pts is None:
          continue
        cache_X.append(pts)

      cache_obs.append(
          torch.tensor(d_obs, dtype=torch.float32, device=device)[None, ...],
      )

    if not cache_X:
      print(f"    ⚠️ No valid point clouds extracted! Skipping.")
      continue

    batch_X = torch.stack(cache_X)
    batch_obs = torch.stack(cache_obs)

    # -- 5 outer × 100 inner restart optimisation ---
    n_outer, n_inner = 5, 100
    best_loss = float('inf')
    best_delta = torch.zeros(6, device=device)

    print(f"      Launching optimisation ({n_outer} outer × {n_inner} inner)...")
    for outer in range(n_outer):
      d_ext = torch.zeros(6, requires_grad=True, device=device)
      optimizer = optim.Adam([d_ext], lr=0.001)

      for inner in range(n_inner):
        optimizer.zero_grad()
        T_opt = T_init_t @ make_T(d_ext, device)

        if is_wrist:
          loss = compute_wrist_loss_batched(batch_X, T_opt, K_t, batch_obs)
        else:
          loss = compute_robot_loss_batched(batch_X, T_opt, K_t, batch_obs)

        loss.backward()
        optimizer.step()

      with torch.no_grad():
        final_loss = loss.item()
        if final_loss < best_loss:
          best_loss = final_loss
          best_delta = d_ext.detach().clone()

      rot_deg = torch.norm(d_ext[:3]).item() * (180.0 / np.pi)
      shift_mm = torch.norm(d_ext[3:]).item() * 1000.0
      print(f"        Outer {outer} | Loss: {final_loss:.4f} | "
            f"Shift: {shift_mm:.2f}mm | Rot: {rot_deg:.2f}°")

    with torch.no_grad():
      T_final_np = (T_init_t @ make_T(best_delta, device)).cpu().numpy()
      shift_mm = torch.norm(best_delta[3:]).item() * 1000.0
      rot_deg = torch.norm(best_delta[:3]).item() * (180.0 / np.pi)
      print(f"  ✅ [{cam}] Best loss: {best_loss:.4f} "
            f"(shift: {shift_mm:.2f}mm, rot: {rot_deg:.2f}°)")

      stage2_scene_state[cam]['base_extrinsic'] = T_final_np
      stage2_scene_state[cam]['extrinsics'] = (
          T_ee_base_all @ T_final_np if is_wrist
          else np.tile(T_final_np, (n_frames, 1, 1))
      )

  return stage2_scene_state


# ===========================================================================
# Chamfer environment point cloud helpers
# ===========================================================================

def batched_chamfer_distance(p1, p2, device):
  """Truncated Chamfer distance with 5cm physical cutoff."""
  dist_matrix = torch.cdist(p1, p2)
  min_dist_12 = torch.min(dist_matrix, dim=2)[0]
  min_dist_21 = torch.min(dist_matrix, dim=1)[0]

  valid_12 = min_dist_12 < 0.05
  valid_21 = min_dist_21 < 0.05

  loss = torch.tensor(0.0, device=device)
  if valid_12.any():
    loss += min_dist_12[valid_12].mean()
  if valid_21.any():
    loss += min_dist_21[valid_21].mean()

  overlap_ratio = (
      (valid_12.sum() + valid_21.sum()) /
      (p1.shape[0] * (p1.shape[1] + p2.shape[1]) + 1e-6)
  )
  return loss, overlap_ratio.item()


def get_cam_points_local_t(t, cam_data, device):
  """Extract downsampled scene point cloud from a single depth frame."""
  depth = cam_data["raw_depth"][t].astype(np.float32)
  K_mat_np = cam_data["K_mat"]

  valid_mask = (depth > 0.) & (depth < 1.5)
  vs, us = np.where(valid_mask)
  if len(us) < 100:
    return None

  zs_obs = depth[vs, us]
  x_c = (us - K_mat_np[0, 2]) * zs_obs / K_mat_np[0, 0]
  y_c = (vs - K_mat_np[1, 2]) * zs_obs / K_mat_np[1, 1]

  P_cam = np.stack([x_c, y_c, zs_obs, np.ones_like(zs_obs)], axis=0)
  if P_cam.shape[1] < 100:
    return None

  idx = np.random.choice(
      P_cam.shape[1], 2000, replace=(P_cam.shape[1] <= 2000),
  )
  return torch.tensor(P_cam[:, idx], dtype=torch.float32, device=device)


# ===========================================================================
# Stage 3: Global joint alignment (PyBullet)
# ===========================================================================

def run_global_joint_alignment_pybullet(scene_constants, prev_scene_state,
                                        pb_renderer, device,
                                        lr=0.001, n_steps=500,
                                        robot_weight=1.0,
                                        stage_name='Stage 3'):
  """Global joint optimisation: Chamfer stitching + Robot depth (PyBullet).

  Jointly optimises all camera extrinsics (two external + one wrist) against
  Chamfer environment stitching losses and robot depth re-projection losses.
  Fully matches the structure of ``run_global_joint_alignment`` in
  ``compute_extrinsics.py`` but uses PyBullet-rendered point clouds instead
  of yourdfpy meshes for the robot depth component.

  Total loss = Chamfer(c1↔c2 + c1↔w + c2↔w) + robot_weight × (Rob1 + Rob2 + Wrist)

  Args:
    scene_constants: Scene metadata / camera / robot dict.
    prev_scene_state: Output of Stage 2 (per-camera optimised extrinsics).
    pb_renderer: ``PyBulletRenderer`` instance.
    device: Torch device string or object.
    lr: Adam learning rate.
    n_steps: Total gradient descent steps.
    robot_weight: Weight multiplier for robot depth losses.
    stage_name: Label printed in logs.

  Returns:
    Final ``scene_state`` dict with the same structure as input.
  """
  print(f"\n{stage_name} (PyBullet): Global joint optimization "
        f"(Chamfer + Robot + Wrist, lr={lr})...")
  wrist_cam = scene_constants['meta']['wrist_serial']
  ext_cams = [c for c in scene_constants['camera'].keys() if c != wrist_cam]
  cam1, cam2 = ext_cams[0], ext_cams[1]
  n_frames = len(scene_constants['robot']['joint_positions'])
  T_ee_all = scene_constants['robot']['T_ee_base_all']

  # -- Pre-extract robot point clouds per camera per frame ---
  print(f"  Extracting PyBullet robot point clouds...")
  caches = {}  # cam_id → (batch_X, batch_obs)

  for cam in [cam1, cam2, wrist_cam]:
    is_wrist = (cam == wrist_cam)
    K_np = scene_constants['camera'][cam]['K_mat']
    cache_X, cache_obs = [], []

    for t in range(n_frames):
      joints = scene_constants['robot']['joint_positions'][t]
      gripper = scene_constants['robot']['gripper_positions'][t]
      pb_renderer.update_robot_pose(joints, gripper)

      d_obs = scene_constants['camera'][cam]['raw_depth'][t].astype(np.float32)
      T_cam_np = prev_scene_state[cam]['extrinsics'][t]

      if is_wrist:
        pts = get_foreground_gripper_points(
            T_cam_np, K_np, d_obs, pb_renderer, device,
        )
        if pts is None:
          continue
        T_world_to_ee = np.linalg.inv(T_ee_all[t])
        pts_world = (T_cam_np @ pts)[:3, :].T
        pts_ee = (T_world_to_ee[:3, :3] @ pts_world.T + T_world_to_ee[:3, 3:4]).T
        cache_X.append(
            torch.tensor(pts_ee, dtype=torch.float32, device=device),
        )
      else:
        pts = get_foreground_robot_points(
            T_cam_np, K_np, d_obs, pb_renderer, device,
        )
        if pts is None:
          continue
        cache_X.append(pts)

      cache_obs.append(
          torch.tensor(d_obs, dtype=torch.float32, device=device)[None, ...],
      )

    if cache_X:
      caches[cam] = (torch.stack(cache_X), torch.stack(cache_obs))
    else:
      caches[cam] = (None, None)

  batch_X1, batch_obs1 = caches[cam1]
  batch_X2, batch_obs2 = caches[cam2]
  batch_P_ee, batch_obs_w = caches[wrist_cam]

  # -- Extract Chamfer environment point clouds ---
  print(f"  Extracting Chamfer environment point clouds...")
  cache_Pc1, cache_Pc2, cache_Pcw, cache_Tee = [], [], [], []

  for t in range(n_frames):
    pc1 = get_cam_points_local_t(t, scene_constants['camera'][cam1], device)
    pc2 = get_cam_points_local_t(t, scene_constants['camera'][cam2], device)
    pcw = get_cam_points_local_t(t, scene_constants['camera'][wrist_cam], device)

    if pc1 is not None and pc2 is not None and pcw is not None:
      cache_Pc1.append(pc1)
      cache_Pc2.append(pc2)
      cache_Pcw.append(pcw)
      cache_Tee.append(
          torch.tensor(T_ee_all[t], dtype=torch.float32, device=device),
      )

  batch_Pc1 = torch.stack(cache_Pc1)
  batch_Pc2 = torch.stack(cache_Pc2)
  batch_Pcw = torch.stack(cache_Pcw)
  batch_Tee = torch.stack(cache_Tee)

  K_t1 = torch.tensor(scene_constants['camera'][cam1]['K_mat'], dtype=torch.float32, device=device)
  K_t2 = torch.tensor(scene_constants['camera'][cam2]['K_mat'], dtype=torch.float32, device=device)
  K_t_w = torch.tensor(scene_constants['camera'][wrist_cam]['K_mat'], dtype=torch.float32, device=device)

  d1 = torch.zeros(6, requires_grad=True, device=device)
  d2 = torch.zeros(6, requires_grad=True, device=device)
  dhe = torch.zeros(6, requires_grad=True, device=device)
  optimizer = optim.Adam([d1, d2, dhe], lr=lr)

  T1_init_t = torch.tensor(prev_scene_state[cam1]['base_extrinsic'], dtype=torch.float32, device=device)
  T2_init_t = torch.tensor(prev_scene_state[cam2]['base_extrinsic'], dtype=torch.float32, device=device)
  Tee_init_t = torch.tensor(prev_scene_state[wrist_cam]['base_extrinsic'], dtype=torch.float32, device=device)

  print(f"  ✅ Data ready! Launching joint optimisation ({n_steps} steps)...")
  for step in range(n_steps):
    optimizer.zero_grad()

    T1_opt = T1_init_t @ make_T(d1, device)
    T2_opt = T2_init_t @ make_T(d2, device)
    Tee_opt = Tee_init_t @ make_T(dhe, device)

    # Chamfer: project environment points to world frame
    bc1 = (T1_opt @ batch_Pc1)[:, :3, :].transpose(1, 2)
    bc2 = (T2_opt @ batch_Pc2)[:, :3, :].transpose(1, 2)
    T_wrist_c2w = batch_Tee @ Tee_opt
    bcw = torch.bmm(T_wrist_c2w, batch_Pcw)[:, :3, :].transpose(1, 2)

    l12, o12 = batched_chamfer_distance(bc1, bc2, device)
    l1w, o1w = batched_chamfer_distance(bc1, bcw, device)
    l2w, o2w = batched_chamfer_distance(bc2, bcw, device)
    loss_chamfer = l12 + l1w + l2w

    # Robot depth losses
    l_rob1 = torch.tensor(0.0, device=device)
    if batch_X1 is not None:
      l_rob1 = compute_robot_loss_batched(batch_X1, T1_opt, K_t1, batch_obs1)

    l_rob2 = torch.tensor(0.0, device=device)
    if batch_X2 is not None:
      l_rob2 = compute_robot_loss_batched(batch_X2, T2_opt, K_t2, batch_obs2)

    l_wrist = torch.tensor(0.0, device=device)
    if batch_P_ee is not None:
      l_wrist = compute_wrist_loss_batched(batch_P_ee, Tee_opt, K_t_w, batch_obs_w)

    loss_total = loss_chamfer + robot_weight * (l_rob1 + l_rob2 + l_wrist)
    loss_total.backward()
    optimizer.step()

    if step % 50 == 0 or step == n_steps - 1:
      bg_overlap = (o12 + o1w + o2w) / 3.0 * 100
      shift_c1 = torch.norm(d1[3:]).item() * 1000
      shift_c2 = torch.norm(d2[3:]).item() * 1000
      shift_w = torch.norm(dhe[3:]).item() * 1000
      print(f"    Step {step:03d} | "
            f"Chmf: {loss_chamfer.item():.4f} | "
            f"Rob1: {l_rob1.item():.4f} | Rob2: {l_rob2.item():.4f} | "
            f"Wrst: {l_wrist.item():.4f} | "
            f"BG Overlap: {bg_overlap:.1f}% | "
            f"Shift → C1: {shift_c1:.2f}mm, C2: {shift_c2:.2f}mm, W: {shift_w:.2f}mm")

  with torch.no_grad():
    final_p1 = (T1_init_t @ make_T(d1, device)).cpu().numpy()
    final_p2 = (T2_init_t @ make_T(d2, device)).cpu().numpy()
    final_cam_ee = (Tee_init_t @ make_T(dhe, device)).cpu().numpy()

  print(f"\n✅ {stage_name} (PyBullet, Chamfer + Robot + Wrist) complete!")

  ultimate_scene_state = {c: {} for c in scene_constants['camera'].keys()}
  ultimate_scene_state[cam1].update({
      "base_extrinsic": final_p1,
      "extrinsics": np.tile(final_p1, (n_frames, 1, 1)),
  })
  ultimate_scene_state[cam2].update({
      "base_extrinsic": final_p2,
      "extrinsics": np.tile(final_p2, (n_frames, 1, 1)),
  })
  ultimate_scene_state[wrist_cam].update({
      "base_extrinsic": final_cam_ee,
      "extrinsics": T_ee_all @ final_cam_ee,
  })

  return ultimate_scene_state
