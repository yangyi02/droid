"""DROID Stage 2: Camera Extrinsics Calibration Pipeline.

Multi-stage camera extrinsics calibration using differentiable rendering
and point cloud alignment. Reads Stage 1 outputs (depth, calibration,
robot kinematics, video) and produces optimized 4x4 extrinsic matrices
for all cameras.

Pipeline:
  Stage 0: Read dataset extrinsics (required, no fallback)
  Stage 1: Unified camera-robot alignment (external + wrist in one loop)
  Stage 2: Global joint optimization (Chamfer + Robot + Wrist)
"""

import argparse
import copy
import fcntl
import gc
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from core.geometry import make_4x4, make_T
from core.io import get_accelerator, load_depth_data, load_metadata
from core.physics import TensorRobotRenderer


# ---------------------------------------------------------------------------
# Metadata & Data Loading
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Stage 0: Read Dataset Extrinsics
# ---------------------------------------------------------------------------
def init_camera_states(scene_constants, extrinsics_db):
  """Assemble initial 3D camera states from dataset metadata."""
  print("  Initializing camera 3D states...")
  wrist_serial = scene_constants["meta"]["wrist_serial"]
  robot_data = scene_constants["robot"]
  n_frames = len(robot_data["T_ee_base_all"])

  episode_id = scene_constants["meta"]["episode_id"]
  episode_extrinsics = extrinsics_db.get(episode_id, {})

  scene_state = {}

  for cam_id in scene_constants["camera"].keys():
    if cam_id == wrist_serial:
      base_ext = robot_data["T_cam_ee_init"]
      cam_trajectory = robot_data["T_ee_base_all"] @ base_ext
    elif cam_id in episode_extrinsics:
      ext_data = episode_extrinsics[cam_id]
      ext_vec = ext_data.get("extrinsics", ext_data) if isinstance(ext_data, dict) else ext_data
      base_ext = make_4x4(ext_vec)
      cam_trajectory = np.tile(base_ext, (n_frames, 1, 1))
      print(f"    ✅ Loaded pre-calibrated extrinsics for camera [{cam_id}] from metadata.")
    else:
      print(f"    ⚠️ No pre-calibrated extrinsics for external camera [{cam_id}], setting to None.")
      base_ext = None
      cam_trajectory = None

    scene_state[cam_id] = {
        "base_extrinsic": base_ext,
        "extrinsics": cam_trajectory,
    }

  return scene_state


def init_extrinsics(scene_constants, extrinsics_db):
  """Initialize camera extrinsics from dataset metadata.

  Requires pre-calibrated extrinsics for all external cameras.
  """
  scene_state = init_camera_states(scene_constants, extrinsics_db)
  missing = [cam for cam, state in scene_state.items()
             if state['extrinsics'] is None]
  if missing:
    raise ValueError(
        f"Missing pre-calibrated extrinsics for cameras: {missing}. "
        f"Only episodes with dataset extrinsics are supported.")
  print("  ✅ All cameras have pre-calibrated extrinsics.")
  return scene_state


# ---------------------------------------------------------------------------
# 8. Shared Loss & Data Factory
# ---------------------------------------------------------------------------
def compute_robot_loss(batch_X_base, T_cam_to_base, K, batch_obs, depth_tolerance):
  """Unified depth re-projection loss with normals, front-face culling, and tolerance."""
  B, _, h_img, w_img = batch_obs.shape

  T_base_to_cam = torch.linalg.inv(T_cam_to_base)
  rot, t = T_base_to_cam[:3, :3], T_base_to_cam[:3, 3]

  pts_base, normals_base = batch_X_base[..., :3], batch_X_base[..., 3:]

  P_c = pts_base @ rot.T + t
  Z_pred = P_c[..., 2].clamp(min=1e-4)

  # Front-face culling via normal dot product
  normals_c = normals_base @ rot.T
  front_facing = (normals_c * P_c).sum(dim=-1) < 0

  u = K[0, 0] * P_c[..., 0] / Z_pred + K[0, 2]
  v = K[1, 1] * P_c[..., 1] / Z_pred + K[1, 2]

  grid = torch.stack([(u / (w_img - 1)) * 2 - 1, (v / (h_img - 1)) * 2 - 1], dim=-1).unsqueeze(1)
  Z_obs = F.grid_sample(batch_obs, grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(1).squeeze(1)

  diff = torch.abs(Z_obs - Z_pred)
  valid = (Z_pred > 0.) & (Z_obs > 0.) & \
          (u >= 0) & (u < w_img - 1) & (v >= 0) & (v < h_img - 1) & \
          front_facing & (diff < depth_tolerance)

  return torch.nan_to_num(diff[valid].mean(), nan=0.0)


def extract_robot_physical_tensors(cam_id, scene_constants, tensor_renderer):
  """One-shot physical tensor extraction factory for any camera.

  For external cameras: returns CAD points in world frame with normals.
  For wrist camera: transforms CAD gripper points into EE frame.
  """
  device = tensor_renderer.device
  is_wrist = (cam_id == scene_constants['meta']['wrist_serial'])
  n_frames = len(scene_constants['camera'][cam_id]['raw_depth'])
  T_ee_base_all = scene_constants['robot']['T_ee_base_all']

  cache_X, cache_obs = [], []
  for t in range(n_frames):
    joints = scene_constants['robot']['joint_positions'][t]
    gripper = scene_constants['robot']['gripper_positions'][t]
    d_obs = scene_constants['camera'][cam_id]['raw_depth'][t].astype(np.float32)

    cad_pts_world = tensor_renderer.get_world_points(joints, gripper, only_gripper=is_wrist)
    if cad_pts_world is None:
      continue

    if is_wrist:
      # Transform world points into EE frame for hand-eye optimization
      T_world_to_ee = torch.linalg.inv(torch.tensor(T_ee_base_all[t], dtype=torch.float32, device=device))
      pts_w_h = torch.cat([cad_pts_world[:, :3], torch.ones((cad_pts_world.shape[0], 1), device=device)], dim=1)
      pts_base = (T_world_to_ee @ pts_w_h.T).T[:, :3]
      normals_base = (T_world_to_ee[:3, :3] @ cad_pts_world[:, 3:].T).T
      cache_X.append(torch.cat([pts_base, normals_base], dim=-1))
    else:
      cache_X.append(cad_pts_world)

    cache_obs.append(torch.tensor(d_obs, dtype=torch.float32, device=device)[None, ...])

  # Free cached GPU tensors from get_world_points (already copied into cache_X)
  tensor_renderer.world_points_cache.clear()

  if not cache_X:
    return None, None
  return torch.stack(cache_X), torch.stack(cache_obs)


# ---------------------------------------------------------------------------
# 9. Stage 2: Unified Camera-Robot Alignment (external + wrist)
# ---------------------------------------------------------------------------
def run_stage2_alignment(scene_constants, tensor_renderer, stage1_scene_state):
  """Unified Stage 2: optimize all cameras against robot body/gripper depth."""
  print("\nStage 2: Unified camera-robot alignment (external + wrist)...")
  device = tensor_renderer.device
  wrist_cam = scene_constants['meta']['wrist_serial']
  stage2_scene_state = copy.deepcopy(stage1_scene_state)
  T_ee_base_all = scene_constants['robot']['T_ee_base_all']

  for cam in scene_constants['camera'].keys():
    is_wrist = (cam == wrist_cam)
    mode = "wrist (gripper-only)" if is_wrist else "external (full body)"
    print(f"\n  Optimizing [{mode}] camera: [{cam}] ...")

    # Extract physical tensors from shared factory
    batch_X_base, batch_obs = extract_robot_physical_tensors(cam, scene_constants, tensor_renderer)
    if batch_X_base is None:
      print(f"    ⚠️ No valid physical point cloud extracted! Skipping.")
      continue

    n_frames_total = len(batch_X_base)
    K_t = torch.tensor(scene_constants['camera'][cam]['K_mat'], dtype=torch.float32, device=device)
    T_init_t = torch.tensor(stage1_scene_state[cam]['base_extrinsic'], dtype=torch.float32, device=device)

    d_ext = torch.zeros(6, requires_grad=True, device=device)
    total_steps = 500
    optimizer = optim.Adam([d_ext], lr=0.001)

    print(f"      Launching GPU tensor gradient descent ({total_steps} steps)...")
    for step in range(total_steps):
      optimizer.zero_grad()
      T_cam_to_base = T_init_t @ make_T(d_ext, device)
      loss_rob = compute_robot_loss(
          batch_X_base, T_cam_to_base, K_t, batch_obs,
          depth_tolerance=float('inf') if is_wrist else 0.15,
      )
      loss_rob.backward()
      optimizer.step()

      if step % 50 == 0 or step == total_steps - 1:
        with torch.no_grad():
          rot_deg = torch.norm(d_ext[:3]).item() * (180.0 / np.pi)
          shift_mm = torch.norm(d_ext[3:]).item() * 1000.0
        print(f"        Step {step:03d} | Loss: {loss_rob.item():.4f} | Shift: {shift_mm:.2f}mm | Rot: {rot_deg:.2f}°")

    with torch.no_grad():
      T_final_np = (T_init_t @ make_T(d_ext, device)).cpu().numpy()
      shift_mm = torch.norm(d_ext[3:]).item() * 1000.0
      rot_deg = torch.norm(d_ext[:3]).item() * (180.0 / np.pi)
      print(f"  ✅ [{cam}] Alignment done! Loss: {loss_rob.item():.4f} (shift: {shift_mm:.2f}mm, rot: {rot_deg:.2f}°)")

      stage2_scene_state[cam]['base_extrinsic'] = T_final_np
      stage2_scene_state[cam]['extrinsics'] = (
          T_ee_base_all @ T_final_np if is_wrist
          else np.tile(T_final_np, (n_frames_total, 1, 1))
      )

  return stage2_scene_state


# ---------------------------------------------------------------------------
# 10. Stage 3: Global Joint Optimization (Chamfer + Robot + Wrist)
# ---------------------------------------------------------------------------
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

  overlap_ratio = (valid_12.sum() + valid_21.sum()) / (p1.shape[0] * (p1.shape[1] + p2.shape[1]) + 1e-6)
  return loss, overlap_ratio.item()




def get_cam_points_local_t(t, cam_data, device, n_points=2000):
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

  idx = np.random.choice(P_cam.shape[1], n_points, replace=(P_cam.shape[1] <= n_points))
  return torch.tensor(P_cam[:, idx], dtype=torch.float32, device=device)


def run_global_joint_alignment(scene_constants, prev_scene_state, tensor_renderer,
                                lr=0.001, n_steps=500,
                                chamfer_weight=1.0, robot_weight=1.0,
                                chamfer_n_points=2000,
                                stage_name="Stage 2"):
  """Global joint optimization: Chamfer + Robot depth + Wrist depth.

  Args:
    scene_constants: Scene data dict.
    prev_scene_state: Previous extrinsics to refine.
    tensor_renderer: TensorRobotRenderer for robot point clouds.
    lr: Learning rate.
    n_steps: Number of optimization steps.
    robot_weight: Weight for robot depth losses.
    stage_name: Display name for logging.
  """
  print(f"\n{stage_name}: Global joint optimization "
        f"(Chamfer + Robot + Wrist, lr={lr})...")
  device = tensor_renderer.device
  wrist_cam = scene_constants['meta']['wrist_serial']
  ext_cams = [c for c in scene_constants['camera'].keys() if c != wrist_cam]
  cam1, cam2 = ext_cams[0], ext_cams[1]
  n_frames = len(scene_constants['robot']['joint_positions'])
  T_ee_all = scene_constants['robot']['T_ee_base_all']

  # Extract robot physical tensors from shared factory
  print(f"  Extracting robot physical tensor caches...")
  batch_X1, batch_obs1 = extract_robot_physical_tensors(cam1, scene_constants, tensor_renderer)
  batch_X2, batch_obs2 = extract_robot_physical_tensors(cam2, scene_constants, tensor_renderer)
  batch_P_ee, batch_obs_w = extract_robot_physical_tensors(wrist_cam, scene_constants, tensor_renderer)

  # Extract Chamfer environment point clouds
  print(f"  Extracting Chamfer environment point clouds...")
  cache_Pc1, cache_Pc2, cache_Pcw, cache_Tee = [], [], [], []

  for t in range(n_frames):
    pc1 = get_cam_points_local_t(t, scene_constants['camera'][cam1], device, n_points=chamfer_n_points)
    pc2 = get_cam_points_local_t(t, scene_constants['camera'][cam2], device, n_points=chamfer_n_points)
    pcw = get_cam_points_local_t(t, scene_constants['camera'][wrist_cam], device, n_points=chamfer_n_points)

    if pc1 is not None and pc2 is not None and pcw is not None:
      cache_Pc1.append(pc1)
      cache_Pc2.append(pc2)
      cache_Pcw.append(pcw)
      cache_Tee.append(torch.tensor(T_ee_all[t], dtype=torch.float32, device=device))

  batch_Pc1, batch_Pc2, batch_Pcw = torch.stack(cache_Pc1), torch.stack(cache_Pc2), torch.stack(cache_Pcw)
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

  # Map camera IDs to their delta/init/K for track loss lookup
  cam_opt_map = {
      cam1: (d1, T1_init_t, K_t1),
      cam2: (d2, T2_init_t, K_t2),
      wrist_cam: (dhe, Tee_init_t, K_t_w),
  }

  print(f"  ✅ Data ready! Launching GPU joint optimization engine ({n_steps} steps)...")
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
    l_rob1 = compute_robot_loss(batch_X1, T1_opt, K_t1, batch_obs1, depth_tolerance=0.15)
    l_rob2 = compute_robot_loss(batch_X2, T2_opt, K_t2, batch_obs2, depth_tolerance=0.15)
    l_wrist = compute_robot_loss(batch_P_ee, Tee_opt, K_t_w, batch_obs_w, depth_tolerance=float('inf'))

    loss_total = chamfer_weight * loss_chamfer + robot_weight * (l_rob1 + l_rob2 + l_wrist)

    loss_total.backward()
    optimizer.step()

    if step % 50 == 0 or step == n_steps - 1:
      bg_overlap = (o12 + o1w + o2w) / 3.0 * 100
      shift_c1 = torch.norm(d1[3:]).item() * 1000
      shift_c2 = torch.norm(d2[3:]).item() * 1000
      shift_w = torch.norm(dhe[3:]).item() * 1000
      log_parts = [
          f"Step {step:03d}",
          f"Chmf: {loss_chamfer.item():.4f}",
          f"Rob1: {l_rob1.item():.4f}", f"Rob2: {l_rob2.item():.4f}",
          f"Wrst: {l_wrist.item():.4f}",
      ]
      log_parts.extend([
          f"BG Overlap: {bg_overlap:.1f}%",
          f"Shift → C1: {shift_c1:.2f}mm, C2: {shift_c2:.2f}mm, W: {shift_w:.2f}mm",
      ])
      print(f"    {' | '.join(log_parts)}")

  with torch.no_grad():
    final_p1 = (T1_init_t @ make_T(d1, device)).cpu().numpy()
    final_p2 = (T2_init_t @ make_T(d2, device)).cpu().numpy()
    final_cam_ee = (Tee_init_t @ make_T(dhe, device)).cpu().numpy()

  print(f"\n✅ {stage_name} complete!")

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


# ---------------------------------------------------------------------------
# Evaluate Extrinsics Quality
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_extrinsics(scene_constants, scene_state, device,
                        pb_renderer=None, tensor_renderer=None):
  """Compute extrinsics quality metrics without re-running optimization.

  Can be called after any stage to monitor calibration quality.

  Args:
    scene_constants: Scene data dict.
    scene_state: Current extrinsics state.
    device: Torch device.
    pb_renderer: Optional PyBulletRenderer to reuse. If None, one
        is created temporarily.

  Returns:
    Dict with metrics: chamfer_total, robot_loss_*, bg_overlap_pct,
    track_reproj_mean_px, shift_mm_*.
  """


  wrist_cam = scene_constants["meta"]["wrist_serial"]
  ext_cams = [c for c in scene_constants["camera"].keys() if c != wrist_cam]
  cam1, cam2 = ext_cams[0], ext_cams[1]
  n_frames = len(scene_constants["robot"]["joint_positions"])
  T_ee_all = scene_constants["robot"]["T_ee_base_all"]

  metrics = {}

  # --- Robot depth losses ---
  if tensor_renderer is not None:
    # Fast path: yourdfpy FK -> direct 3D point clouds (no rendering)
    for cam_id, key_prefix in [(cam1, "cam1"), (cam2, "cam2"),
                                (wrist_cam, "wrist")]:
      try:
        is_wrist = (cam_id == wrist_cam)
        K_t = torch.tensor(
            scene_constants["camera"][cam_id]["K_mat"],
            dtype=torch.float32, device=device)
        batch_X, batch_obs = extract_robot_physical_tensors(
            cam_id, scene_constants, tensor_renderer)
        if batch_X is None:
          metrics[f"robot_loss_{key_prefix}"] = float("nan")
          continue
        T_opt = torch.tensor(
            scene_state[cam_id]["base_extrinsic"],
            dtype=torch.float32, device=device)
        loss = compute_robot_loss(
            batch_X, T_opt, K_t, batch_obs,
            depth_tolerance=float('inf') if is_wrist else 0.15)
        metrics[f"robot_loss_{key_prefix}"] = loss.item()
      except Exception as e:
        metrics[f"robot_loss_{key_prefix}"] = float("nan")

  # --- Chamfer losses ---
  try:
    cache_Pc1, cache_Pc2, cache_Pcw, cache_Tee = [], [], [], []
    for t in range(n_frames):
      pc1 = get_cam_points_local_t(t, scene_constants["camera"][cam1], device, n_points=5000)
      pc2 = get_cam_points_local_t(t, scene_constants["camera"][cam2], device, n_points=5000)
      pcw = get_cam_points_local_t(
          t, scene_constants["camera"][wrist_cam], device, n_points=5000)
      if pc1 is not None and pc2 is not None and pcw is not None:
        cache_Pc1.append(pc1)
        cache_Pc2.append(pc2)
        cache_Pcw.append(pcw)
        cache_Tee.append(
            torch.tensor(T_ee_all[t], dtype=torch.float32, device=device))

    batch_Pc1 = torch.stack(cache_Pc1)
    batch_Pc2 = torch.stack(cache_Pc2)
    batch_Pcw = torch.stack(cache_Pcw)
    batch_Tee = torch.stack(cache_Tee)

    T1 = torch.tensor(
        scene_state[cam1]["base_extrinsic"],
        dtype=torch.float32, device=device)
    T2 = torch.tensor(
        scene_state[cam2]["base_extrinsic"],
        dtype=torch.float32, device=device)
    Tw = torch.tensor(
        scene_state[wrist_cam]["base_extrinsic"],
        dtype=torch.float32, device=device)

    bc1 = (T1 @ batch_Pc1)[:, :3, :].transpose(1, 2)
    bc2 = (T2 @ batch_Pc2)[:, :3, :].transpose(1, 2)
    bcw = torch.bmm(batch_Tee @ Tw, batch_Pcw)[:, :3, :].transpose(1, 2)

    l12, o12 = batched_chamfer_distance(bc1, bc2, device)
    l1w, o1w = batched_chamfer_distance(bc1, bcw, device)
    l2w, o2w = batched_chamfer_distance(bc2, bcw, device)

    metrics["chamfer_12"] = l12.item()
    metrics["chamfer_1w"] = l1w.item()
    metrics["chamfer_2w"] = l2w.item()
    metrics["chamfer_total"] = (l12 + l1w + l2w).item()
    metrics["bg_overlap_pct"] = (o12 + o1w + o2w) / 3.0 * 100
  except Exception as e:
    metrics["chamfer_total"] = float("nan")
    metrics["bg_overlap_pct"] = float("nan")

  return metrics


def print_metrics(metrics, stage_name=""):
  """Pretty-print key metrics in a compact one-block summary."""
  chamfer = metrics.get("chamfer_total", float("nan"))
  rob1 = metrics.get("robot_loss_cam1", float("nan"))
  rob2 = metrics.get("robot_loss_cam2", float("nan"))
  robw = metrics.get("robot_loss_wrist", float("nan"))
  overlap = metrics.get("bg_overlap_pct", float("nan"))
  track = metrics.get("track_reproj_mean_px", float("nan"))
  track_med = metrics.get("track_reproj_median_px", float("nan"))
  wrist_bg = metrics.get("track_reproj_wrist_bg_mean_px", float("nan"))
  wrist_bg_med = metrics.get("track_reproj_wrist_bg_median_px", float("nan"))
  static_robot = metrics.get("track_reproj_static_robot_mean_px", float("nan"))
  static_robot_med = metrics.get("track_reproj_static_robot_median_px", float("nan"))

  header = f"Metrics after {stage_name}" if stage_name else "Metrics"
  print(f"\n{header}")
  print(f"  Chamfer total: {chamfer:.4f}")
  print(f"  Robot depth:   cam1={rob1:.4f}  cam2={rob2:.4f}  wrist={robw:.4f}")
  print(f"  BG overlap:    {overlap:.1f}%")
  if not np.isnan(wrist_bg):
    med_s = f"  median={wrist_bg_med:.2f}" if not np.isnan(wrist_bg_med) else ""
    print(f"  Track wristBG: mean={wrist_bg:.2f} px{med_s}  ★ primary")
  if not np.isnan(static_robot):
    med_s2 = f"  median={static_robot_med:.2f}" if not np.isnan(static_robot_med) else ""
    print(f"  Track robot:   mean={static_robot:.2f} px{med_s2}")
  if not np.isnan(track) and np.isnan(wrist_bg) and np.isnan(static_robot):
    print(f"  Track reproj:  mean={track:.2f} px")

  shift_keys = [k for k in sorted(metrics.keys()) if k.startswith("shift_mm_")]
  if shift_keys:
    shifts = [f"{k.replace('shift_mm_', '')}={metrics[k]:.1f}mm"
              for k in shift_keys]
    print(f"  Shift from 0:  {', '.join(shifts)}")
  print()



def export_extrinsics(scene_constants, scene_state,
                      export_root="~/droid_data/output/mv-tap/droid/extrinsics",
                      stage_suffix=None):
  """Save calibrated extrinsics as per-camera JSON files.

  Output layout:
    <episode_id>/<cam_id>/extrinsics.json            (final)
    <episode_id>/<cam_id>/extrinsics_stage1.json     (after dataset init)
    <episode_id>/<cam_id>/extrinsics_stage2.json     (after robot alignment)

  Each JSON contains:
    base_extrinsic  → (4x4) list-of-lists, static extrinsic matrix
    extrinsics      → (Nx4x4) list-of-lists, per-frame trajectory
    is_wrist        → bool
  """
  ep_str = scene_constants["meta"]["episode_id"]
  wrist_serial = scene_constants["meta"]["wrist_serial"]
  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(export_root, ep_str)))
  fname = f"extrinsics_{stage_suffix}.json" if stage_suffix else "extrinsics.json"

  for cam_id, state in scene_state.items():
    if state.get("base_extrinsic") is None or state.get("extrinsics") is None:
      continue

    cam_dir = os.path.join(ep_dir, cam_id)
    os.makedirs(cam_dir, exist_ok=True)

    payload = {
        "base_extrinsic": state["base_extrinsic"].astype(np.float64).tolist(),
        "extrinsics": state["extrinsics"].astype(np.float64).tolist(),
        "is_wrist": (cam_id == wrist_serial),
    }

    out_path = os.path.join(cam_dir, fname)
    with open(out_path, "w") as f:
      json.dump(payload, f, indent=2)

  print(f"  Extrinsics saved to {ep_dir}/*/{fname}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="DROID Stage 2: Camera Extrinsics Calibration")
  parser.add_argument("--rank", type=int, default=0, help="Rank of the process")
  parser.add_argument("--world_size", type=int, default=1, help="Total number of processes")
  parser.add_argument("--limit", type=int, default=-1, help="Limit total number of episodes to process")
  parser.add_argument("--depth_root", type=str, default="~/droid_data/output/mv-tap/droid/depth",
                       help="Root directory of depth outputs")
  parser.add_argument("--method", type=str, choices=["yourdfpy", "pybullet"], default="yourdfpy",
                       help="Rendering engine for robot point cloud extraction")
  parser.add_argument("--skip_stage3", action="store_true",
                       help="Skip Stage 3 global joint optimization")
  parser.add_argument("--export_root", type=str,
                       default="~/droid_data/output/mv-tap/droid/extrinsics",
                       help="Root directory for extrinsics output")
  args = parser.parse_args()

  print(f"DROID Stage 2: Camera Extrinsics Calibration Pipeline")
  device = get_accelerator()
  serials_db, _, _, extrinsics_db, _ = load_metadata()



  # Discover available episodes from depth output
  depth_abs = os.path.abspath(os.path.expanduser(args.depth_root))
  available_eps = sorted([
      d for d in os.listdir(depth_abs)
      if os.path.isdir(os.path.join(depth_abs, d))
  ])
  import random
  random.seed(42)
  random.shuffle(available_eps)
  if args.limit > 0:
    available_eps = available_eps[:args.limit]
  target_eps = available_eps[args.rank::args.world_size]
  print(f"Selected via distributed rank {args.rank}/{args.world_size} targeting: {len(target_eps)} episodes")

  tensor_renderer = TensorRobotRenderer(device=device)
  print("  Using TensorRobotRenderer (yourdfpy)")

  succeeded_eps = []

  for idx, ep_id in enumerate(target_eps):
    print(f"\n[{idx + 1}/{len(target_eps)}] Processing Episode: {ep_id}")

    # Pre-initialize for safe cleanup in finally block
    scene_constants = None
    stage1_scene_state = None
    stage2_state = None
    stage3_state = None
    final_state = None

    try:
      # Load Stage 1 outputs
      scene_constants = load_depth_data(ep_id, args.depth_root)

      # Stage 0: Load dataset extrinsics
      stage1_scene_state = init_extrinsics(scene_constants, extrinsics_db)
      print_metrics(
          evaluate_extrinsics(scene_constants, stage1_scene_state, device,
                              tensor_renderer=tensor_renderer),
          stage_name="Stage 0+1 (Init)")

      # Save Stage 1 extrinsics
      export_extrinsics(scene_constants, stage1_scene_state,
                        export_root=args.export_root, stage_suffix="stage1")

      # Stage 2: Per-camera independent alignment (external + wrist)
      stage2_state = run_stage2_alignment(
          scene_constants, tensor_renderer, stage1_scene_state,
      )
      print_metrics(
          evaluate_extrinsics(scene_constants, stage2_state, device,
                              tensor_renderer=tensor_renderer),
          stage_name="Stage 2 (Per-Camera Alignment)")
      export_extrinsics(scene_constants, stage2_state,
                        export_root=args.export_root, stage_suffix="stage2")

      if args.skip_stage3:
        # Export Stage 2 result as final
        export_extrinsics(scene_constants, stage2_state,
                          export_root=args.export_root)
        succeeded_eps.append(ep_id)
      else:
        # Stage 3: Global joint optimization (Chamfer + Robot + Wrist)
        stage3_state = run_global_joint_alignment(
            scene_constants, stage2_state, tensor_renderer,
            lr=0.001, n_steps=500, robot_weight=1.0, stage_name="Stage 3",
        )
        print_metrics(
            evaluate_extrinsics(scene_constants, stage3_state, device,
                                tensor_renderer=tensor_renderer),
            stage_name="Stage 3 (Global Joint)")
        export_extrinsics(scene_constants, stage3_state,
                          export_root=args.export_root, stage_suffix="stage3")
        # Final export (canonical name)
        export_extrinsics(scene_constants, stage3_state,
                          export_root=args.export_root)
        succeeded_eps.append(ep_id)
      print(f"  ✅ Episode {ep_id} completed successfully.")

    except Exception as e:
      print(f"  ❌ Episode {ep_id} failed: {e}")
      import traceback
      traceback.print_exc()

    finally:
      # Free GPU memory between episodes to prevent OOM from fragmentation
      scene_constants = None
      stage1_scene_state = None
      stage2_state = None
      stage3_state = None
      final_state = None
      # Clear the per-joint-config GPU tensor cache (main source of OOM)
      if tensor_renderer is not None:
        tensor_renderer.world_points_cache.clear()
      gc.collect()
      torch.cuda.empty_cache()

  # Multi-process safe append
  extrinsics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "episodes_extrinsics.txt")
  if succeeded_eps:
    batch = "".join(ep_id + "\n" for ep_id in succeeded_eps)
    with open(extrinsics_path, "a") as f:
      fcntl.flock(f, fcntl.LOCK_EX)
      f.write(batch)
      fcntl.flock(f, fcntl.LOCK_UN)
    print(f"\nAppended {len(succeeded_eps)} episodes to {extrinsics_path}")

  print(f"\nStage 2 complete! {len(succeeded_eps)}/{len(target_eps)} episodes succeeded.")
