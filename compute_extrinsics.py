"""DROID Stage 2: Camera Extrinsics Calibration Pipeline.

Multi-stage camera extrinsics calibration using differentiable rendering,
point cloud alignment, and VGGT visual anchoring. Reads Stage 1 outputs
(depth, calibration, robot kinematics, video) and produces optimized 4x4
extrinsic matrices for all cameras.

Pipeline:
  Stage 0: Read dataset extrinsics (if available in metadata)
  Stage 1: VGGT visual-physical chain anchoring (first frame only)
  Stage 2: Unified camera-robot alignment (external + wrist in one loop)
  Stage 3: Global joint optimization (Chamfer + Robot + Wrist)
"""

import argparse
import copy
import fcntl
import gc
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from core.geometry import make_4x4, make_T
from core.io import get_accelerator, load_depth_data, load_metadata
from core.physics import TensorRobotRenderer


# ---------------------------------------------------------------------------
# 1. Model Loading (VGGT only for Stage 2)
# ---------------------------------------------------------------------------
def init_calibration_models():
  """Load VGGT model for camera extrinsics estimation.

  Installs the vggt package on-demand if not already installed.
  """
  import subprocess

  device = get_accelerator()
  print(f"🚀 Launching VGGT model onto {device} | CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
  if not torch.cuda.is_available():
    print("⚠️ WARNING: PyTorch cannot find a valid CUDA device.")

  # Install vggt package on-demand from GitHub (skipped if already installed)
  try:
    import vggt  # noqa: F401
  except ImportError:
    print("  📦 Installing vggt package from GitHub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "git+https://github.com/facebookresearch/vggt.git"])

  from vggt.models.vggt import VGGT
  from vggt.utils.load_fn import load_and_preprocess_images
  from vggt.utils.pose_enc import pose_encoding_to_extri_intri

  vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

  print("  ✅ VGGT model loaded.")
  return vggt_model, load_and_preprocess_images, pose_encoding_to_extri_intri


# ---------------------------------------------------------------------------
# 2. Metadata & Data Loading
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 3. VGGT Visual Pose Estimation
# ---------------------------------------------------------------------------
def estimate_multi_camera_vggt(img_list, vggt_model, load_fn, pose_fn, device):
  """Estimate relative camera poses from N images using VGGT.

  The first image is treated as the reference frame origin.
  Returns a list of (N-1) relative T matrices from ref to each target.
  """
  dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

  filenames = []
  for i, img in enumerate(img_list):
    fname = f"/tmp/tmp_vggt_{i}.png"
    cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    filenames.append(fname)

  images = load_fn(filenames).to(device)
  images_input = images.unsqueeze(0)

  with torch.inference_mode():
    with torch.autocast(device_type="cuda", dtype=dtype):
      aggregated_tokens_list, ps_idx = vggt_model.aggregator(images_input)
      pose_enc = vggt_model.camera_head(aggregated_tokens_list)[-1]
      extrinsic, _ = pose_fn(pose_enc, images_input.shape[-2:])

  T_ref_to_tgts = []
  for i in range(1, len(img_list)):
    ext_mat = extrinsic[0, i].cpu().numpy()
    T = np.eye(4)
    T[:3, :] = ext_mat
    T_ref_to_tgts.append(T)

  return T_ref_to_tgts


# ---------------------------------------------------------------------------
# 6. Stage 0: Read Dataset Extrinsics
# ---------------------------------------------------------------------------
def init_camera_states(scene_constants, extrinsics_db):
  """Assemble initial 3D camera states from dataset metadata."""
  print("  🌐 Initializing camera 3D states...")
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


# ---------------------------------------------------------------------------
# 7. Stage 1: VGGT Visual-Physical Chain Anchoring
# ---------------------------------------------------------------------------
def vggt_warmup_extrinsics(scene_constants, vggt_model, load_fn, pose_fn, device):
  """Estimate absolute camera poses via VGGT on the first frame."""
  print("\n🌍 Stage 1: VGGT visual-physical chain anchoring (first frame only)...")
  wrist_serial = scene_constants["meta"]["wrist_serial"]
  ext_cams = [cam for cam in scene_constants["camera"].keys() if cam != wrist_serial]

  ref_cam = ext_cams[0]
  other_cams = ext_cams[1:]

  new_scene_state = {}
  robot_data = scene_constants["robot"]
  n_frames = len(robot_data["joint_positions"])

  # Wrist camera: full kinematic trajectory
  new_scene_state[wrist_serial] = {
      "base_extrinsic": robot_data["T_cam_ee_init"],
      "extrinsics": robot_data["T_ee_base_all"] @ robot_data["T_cam_ee_init"],
  }

  # Read first frame from each camera
  print("  📸 Extracting first-frame images for VGGT inference...")
  img_ref = scene_constants["camera"][ref_cam]["first_frame_rgb"]
  img_others = [scene_constants["camera"][cam]["first_frame_rgb"] for cam in other_cams]
  img_wrist = scene_constants["camera"][wrist_serial]["first_frame_rgb"]

  img_list = [img_ref] + img_others + [img_wrist]
  T_rel_list = estimate_multi_camera_vggt(img_list, vggt_model, load_fn, pose_fn, device)

  T_ref_to_others = T_rel_list[:-1]
  T_ref_to_wrist = T_rel_list[-1]

  # Chain rule: ref → base via wrist GT
  T_ee_base_first = scene_constants["robot"]["T_ee_base_all"][0]
  T_cam_ee = scene_constants["robot"]["T_cam_ee_init"]
  T_wrist_to_base_first = T_ee_base_first @ T_cam_ee

  T_ref_to_base = T_wrist_to_base_first @ T_ref_to_wrist

  new_scene_state[ref_cam] = {
      "base_extrinsic": T_ref_to_base,
      "extrinsics": np.tile(T_ref_to_base, (n_frames, 1, 1)),
  }

  for tgt_cam, T_ref_to_tgt in zip(other_cams, T_ref_to_others):
    T_tgt_to_base = T_ref_to_base @ np.linalg.inv(T_ref_to_tgt)
    new_scene_state[tgt_cam] = {
        "base_extrinsic": T_tgt_to_base,
        "extrinsics": np.tile(T_tgt_to_base, (n_frames, 1, 1)),
    }

  print("  ✅ VGGT anchoring complete!")
  return new_scene_state


def init_extrinsics(scene_constants, extrinsics_db, device,
                    vggt_models=None):
  """Initialize camera extrinsics: dataset first, VGGT fallback.

  Combines Stage 0 (dataset init) and Stage 1 (VGGT anchoring) into a
  single call.  If all cameras have pre-calibrated extrinsics in the
  dataset metadata, VGGT is skipped entirely.

  Args:
    scene_constants: Scene data dict.
    extrinsics_db: Dict of pre-calibrated extrinsics keyed by episode_id.
    device: Torch device.
    vggt_models: Optional tuple ``(vggt_model, load_fn, pose_fn)`` to
        reuse a previously loaded VGGT model.  If ``None`` and VGGT is
        needed, the model is loaded on demand.

  Returns:
    scene_state: Dict with ``base_extrinsic`` and ``extrinsics`` per camera.
    vggt_models: Tuple ``(vggt_model, load_fn, pose_fn)``, or ``None``
        if VGGT was never loaded.  Callers should cache this across
        episodes to avoid reloading the 1B model.
  """
  # Stage 0: try dataset extrinsics
  scene_state = init_camera_states(scene_constants, extrinsics_db)

  all_extrinsics_exist = all(
      state["extrinsics"] is not None for state in scene_state.values()
  )

  if all_extrinsics_exist:
    print("  ✅ Full pre-calibrated extrinsics found, skipping VGGT.")
    return scene_state, vggt_models

  # Stage 1: VGGT visual anchoring (lazy-load model on first use)
  if vggt_models is None:
    print("  📦 Loading VGGT model (first use)...")
    vggt_models = init_calibration_models()
  vggt_model, load_fn, pose_fn = vggt_models

  scene_state = vggt_warmup_extrinsics(
      scene_constants, vggt_model, load_fn, pose_fn, device)

  return scene_state, vggt_models


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
  print("\n🦾 Stage 2: Unified camera-robot alignment (external + wrist)...")
  device = tensor_renderer.device
  wrist_cam = scene_constants['meta']['wrist_serial']
  stage2_scene_state = copy.deepcopy(stage1_scene_state)
  T_ee_base_all = scene_constants['robot']['T_ee_base_all']

  for cam in scene_constants['camera'].keys():
    is_wrist = (cam == wrist_cam)
    mode = "wrist (gripper-only)" if is_wrist else "external (full body)"
    print(f"\n  📷 Optimizing [{mode}] camera: [{cam}] ...")

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
                                track_anchors=None, track_weight=0.0,
                                chamfer_n_points=2000,
                                stage_name="Stage 3"):
  """Global joint optimization: Chamfer + Robot depth + Wrist depth + optional 2D tracks.

  When track_anchors and track_weight > 0 are provided, the 2D track
  reprojection loss is added to the same optimization loop as Chamfer and
  Robot depth.  This ensures all constraints pull together without any
  single signal dominating.

  Args:
    scene_constants: Scene data dict.
    prev_scene_state: Previous extrinsics to refine.
    tensor_renderer: TensorRobotRenderer for robot point clouds.
    lr: Learning rate.
    n_steps: Number of optimization steps.
    robot_weight: Weight for robot depth losses.
    track_anchors: Optional dict from prepare_track_anchors(). When provided
        with track_weight > 0, adds 2D track reprojection constraints.
    track_weight: Weight for 2D track reprojection loss (0 = off).
    stage_name: Display name for logging.
  """
  use_tracks = (track_anchors is not None and track_weight > 0)
  track_tag = f" + Tracks(w={track_weight})" if use_tracks else ""
  print(f"\n🌍 {stage_name}: Global joint optimization "
        f"(Chamfer + Robot + Wrist{track_tag}, lr={lr})...")
  device = tensor_renderer.device
  wrist_cam = scene_constants['meta']['wrist_serial']
  ext_cams = [c for c in scene_constants['camera'].keys() if c != wrist_cam]
  cam1, cam2 = ext_cams[0], ext_cams[1]
  n_frames = len(scene_constants['robot']['joint_positions'])
  T_ee_all = scene_constants['robot']['T_ee_base_all']

  # Extract robot physical tensors from shared factory
  print(f"  🔍 Extracting robot physical tensor caches...")
  batch_X1, batch_obs1 = extract_robot_physical_tensors(cam1, scene_constants, tensor_renderer)
  batch_X2, batch_obs2 = extract_robot_physical_tensors(cam2, scene_constants, tensor_renderer)
  batch_P_ee, batch_obs_w = extract_robot_physical_tensors(wrist_cam, scene_constants, tensor_renderer)

  # Extract Chamfer environment point clouds
  print(f"  🔍 Extracting Chamfer environment point clouds...")
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

  # Prepare FK tensor for track loss (full sequence, not just valid Chamfer frames)
  if use_tracks:
    T_ee_all_t = torch.tensor(T_ee_all, dtype=torch.float32, device=device)
    n_track_cams = sum(1 for c in [cam1, cam2, wrist_cam] if c in track_anchors)
    print(f"  🎯 Track anchors available for {n_track_cams} cameras")

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

    # Chamfer: project environment points to world frame (stochastic sampling to prevent OOM)
    B_total = batch_Pc1.shape[0]
    base_ops = 80 * (2000 ** 2)
    max_batch = max(8, int(base_ops / (chamfer_n_points ** 2)))
    sample_size = min(max_batch, B_total)

    if sample_size < B_total:
      sample_idx = torch.randperm(B_total, device=device)[:sample_size]
      s_Pc1 = batch_Pc1[sample_idx]
      s_Pc2 = batch_Pc2[sample_idx]
      s_Pcw = batch_Pcw[sample_idx]
      s_Tee = batch_Tee[sample_idx]
    else:
      s_Pc1, s_Pc2, s_Pcw, s_Tee = batch_Pc1, batch_Pc2, batch_Pcw, batch_Tee

    bc1 = (T1_opt @ s_Pc1)[:, :3, :].transpose(1, 2)
    bc2 = (T2_opt @ s_Pc2)[:, :3, :].transpose(1, 2)
    T_wrist_c2w = s_Tee @ Tee_opt
    bcw = torch.bmm(T_wrist_c2w, s_Pcw)[:, :3, :].transpose(1, 2)

    l12, o12 = batched_chamfer_distance(bc1, bc2, device)
    l1w, o1w = batched_chamfer_distance(bc1, bcw, device)
    l2w, o2w = batched_chamfer_distance(bc2, bcw, device)
    loss_chamfer = l12 + l1w + l2w

    # Robot depth losses
    l_rob1 = compute_robot_loss(batch_X1, T1_opt, K_t1, batch_obs1, depth_tolerance=0.15)
    l_rob2 = compute_robot_loss(batch_X2, T2_opt, K_t2, batch_obs2, depth_tolerance=0.15)
    l_wrist = compute_robot_loss(batch_P_ee, Tee_opt, K_t_w, batch_obs_w, depth_tolerance=float('inf'))

    loss_total = chamfer_weight * loss_chamfer + robot_weight * (l_rob1 + l_rob2 + l_wrist)

    # 2D track reprojection losses (when enabled)
    loss_track = torch.tensor(0.0, device=device)
    if use_tracks:
      for cam_id in [cam1, cam2, wrist_cam]:
        if cam_id not in track_anchors:
          continue
        d_cam, T_init_cam, K_cam = cam_opt_map[cam_id]
        T_opt_cam = T_init_cam @ make_T(d_cam, device)
        scheme = track_anchors[cam_id]['scheme']
        if scheme == 'robot_fk':
          continue  # robot_fk anchors are eval-only, no P_cam0 for optimization
        l_trk = compute_track_reproj_loss(
            track_anchors[cam_id], T_opt_cam, K_cam, T_ee_all_t,
            scheme, device)
        loss_track = loss_track + l_trk
      loss_total = loss_total + track_weight * loss_track

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
      if use_tracks:
        log_parts.append(f"Track: {loss_track.item():.2f}")
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
# 11. Stage 5: 2D Track-Based Extrinsics Refinement
# ---------------------------------------------------------------------------

def prepare_track_anchors(scene_constants, scene_state, pb_renderer, device):
  """Lift tracked 2D points at t=0 to 3D and classify as robot or background.

  For static cameras (Scheme A): selects robot-region tracks — points are
  assumed to be on the EE link, moved by forward kinematics.
  For the wrist camera (Scheme B): selects background tracks — points are
  static in world frame while the camera moves.

  Args:
    scene_constants: Scene data dict with tracks_2d / vis_2d populated.
    scene_state: Current extrinsics estimates for each camera.
    pb_renderer: PyBulletRenderer for robot mask rendering.
    device: Torch device.

  Returns:
    Dict[cam_id, dict] with per-camera anchor data:
      P_cam0: (N, 4) float32, 3D points in camera frame at t=0 (homogeneous)
      tracks_2d: (T, N, 2) float32, selected track positions over time
      vis: (T, N) bool, visibility flags
      scheme: "robot" or "background"
  """
  wrist_cam = scene_constants['meta']['wrist_serial']
  anchors = {}

  print("  📌 Preparing track anchors from t=0...")

  for cam_id in scene_constants['camera']:
    cam_data = scene_constants['camera'][cam_id]

    if 'tracks_2d' not in cam_data or 'raw_depth' not in cam_data:
      print(f"    ⚠️ [{cam_id}] Missing tracks_2d or depth, skipping.")
      continue

    tracks = cam_data['tracks_2d']   # (T, N, 2) float32
    vis = cam_data['vis_2d']         # (T, N) bool
    T_frames, N_total, _ = tracks.shape
    is_wrist = (cam_id == wrist_cam)

    h_img, w_img = cam_data['raw_depth'][0].shape[:2]
    K = cam_data['K_mat']

    # Render masks at t=0 using current extrinsics
    ext_t0 = scene_state[cam_id]['extrinsics'][0]
    pb_renderer.update_robot_pose(
        scene_constants['robot']['joint_positions'][0],
        gripper_state=scene_constants['robot']['gripper_positions'][0])

    # Track positions at t=0
    u0 = tracks[0, :, 0]
    v0 = tracks[0, :, 1]
    u0i = np.clip(np.round(u0).astype(int), 0, w_img - 1)
    v0i = np.clip(np.round(v0).astype(int), 0, h_img - 1)

    if is_wrist:
      # Scheme B: wrist camera — select BACKGROUND tracks
      # Use SAM consensus gripper mask (from gripper_mask.npz).
      # sam_real_masks is (T, H, W) with mask only at closed-gripper frames
      # (same static consensus mask broadcast). OR across time gives the mask.
      sam_masks = cam_data.get('sam_real_masks')
      if sam_masks is None or not sam_masks.any():
        print(f"    ⚠️ [{cam_id}] No SAM gripper mask, skipping wrist track eval.")
        continue

      consensus = sam_masks.any(axis=0)  # static gripper mask in camera frame
      kernel = np.ones((15, 15), np.uint8)
      robot_mask_d = cv2.dilate(
          consensus.astype(np.uint8), kernel, iterations=1) > 0
      on_robot = robot_mask_d[v0i, u0i]
      selected = ~on_robot
      scheme = "background"

      # Only evaluate on closed-gripper frames (< 0.05) where the SAM mask
      # is accurate. No PyBullet rendering needed at all.
      gripper_pos = scene_constants['robot']['gripper_positions']
      gripper_closed_frames = gripper_pos < 0.05
      n_closed = gripper_closed_frames.sum()
      print(f"    🔒 [{cam_id}] Closed-gripper eval frames: "
            f"{n_closed}/{len(gripper_pos)}  mask: SAM")
    else:
      # Scheme A: static camera — ALL robot points via URDF FK
      # Use URDFKinematicsTracker to bind surface points to link frames at t=0,
      # propagate via FK, and get FK-based 2D predictions + visibility.
      from core.tracking import URDFKinematicsTracker
      urdf_tracker = URDFKinematicsTracker(pb_renderer)
      fk_result = urdf_tracker.extract_robot_tracks(
          cam_id, scene_constants, scene_state)
      traj_3d, traj_2d_fk, vis_fk, robot_indices = fk_result

      if robot_indices is None or len(robot_indices) < 5:
        print(f"    ⚠️ [{cam_id}] URDF tracker found <5 robot points, skipping.")
        continue

      # FK 2D = predicted, tracker 2D = target
      tracker_2d = tracks[:, robot_indices]  # (T, N_robot, 2)
      tracker_vis = vis[:, robot_indices]    # (T, N_robot)

      # Combined visibility: both FK visible and tracker visible
      combined_vis = vis_fk & tracker_vis  # (T, N_robot)

      # L1 pixel error: |FK_pred - tracker_pred|
      pixel_err = np.abs(traj_2d_fk - tracker_2d).sum(axis=-1)  # (T, N_robot)

      valid = combined_vis & (pixel_err < 500)  # sanity cap
      if valid.sum() < 10:
        print(f"    ⚠️ [{cam_id}] Too few valid FK-vs-tracker comparisons, skipping.")
        continue

      mean_err = float(pixel_err[valid].mean())
      median_err = float(np.median(pixel_err[valid]))
      n_pts = len(robot_indices)
      avg_vis_pct = combined_vis.mean() * 100
      print(f"    ✅ [{cam_id}] robot FK: {n_pts} points "
            f"(avg vis: {avg_vis_pct:.0f}%, "
            f"mean={mean_err:.1f}px, median={median_err:.1f}px)")

      # Store directly as per-camera metric (no anchor needed)
      anchors[cam_id] = {
          'scheme': 'robot_fk',
          'mean_px': mean_err,
          'median_px': median_err,
      }
      continue  # skip the anchor building below

    # --- Below only runs for wrist "background" scheme ---
    depth_0 = cam_data['raw_depth'][0].astype(np.float32)
    z0 = depth_0[v0i, u0i]

    # Diagnostic: show how many points survive each filter
    n_total = len(u0)
    n_not_robot = selected.sum()
    n_vis0 = (selected & vis[0]).sum()
    n_has_depth = (z0 > 0.01).sum()
    n_depth_range = ((z0 > 0.05) & (z0 < 3.0)).sum()
    print(f"    📊 [{cam_id}] Filter chain: total={n_total} "
          f"→ not_gripper={n_not_robot} "
          f"→ vis[0]={n_vis0} "
          f"→ has_depth(>0.01)={n_has_depth} "
          f"→ depth_range(0.05-3m)={n_depth_range}")

    has_depth = (z0 > 0.05) & (z0 < 3.0)
    selected = selected & vis[0] & has_depth

    if selected.sum() < 5:
      print(f"    ⚠️ [{cam_id}] background: only {selected.sum()} initial tracks, skipping.")
      continue

    sel_idx = np.where(selected)[0]

    # Keep only tracks visible in >20% of frames
    vis_frac = vis[:, sel_idx].mean(axis=0)
    good = vis_frac > 0.2
    sel_idx = sel_idx[good]

    if len(sel_idx) < 5:
      print(f"    ⚠️ [{cam_id}] background: only {len(sel_idx)} well-visible tracks, skipping.")
      continue

    # Cap at 500 for memory efficiency
    if len(sel_idx) > 500:
      rng = np.random.RandomState(42)
      sel_idx = rng.choice(sel_idx, 500, replace=False)
      sel_idx.sort()

    # Lift to 3D in camera frame at t=0
    u_s = u0[sel_idx].astype(np.float32)
    v_s = v0[sel_idx].astype(np.float32)
    z_s = z0[sel_idx].astype(np.float32)
    x_c = (u_s - K[0, 2]) * z_s / K[0, 0]
    y_c = (v_s - K[1, 2]) * z_s / K[1, 1]
    P_cam0 = np.stack([x_c, y_c, z_s, np.ones_like(z_s)], axis=-1)

    avg_vis = vis[:, sel_idx].mean() * 100
    print(f"    ✅ [{cam_id}] background: {len(sel_idx)} tracks "
          f"(avg vis: {avg_vis:.0f}%, depth: sensor)")

    anchor_data = {
        'P_cam0': torch.tensor(P_cam0, dtype=torch.float32, device=device),
        'tracks_2d': torch.tensor(
            tracks[:, sel_idx], dtype=torch.float32, device=device),
        'vis': torch.tensor(
            vis[:, sel_idx], dtype=torch.bool, device=device),
        'scheme': 'background',
    }
    anchor_data['eval_frame_mask'] = torch.tensor(
        gripper_closed_frames, dtype=torch.bool, device=device)
    anchors[cam_id] = anchor_data

  return anchors


def compute_track_reproj_loss(anchor, T_opt, K, T_ee_all, scheme, device,
                              return_median=False):
  """Differentiable 2D track reprojection loss for one camera.

  Scheme A (robot tracks on static camera):
    Points are on the EE link. At t=0 they are lifted to EE-local coords:
      P_ee = T_ee(0)^{-1} @ T_cam_to_base @ P_cam0
    At time t, they are projected back through FK:
      P_cam(t) = T_base_to_cam @ T_ee(t) @ P_ee

  Scheme B (background tracks on wrist camera):
    Points are static in world frame. At t=0 they are lifted to world:
      P_world = T_ee(0) @ T_cam_ee @ P_cam0
    At time t, the wrist camera has moved:
      P_cam(t) = inv(T_ee(t) @ T_cam_ee) @ P_world

  Both schemes produce predicted 2D positions via standard pinhole projection,
  compared to tracked positions using L1 pixel error.

  Args:
    anchor: Dict from prepare_track_anchors: P_cam0, tracks_2d, vis, scheme.
    T_opt: (4, 4) Current optimized extrinsic (cam-to-base for static,
        cam-to-ee for wrist). Differentiable w.r.t. delta.
    K: (3, 3) Camera intrinsics tensor.
    T_ee_all: (T, 4, 4) FK transforms for all timesteps.
    scheme: "robot" or "background".
    device: Torch device.
    return_median: If True, return (mean, median) tuple for evaluation.

  Returns:
    Scalar mean L1 loss, or (mean, median) tuple when return_median=True.
  """
  P_cam0 = anchor['P_cam0']       # (N, 4)
  targets = anchor['tracks_2d']   # (T, N, 2)
  vis = anchor['vis']             # (T, N)

  if scheme in ("robot", "gripper"):
    # Scheme A: static camera, gripper/EE tracks
    # T_opt = T_cam_to_base (static camera extrinsic)
    T_base_to_cam = torch.linalg.inv(T_opt)
    T_ee_0_inv = torch.linalg.inv(T_ee_all[0])

    # P_world at t=0 → P_ee (constant on EE link)
    P_world0 = T_opt @ P_cam0.T                  # (4, N)
    P_ee = T_ee_0_inv @ P_world0                  # (4, N)

    # For all t: project through FK → camera
    # P_world(t) = T_ee(t) @ P_ee → P_cam(t) = T_base_to_cam @ P_world(t)
    P_ee_batch = P_ee.unsqueeze(0)                # (1, 4, N)
    P_world_all = T_ee_all @ P_ee_batch           # (T, 4, N)
    P_cam_all = T_base_to_cam.unsqueeze(0) @ P_world_all  # (T, 4, N)

  elif scheme == "background":
    # Scheme B: wrist camera, background tracks
    # T_opt = T_cam_ee (hand-eye extrinsic)
    # P_world = T_ee(0) @ T_cam_ee @ P_cam0 (constant in world)
    T_cam_to_world_0 = T_ee_all[0] @ T_opt       # (4, 4)
    P_world = T_cam_to_world_0 @ P_cam0.T         # (4, N)

    # For all t: inv(T_ee(t) @ T_cam_ee) @ P_world
    T_cam_to_world_all = T_ee_all @ T_opt.unsqueeze(0)   # (T, 4, 4)
    T_world_to_cam_all = torch.linalg.inv(T_cam_to_world_all)
    P_cam_all = T_world_to_cam_all @ P_world.unsqueeze(0)  # (T, 4, N)

  else:
    zero = torch.tensor(0.0, device=device)
    return (zero, zero) if return_median else zero

  # Pinhole projection → predicted 2D
  Z = P_cam_all[:, 2, :].clamp(min=1e-4)                   # (T, N)
  u_pred = K[0, 0] * P_cam_all[:, 0, :] / Z + K[0, 2]     # (T, N)
  v_pred = K[1, 1] * P_cam_all[:, 1, :] / Z + K[1, 2]     # (T, N)
  pred = torch.stack([u_pred, v_pred], dim=-1)              # (T, N, 2)

  # Mean pixel reprojection error (L1: |Δu| + |Δv|)
  pixel_err = (pred - targets).abs().sum(dim=-1)            # (T, N)
  valid = vis & (Z > 0.05)

  # For wrist background: only evaluate on closed-gripper frames
  # where SAM mask is accurate
  if 'eval_frame_mask' in anchor:
    valid = valid & anchor['eval_frame_mask'][:, None]

  if valid.any():
    err = pixel_err[valid]
    if return_median:
      return err.mean(), err.median()
    return err.mean()

  zero = torch.tensor(0.0, device=device)
  return (zero, zero) if return_median else zero


# ---------------------------------------------------------------------------
# 11b. Evaluate Extrinsics Quality
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_extrinsics(scene_constants, scene_state, device,
                        pb_renderer=None, track_anchors=None):
  """Compute extrinsics quality metrics without re-running optimization.

  Can be called after any stage to monitor calibration quality.

  Args:
    scene_constants: Scene data dict.
    scene_state: Current extrinsics state.
    device: Torch device.
    pb_renderer: Optional PyBulletRenderer to reuse. If None, one
        is created temporarily.
    track_anchors: Optional track anchors for 2D reprojection eval.

  Returns:
    Dict with metrics: chamfer_total, robot_loss_*, bg_overlap_pct,
    track_reproj_mean_px, shift_mm_*.
  """
  own_renderer = False
  if pb_renderer is None:
    from core.physics import PyBulletRenderer
    pb_renderer = PyBulletRenderer()
    own_renderer = True

  wrist_cam = scene_constants["meta"]["wrist_serial"]
  ext_cams = [c for c in scene_constants["camera"].keys() if c != wrist_cam]
  cam1, cam2 = ext_cams[0], ext_cams[1]
  n_frames = len(scene_constants["robot"]["joint_positions"])
  T_ee_all = scene_constants["robot"]["T_ee_base_all"]

  metrics = {}

  # --- Robot depth losses (via pybullet_extrinsics) ---
  from core.pybullet_extrinsics import (
      get_foreground_robot_points, get_foreground_gripper_points,
      compute_robot_loss_batched, compute_wrist_loss_batched,
  )
  for cam_id, key_prefix in [(cam1, "cam1"), (cam2, "cam2"),
                              (wrist_cam, "wrist")]:
    try:
      is_wrist = (cam_id == wrist_cam)
      K_np = scene_constants["camera"][cam_id]["K_mat"]
      K_t = torch.tensor(K_np, dtype=torch.float32, device=device)

      cache_X, cache_obs = [], []
      for t in range(n_frames):
        joints = scene_constants["robot"]["joint_positions"][t]
        gripper = scene_constants["robot"]["gripper_positions"][t]
        pb_renderer.update_robot_pose(joints, gripper_state=gripper)

        d_obs = scene_constants["camera"][cam_id]["raw_depth"][t].astype(np.float32)
        ext_t = scene_state[cam_id]["extrinsics"][t]

        if is_wrist:
          pts = get_foreground_gripper_points(
              ext_t, K_np, d_obs, pb_renderer, device)
          if pts is None:
            continue
          T_world_to_ee = np.linalg.inv(T_ee_all[t])
          pts_world = (ext_t @ pts)[:3, :].T
          pts_ee = (T_world_to_ee[:3, :3] @ pts_world.T +
                    T_world_to_ee[:3, 3:4]).T
          cache_X.append(
              torch.tensor(pts_ee, dtype=torch.float32, device=device))
        else:
          pts = get_foreground_robot_points(
              ext_t, K_np, d_obs, pb_renderer, device)
          if pts is None:
            continue
          cache_X.append(pts)

        cache_obs.append(
            torch.tensor(d_obs, dtype=torch.float32, device=device)[None, ...])

      if not cache_X:
        metrics[f"robot_loss_{key_prefix}"] = float("nan")
        continue

      batch_X = torch.stack(cache_X)
      batch_obs = torch.stack(cache_obs)
      T_opt = torch.tensor(
          scene_state[cam_id]["base_extrinsic"],
          dtype=torch.float32, device=device)

      with torch.no_grad():
        if is_wrist:
          loss = compute_wrist_loss_batched(batch_X, T_opt, K_t, batch_obs)
        else:
          loss = compute_robot_loss_batched(batch_X, T_opt, K_t, batch_obs)
      metrics[f"robot_loss_{key_prefix}"] = loss.item()
    except Exception as e:
      metrics[f"robot_loss_{key_prefix}"] = float("nan")

  # --- Chamfer losses ---
  try:
    cache_Pc1, cache_Pc2, cache_Pcw, cache_Tee = [], [], [], []
    for t in range(n_frames):
      pc1 = get_cam_points_local_t(t, scene_constants["camera"][cam1], device)
      pc2 = get_cam_points_local_t(t, scene_constants["camera"][cam2], device)
      pcw = get_cam_points_local_t(
          t, scene_constants["camera"][wrist_cam], device)
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

  # --- Track model error (if anchors provided) ---
  if track_anchors:
    T_ee_t = torch.tensor(T_ee_all, dtype=torch.float32, device=device)
    # Per-scheme accumulators
    bg_means, bg_medians = [], []
    robot_fk_means, robot_fk_medians = [], []
    for cam_id in track_anchors:
      scheme = track_anchors[cam_id]["scheme"]

      if scheme == "robot_fk":
        # Pre-computed in prepare_track_anchors via URDFKinematicsTracker
        robot_fk_means.append(track_anchors[cam_id]["mean_px"])
        robot_fk_medians.append(track_anchors[cam_id]["median_px"])
      elif scheme == "background":
        T_opt = torch.tensor(
            scene_state[cam_id]["base_extrinsic"],
            dtype=torch.float32, device=device)
        K_t = torch.tensor(
            scene_constants["camera"][cam_id]["K_mat"],
            dtype=torch.float32, device=device)
        l_mean, l_median = compute_track_reproj_loss(
            track_anchors[cam_id], T_opt, K_t, T_ee_t, scheme, device,
            return_median=True)
        bg_means.append(l_mean.item())
        bg_medians.append(l_median.item())

    # Primary metric: wrist background (cleanest test of T_cam_ee)
    metrics["track_reproj_wrist_bg_mean_px"] = (
        float(np.mean(bg_means)) if bg_means else float("nan"))
    metrics["track_reproj_wrist_bg_median_px"] = (
        float(np.mean(bg_medians)) if bg_medians else float("nan"))
    # Secondary metric: static camera robot FK vs tracker
    metrics["track_reproj_static_robot_mean_px"] = (
        float(np.mean(robot_fk_means)) if robot_fk_means else float("nan"))
    metrics["track_reproj_static_robot_median_px"] = (
        float(np.mean(robot_fk_medians)) if robot_fk_medians else float("nan"))
    # Overall (all cameras, backward compat)
    all_means = bg_means + robot_fk_means
    all_medians = bg_medians + robot_fk_medians
    metrics["track_reproj_mean_px"] = (
        float(np.mean(all_means)) if all_means else float("nan"))
    metrics["track_reproj_median_px"] = (
        float(np.mean(all_medians)) if all_medians else float("nan"))
  else:
    metrics["track_reproj_mean_px"] = float("nan")
    metrics["track_reproj_median_px"] = float("nan")
    metrics["track_reproj_wrist_bg_mean_px"] = float("nan")
    metrics["track_reproj_wrist_bg_median_px"] = float("nan")
    metrics["track_reproj_static_robot_mean_px"] = float("nan")
    metrics["track_reproj_static_robot_median_px"] = float("nan")

  # --- Extrinsic magnitude from identity ---
  for cam_id in scene_constants["camera"]:
    T = scene_state[cam_id]["base_extrinsic"]
    metrics[f"shift_mm_{cam_id}"] = float(np.linalg.norm(T[:3, 3]) * 1000)

  if own_renderer:
    del pb_renderer

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

  header = f"📊 Metrics after {stage_name}" if stage_name else "📊 Metrics"
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
    <episode_id>/<cam_id>/extrinsics_stage1.json     (after VGGT / init)
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

  print(f"  💾 Extrinsics saved to {ep_dir}/*/{fname}")


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
  parser.add_argument("--stage4", action="store_true",
                       help="Run Stage 4 fine-tuning after Stage 3")
  parser.add_argument("--stage4_lr", type=float, default=0.0001,
                       help="Learning rate for Stage 4 fine-tuning")
  parser.add_argument("--stage4_robot_weight", type=float, default=0.1,
                       help="Robot loss weight for Stage 4 fine-tuning")
  parser.add_argument("--stage4_steps", type=int, default=500,
                       help="Number of optimization steps for Stage 4")
  parser.add_argument("--export_root", type=str,
                       default="~/droid_data/output/mv-tap/droid/extrinsics",
                       help="Root directory for extrinsics output")
  args = parser.parse_args()

  print(f"🚀 DROID Stage 2: Camera Extrinsics Calibration Pipeline [method={args.method}]")
  device = get_accelerator()
  serials_db, _, _, extrinsics_db, _ = load_metadata()

  # VGGT is lazy-loaded only when needed (cached across episodes)
  vggt_models = None

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
  print(f"📋 Selected via distributed rank {args.rank}/{args.world_size} targeting: {len(target_eps)} episodes")

  # Initialize renderer based on method
  if args.method == "pybullet":
    from core.physics import PyBulletRenderer
    from core.pybullet_extrinsics import (
        run_stage2_alignment_pybullet,
        run_global_joint_alignment_pybullet,
    )
    pb_renderer = PyBulletRenderer()
    tensor_renderer = None
    print("  🔧 Using PyBullet rendering engine (V52 style)")
  else:
    pb_renderer = None
    tensor_renderer = TensorRobotRenderer(device=device)
    print("  🔧 Using yourdfpy TensorRobotRenderer (V55 style)")

  succeeded_eps = []

  for idx, ep_id in enumerate(target_eps):
    print(f"\n🎬 [{idx + 1}/{len(target_eps)}] Processing Episode: {ep_id}")

    # Pre-initialize for safe cleanup in finally block
    scene_constants = None
    stage1_scene_state = None
    stage2_state = None
    stage3_state = None
    final_state = None

    try:
      # Load Stage 1 outputs
      scene_constants = load_depth_data(ep_id, args.depth_root)

      # Stage 0+1: Dataset extrinsics → VGGT fallback
      stage1_scene_state, vggt_models = init_extrinsics(
          scene_constants, extrinsics_db, device, vggt_models=vggt_models)
      print_metrics(
          evaluate_extrinsics(scene_constants, stage1_scene_state, device,
                              tensor_renderer=tensor_renderer),
          stage_name="Stage 0+1 (Init)")

      # Save Stage 1 extrinsics
      export_extrinsics(scene_constants, stage1_scene_state,
                        export_root=args.export_root, stage_suffix="stage1")

      # Stage 2: Per-camera independent alignment (external + wrist)
      if args.method == "pybullet":
        stage2_state = run_stage2_alignment_pybullet(
            scene_constants, pb_renderer, stage1_scene_state, device,
        )
      else:
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
        if args.method == "pybullet":
          stage3_state = run_global_joint_alignment_pybullet(
              scene_constants, stage2_state, pb_renderer, device,
              lr=0.001, n_steps=500, robot_weight=1.0, stage_name="Stage 3",
          )
        else:
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

        # Stage 4: Optional fine-tuning (second pass of global joint alignment)
        if args.stage4:
          if args.method == "pybullet":
            final_state = run_global_joint_alignment_pybullet(
                scene_constants, stage3_state, pb_renderer, device,
                lr=args.stage4_lr, n_steps=args.stage4_steps,
                robot_weight=args.stage4_robot_weight, stage_name="Stage 4",
            )
          else:
            final_state = run_global_joint_alignment(
                scene_constants, stage3_state, tensor_renderer,
                lr=args.stage4_lr, n_steps=args.stage4_steps,
                robot_weight=args.stage4_robot_weight, stage_name="Stage 4",
            )
          print_metrics(
              evaluate_extrinsics(scene_constants, final_state, device,
                                  tensor_renderer=tensor_renderer),
              stage_name="Stage 4 (Fine-Tuning)")
        else:
          final_state = stage3_state

        # Final export (canonical name)
        export_extrinsics(scene_constants, final_state,
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
    print(f"\n📝 Appended {len(succeeded_eps)} episodes to {extrinsics_path}")

  print(f"\n🎉 Stage 2 complete! {len(succeeded_eps)}/{len(target_eps)} episodes succeeded.")
