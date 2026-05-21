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
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F
import torch.optim as optim
import trimesh
from tqdm import tqdm
import yourdfpy


# ---------------------------------------------------------------------------
# Hardware Setup
# ---------------------------------------------------------------------------
def get_accelerator(force_egl=True):
  """Configure headless GPU hardware acceleration and return active device."""
  if force_egl:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Model Loading (VGGT only for Stage 2)
# ---------------------------------------------------------------------------
def init_calibration_models():
  """Load VGGT model for camera extrinsics estimation."""
  device = get_accelerator()
  print(f"🚀 Launching VGGT model onto {device} | CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
  if not torch.cuda.is_available():
    print("⚠️ WARNING: PyTorch cannot find a valid CUDA device.")

  # Inject third-party repo paths (droid/third_party/ lives next to this script)
  vendor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party")
  for pkg in ["vggt"]:
    path = os.path.join(vendor_dir, pkg)
    if path not in sys.path:
      sys.path.append(path)

  from vggt.models.vggt import VGGT
  from vggt.utils.load_fn import load_and_preprocess_images
  from vggt.utils.pose_enc import pose_encoding_to_extri_intri

  vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

  print("  ✅ VGGT model loaded.")
  return vggt_model, load_and_preprocess_images, pose_encoding_to_extri_intri


# ---------------------------------------------------------------------------
# 2. Metadata & Data Loading
# ---------------------------------------------------------------------------
def load_metadata():
  """Load global dataset JSON mappings (same as Stage 1)."""
  root_path = os.path.expanduser("~/droid_data/meta/1.0.1")
  os.makedirs(root_path, exist_ok=True)

  base_url = "https://huggingface.co/KarlP/droid/resolve/main"
  files = [
      "camera_serials.json",
      "cam2base_extrinsic_superset.json",
  ]

  print(f"⬇️ Synchronizing metadata to {root_path}...")
  for f in files:
    if not os.path.exists(os.path.join(root_path, f)):
      os.system(f"wget -q -nc -P {root_path} {base_url}/{f}")

  def load_json(name):
    with open(os.path.join(root_path, name), "r") as f:
      return json.load(f)

  serials_db = load_json("camera_serials.json")
  extrinsics_db = load_json("cam2base_extrinsic_superset.json")

  print(f"✅ Metadata ready!")
  return serials_db, extrinsics_db


def load_depth_data(episode_id, depth_root="~/droid_data/output/mv-tap/droid/depth"):
  """Reconstruct scene_constants from Stage 1 disk outputs.

  Reads:
    robot.npz           → joint_positions, gripper_positions, T_ee_base_all,
                          T_cam_ee_init, wrist_serial, valid_indices
    <cam>/calibration.npz → K matrices, baseline
    <cam>/raw_depth.npz   → depth (uint16 mm → float32 meters)
    <cam>/video_left.mp4  → first frame (for VGGT)

  Returns:
    scene_constants dict matching the Stage 1 in-memory format.
  """
  ep_dir = os.path.abspath(
      os.path.expanduser(os.path.join(depth_root, episode_id))
  )
  if not os.path.isdir(ep_dir):
    raise FileNotFoundError(f"Depth output not found: {ep_dir}")

  print(f"  📂 Loading depth data from {ep_dir}...")

  # --- Robot kinematics ---
  robot_data = np.load(os.path.join(ep_dir, "robot.npz"), allow_pickle=True)
  wrist_serial = str(robot_data["wrist_serial"]) if "wrist_serial" in robot_data else None

  robot = {
      "joint_positions": robot_data["joint_positions"].astype(np.float32),
      "gripper_positions": robot_data["gripper_positions"].astype(np.float32),
      "T_ee_base_all": robot_data["T_ee_base_all"].astype(np.float32),
      "T_cam_ee_init": robot_data["T_cam_ee_init"].astype(np.float32),
  }

  valid_indices = robot_data.get("valid_indices")

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
      cam_data["baseline"] = float(calib["baseline"])

    # Depth (uint16 mm → float32 meters)
    depth_path = os.path.join(cam_path, "raw_depth.npz")
    if os.path.exists(depth_path):
      depth_uint16 = np.load(depth_path)["depth"]
      cam_data["raw_depth"] = depth_uint16.astype(np.float32) / 1000.0

    # First RGB frame (for VGGT)
    video_path = os.path.join(cam_path, "video_left.mp4")
    if os.path.exists(video_path):
      cap = cv2.VideoCapture(video_path)
      ret, frame = cap.read()
      cap.release()
      if ret:
        cam_data["first_frame_rgb"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    camera[cam_id] = cam_data

  scene_constants = {
      "meta": {
          "episode_id": episode_id,
          "wrist_serial": wrist_serial,
          "valid_indices": valid_indices,
      },
      "robot": robot,
      "camera": camera,
  }

  n_frames = len(robot["joint_positions"])
  n_cams = len(camera)
  print(f"  ✅ Loaded: {n_cams} cameras, {n_frames} frames.")
  return scene_constants


# ---------------------------------------------------------------------------
# 3. Utility Functions
# ---------------------------------------------------------------------------
def make_4x4(vec_6d):
  """Convert 6DoF vector [x, y, z, rx, ry, rz] to 4x4 homogeneous transform."""
  transform = np.eye(4)
  transform[:3, :3] = R.from_euler("xyz", vec_6d[3:]).as_matrix()
  transform[:3, 3] = vec_6d[:3]
  return transform


def axis_angle_to_matrix(rot_vec):
  """Differentiable Rodrigues rotation formula with small-angle Taylor fallback."""
  theta2 = torch.sum(rot_vec ** 2)
  theta = torch.sqrt(theta2 + 1e-16)
  k = rot_vec / theta

  K = torch.zeros((3, 3), device=rot_vec.device)
  K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]

  R_exact = torch.eye(3, device=rot_vec.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.mm(K, K)

  K_approx = torch.zeros_like(K)
  K_approx[0, 1], K_approx[0, 2], K_approx[1, 0], K_approx[1, 2], K_approx[2, 0], K_approx[2, 1] = -rot_vec[2], rot_vec[1], rot_vec[2], -rot_vec[0], -rot_vec[1], rot_vec[0]

  return torch.where(theta2 < 1e-8, torch.eye(3, device=rot_vec.device) + K_approx, R_exact)


def make_T(delta, device):
  """Build incremental 4x4 from a 6D parameter vector."""
  rot = axis_angle_to_matrix(delta[:3])
  t = delta[3:].unsqueeze(1)
  T_top = torch.cat([rot, t], dim=1)
  T_bottom = torch.tensor([[0., 0., 0., 1.]], device=device, dtype=torch.float32)
  return torch.cat([T_top, T_bottom], dim=0)


# ---------------------------------------------------------------------------
# 4. yourdfpy-based Tensor Robot Renderer
# ---------------------------------------------------------------------------
class TensorRobotRenderer:
  """High-speed robot point cloud renderer using yourdfpy forward kinematics."""

  _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
  _DEFAULT_URDF = os.path.join(
      _SCRIPT_DIR, "third_party", "PointWorld", "assets", "franka_description",
      "franka_panda_robotiq_2f85_og.urdf",
  )

  def __init__(self, urdf_path=None, device="cuda", total_samples=100000):
    if urdf_path is None:
      urdf_path = self._DEFAULT_URDF
    self.device = device
    self.dtype = torch.float32
    self.total_samples = total_samples
    print(f"⚡ [TensorRobotRenderer] Loading yourdfpy model from {urdf_path}...")
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


# ---------------------------------------------------------------------------
# 5. VGGT Visual Pose Estimation
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

  idx = np.random.choice(P_cam.shape[1], 2000, replace=(P_cam.shape[1] <= 2000))
  return torch.tensor(P_cam[:, idx], dtype=torch.float32, device=device)


def run_global_joint_alignment(scene_constants, prev_scene_state, tensor_renderer,
                                lr=0.001, n_steps=500, robot_weight=1.0, stage_name="Stage 3"):
  """Global joint optimization: Chamfer environment stitching + Robot depth + Wrist depth."""
  print(f"\n🌍 {stage_name}: Global joint optimization (Chamfer + Robot + Wrist, lr={lr})...")
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
    pc1 = get_cam_points_local_t(t, scene_constants['camera'][cam1], device)
    pc2 = get_cam_points_local_t(t, scene_constants['camera'][cam2], device)
    pcw = get_cam_points_local_t(t, scene_constants['camera'][wrist_cam], device)

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
            f"Rob1: {l_rob1.item():.4f} | Rob2: {l_rob2.item():.4f} | Wrst: {l_wrist.item():.4f} | "
            f"BG Overlap: {bg_overlap:.1f}% | "
            f"Shift → C1: {shift_c1:.2f}mm, C2: {shift_c2:.2f}mm, W: {shift_w:.2f}mm")

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
# 11. Export
# ---------------------------------------------------------------------------
def export_extrinsics(scene_constants, scene_state,
                      export_root="~/droid_data/output/mv-tap/droid/extrinsics",
                      stage_suffix=None):
  """Save calibrated extrinsics for all cameras.

  Output layout:
    <episode_id>/extrinsics.npz            (final)
    <episode_id>/extrinsics_stage1.npz     (after VGGT / init)
    <episode_id>/extrinsics_stage2.npz     (after robot alignment)
    <episode_id>/extrinsics_stage3.npz     (after joint optimization)

      Keys per camera:
        <cam_id>_base_extrinsic  → (4, 4) static extrinsic matrix
        <cam_id>_extrinsics      → (N, 4, 4) per-frame trajectory
  """
  ep_str = scene_constants["meta"]["episode_id"]
  out_dir = os.path.abspath(os.path.expanduser(os.path.join(export_root, ep_str)))
  os.makedirs(out_dir, exist_ok=True)

  fname = f"extrinsics_{stage_suffix}.npz" if stage_suffix else "extrinsics.npz"

  save_dict = {}
  for cam_id, state in scene_state.items():
    if state.get("base_extrinsic") is None or state.get("extrinsics") is None:
      continue
    save_dict[f"{cam_id}_base_extrinsic"] = state["base_extrinsic"].astype(np.float32)
    save_dict[f"{cam_id}_extrinsics"] = state["extrinsics"].astype(np.float32)

  # Also save wrist serial for downstream convenience
  save_dict["wrist_serial"] = np.array(scene_constants["meta"]["wrist_serial"])

  np.savez_compressed(os.path.join(out_dir, fname), **save_dict)
  print(f"  💾 Extrinsics saved to {out_dir}/{fname}")


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
  args = parser.parse_args()

  print("🚀 DROID Stage 2: Camera Extrinsics Calibration Pipeline")
  device = get_accelerator()
  vggt_model, load_fn, pose_fn = init_calibration_models()
  serials_db, extrinsics_db = load_metadata()

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

  # Initialize tensor renderer once (shared across all episodes)
  tensor_renderer = TensorRobotRenderer(device=device)

  succeeded_eps = []

  for idx, ep_id in enumerate(target_eps):
    print(f"\n🎬 [{idx + 1}/{len(target_eps)}] Processing Episode: {ep_id}")

    try:
      # Load Stage 1 outputs
      scene_constants = load_depth_data(ep_id, args.depth_root)

      # Stage 0: Initialize from dataset extrinsics (if available)
      init_scene_state = init_camera_states(scene_constants, extrinsics_db)
      all_extrinsics_exist = all(
          state["extrinsics"] is not None
          for state in init_scene_state.values()
      )

      # Stage 1: VGGT visual anchoring (only if init extrinsics incomplete)
      if all_extrinsics_exist:
        print("  ✅ Full pre-calibrated extrinsics found, using Dataset Init as base.")
        stage1_scene_state = init_scene_state
        vggt_scene_state = None
      else:
        print("  ⚠️ Incomplete extrinsics, running VGGT visual anchoring...")
        vggt_scene_state = vggt_warmup_extrinsics(
            scene_constants, vggt_model, load_fn, pose_fn, device,
        )
        stage1_scene_state = vggt_scene_state

      # Save Stage 1 extrinsics
      export_extrinsics(scene_constants, stage1_scene_state, stage_suffix="stage1")

      # Stage 2: Unified camera-robot alignment (external + wrist)
      stage2_state = run_stage2_alignment(
          scene_constants, tensor_renderer, stage1_scene_state,
      )
      export_extrinsics(scene_constants, stage2_state, stage_suffix="stage2")

      # Stage 3: Global joint optimization (Chamfer + Robot + Wrist)
      final_state = run_global_joint_alignment(
          scene_constants, stage2_state, tensor_renderer,
          lr=0.001, n_steps=500, robot_weight=1.0, stage_name="Stage 3",
      )
      export_extrinsics(scene_constants, final_state, stage_suffix="stage3")

      # Final export (canonical name)
      export_extrinsics(scene_constants, final_state)
      succeeded_eps.append(ep_id)
      print(f"  ✅ Episode {ep_id} completed successfully.")

    except Exception as e:
      print(f"  ❌ Episode {ep_id} failed: {e}")
      import traceback
      traceback.print_exc()
      continue

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
