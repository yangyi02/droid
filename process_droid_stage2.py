"""DROID Stage 2: Camera Extrinsics Calibration Pipeline.

Multi-stage camera extrinsics calibration using differentiable rendering,
point cloud alignment, and VGGT visual anchoring. Reads Stage 1 outputs
(depth, calibration, robot kinematics, video) and produces optimized 4x4
extrinsic matrices for all cameras.

Pipeline:
  Stage 0: Read dataset extrinsics (if available in metadata)
  Stage 1: VGGT visual-physical chain anchoring (first frame only)
  Stage 2a: External camera-robot arm rigid alignment
  Stage 2b: Wrist camera-gripper body alignment
  Stage 3: Global joint optimization (Chamfer + Robot + Wrist, lr=0.001)
  Stage 4: Fine-tuning refinement (lr=0.0001, lower robot weight)
"""

import argparse
import copy
import fcntl
import importlib.util
import json
import os
import sys

import cv2
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F
import torch.optim as optim
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
# 1. Model Loading (VGGT only for Stage 2)
# ---------------------------------------------------------------------------
def init_stage2_models():
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


def load_stage1_data(episode_id, stage1_root="~/droid_data/output/mv-tap/droid/stage1"):
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
      os.path.expanduser(os.path.join(stage1_root, episode_id))
  )
  if not os.path.isdir(ep_dir):
    raise FileNotFoundError(f"Stage 1 output not found: {ep_dir}")

  print(f"  📂 Loading Stage 1 data from {ep_dir}...")

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
  K[0, 1], K[0, 2] = -k[2], k[1]
  K[1, 0], K[1, 2] = k[2], -k[0]
  K[2, 0], K[2, 1] = -k[1], k[0]

  R_exact = torch.eye(3, device=rot_vec.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.mm(K, K)

  K_approx = torch.zeros_like(K)
  K_approx[0, 1], K_approx[0, 2] = -rot_vec[2], rot_vec[1]
  K_approx[1, 0], K_approx[1, 2] = rot_vec[2], -rot_vec[0]
  K_approx[2, 0], K_approx[2, 1] = -rot_vec[1], rot_vec[0]
  R_approx = torch.eye(3, device=rot_vec.device) + K_approx

  return torch.where(theta2 < 1e-8, R_approx, R_exact)


def make_delta_T(delta, device):
  """Build incremental 4x4 from a 6D parameter vector."""
  T = torch.eye(4, device=device)
  T[:3, :3] = axis_angle_to_matrix(delta[:3])
  T[:3, 3] = delta[3:]
  return T


# ---------------------------------------------------------------------------
# 4. PyBullet Digital Twin Renderer
# ---------------------------------------------------------------------------
class PyBulletRenderer_Robotiq:
  """Dual-body physics renderer: vanilla Panda arm + Robotiq gripper ghost."""

  # PointWorld lives in droid/third_party/PointWorld/ (next to this script)
  _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
  _DEFAULT_GHOST_URDF = os.path.join(
      _SCRIPT_DIR, "third_party", "PointWorld", "assets", "franka_description",
      "franka_panda_robotiq_2f85_og.urdf",
  )

  def __init__(self, ghost_urdf=None):
    if ghost_urdf is None:
      ghost_urdf = self._DEFAULT_GHOST_URDF
    if p.isConnected():
      p.disconnect()
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    plugin_id = -1
    try:
      if importlib.util.find_spec("eglRendererPlugin"):
        print("  🔍 Found eglRendererPlugin via spec, attempting to load...")
        plugin_id = p.loadPlugin(
            importlib.util.find_spec("eglRendererPlugin").origin,
            "_eglRendererPlugin",
        )
      else:
        print("  � eglRendererPlugin not found via spec, trying direct load...")
        plugin_id = p.loadPlugin("eglRendererPlugin")
    except Exception as e:
      print(f"  ⚠️ EGL plugin check/load raised exception: {e}")

    if plugin_id >= 0:
      print(f"  🔌 PyBullet EGL plugin loaded successfully! (ID: {plugin_id})")
    else:
      print("  ⚠️ WARNING: Failed to load eglRendererPlugin! Falling back to slow TinyRenderer.")

    # Body: vanilla Panda arm (hide hand/finger)
    self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    self.arm_joints = [
        i for i in range(p.getNumJoints(self.robot_id))
        if "panda_joint" in p.getJointInfo(self.robot_id, i)[1].decode("utf-8")
        and p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED
    ]

    self.hidden_robot_links = []
    for i in range(-1, p.getNumJoints(self.robot_id)):
      name = (
          p.getBodyInfo(self.robot_id)[0].decode("utf-8")
          if i == -1
          else p.getJointInfo(self.robot_id, i)[12].decode("utf-8")
      )
      if "hand" in name or "finger" in name:
        p.changeVisualShape(self.robot_id, i, rgbaColor=[0, 0, 0, 0])
        self.hidden_robot_links.append(i)

    # Ghost: PointWorld arm with Robotiq gripper
    self.ghost_id = p.loadURDF(ghost_urdf, useFixedBase=True)
    self.ghost_arm_joints = [
        i for i in range(p.getNumJoints(self.ghost_id))
        if "panda_joint" in p.getJointInfo(self.ghost_id, i)[1].decode("utf-8")
        and p.getJointInfo(self.ghost_id, i)[2] != p.JOINT_FIXED
    ]

    self.gripper_joints = []
    self.gripper_signs = []

    for i in range(p.getNumJoints(self.ghost_id)):
      info = p.getJointInfo(self.ghost_id, i)
      joint_name = info[1].decode("utf-8")
      joint_type = info[2]

      if joint_type != p.JOINT_FIXED and "panda_joint" not in joint_name:
        self.gripper_joints.append(i)
        base_sign = -1 if "right" in joint_name else 1
        if "inner_finger" in joint_name or "follower" in joint_name or "finger_tip" in joint_name:
          self.gripper_signs.append(base_sign * -1)
        else:
          self.gripper_signs.append(base_sign)

    self.hidden_ghost_links = []
    for i in range(-1, p.getNumJoints(self.ghost_id)):
      name = (
          p.getBodyInfo(self.ghost_id)[0].decode("utf-8")
          if i == -1
          else p.getJointInfo(self.ghost_id, i)[12].decode("utf-8")
      )
      if "panda_link" in name:
        p.changeVisualShape(self.ghost_id, i, rgbaColor=[0, 0, 0, 0])
        self.hidden_ghost_links.append(i)

  def _get_projection_matrix(self, intrinsics, width, height):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    near, far = 0.01, 10.0
    return [
        2.0 * fx / width, 0.0, 0.0, 0.0,
        0.0, 2.0 * fy / height, 0.0, 0.0,
        1.0 - 2.0 * cx / width, 2.0 * cy / height - 1.0,
        (far + near) / (near - far), -1.0,
        0.0, 0.0, 2.0 * far * near / (near - far), 0.0,
    ]

  def update_robot_pose(self, joint_angles, gripper_state=None, gripper_width_offset=0.08):
    for i, angle in zip(self.arm_joints, joint_angles):
      p.resetJointState(self.robot_id, i, angle)
    for i, angle in zip(self.ghost_arm_joints, joint_angles):
      p.resetJointState(self.ghost_id, i, angle)

    if gripper_state is not None and len(self.gripper_joints) > 0:
      raw_val = gripper_state[0] if isinstance(gripper_state, (list, np.ndarray)) else gripper_state
      raw_val = np.clip(raw_val, 0.0, 1.0)
      max_urdf_radian = 0.8028
      angle = (raw_val * max_urdf_radian) - gripper_width_offset

      for i, sign in zip(self.gripper_joints, self.gripper_signs):
        p.resetJointState(self.ghost_id, i, angle * sign)

    p.performCollisionDetection()

  def render_depth(self, extrinsics, intrinsics, width, height):
    cam_pos = extrinsics[:3, 3]
    target_pos = extrinsics[:3, 3] + extrinsics[:3, 2]
    view_matrix = p.computeViewMatrix(cam_pos, target_pos, -extrinsics[:3, 1])
    proj_matrix = self._get_projection_matrix(intrinsics, width, height)
    _, _, _, depth_buffer, _ = p.getCameraImage(
        width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    metric_depth = 0.1 / (10.0 - 9.99 * np.reshape(depth_buffer, (height, width)))
    return np.where(metric_depth < 9.9, metric_depth, 0.0)

  def render_mask(self, extrinsics, intrinsics, width, height):
    cam_pos = extrinsics[:3, 3]
    target_pos = extrinsics[:3, 3] + extrinsics[:3, 2]
    view_matrix = p.computeViewMatrix(cam_pos, target_pos, -extrinsics[:3, 1])
    proj_matrix = self._get_projection_matrix(intrinsics, width, height)
    _, _, _, _, seg_buffer = p.getCameraImage(
        width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    )
    seg_array = np.reshape(seg_buffer, (height, width)).astype(np.int32)
    obj_ids = seg_array & 0xFFFFFF
    link_ids = (seg_array >> 24) - 1
    valid_robot = (obj_ids == self.robot_id) & ~np.isin(link_ids, self.hidden_robot_links)
    valid_ghost = (obj_ids == self.ghost_id) & ~np.isin(link_ids, self.hidden_ghost_links)
    return valid_robot | valid_ghost


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
# 8. Stage 2a: External Camera – Robot Arm Alignment
# ---------------------------------------------------------------------------
def get_foreground_robot_points(T_init, K, obs_depth, pb_renderer, max_pts, device):
  """Extract robot foreground point cloud from rendered depth."""
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

  idx = np.random.choice(len(pts_robot_world), max_pts, replace=(len(pts_robot_world) < max_pts))
  return torch.tensor(pts_robot_world[idx], dtype=torch.float32, device=device)


def compute_robot_loss_batched(batch_X, T_opt, K, batch_obs):
  """Batched depth re-projection loss for robot body points."""
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
      batch_obs, grid, mode="bilinear", padding_mode="border", align_corners=True,
  ).squeeze(1).squeeze(1)

  valid_mask = (
      (Z_pred > 0.) & (Z_pred < 1.5) & (Z_obs_raw > 0.) & (Z_obs_raw < 1.5) &
      (u >= 0) & (u < w_img - 1) & (v >= 0) & (v < h_img - 1)
  )

  diff = torch.abs(Z_obs_raw[valid_mask] - Z_pred[valid_mask])
  return torch.nan_to_num(diff.mean(), nan=0.0)


def run_stage2_robot_alignment(scene_constants, pb_renderer, device,
                               init_scene_state=None, vggt_scene_state=None):
  """Stage 2a: Dual-base competition alignment for external cameras.

  Tries both VGGT and dataset-init extrinsics (if available), optimizes each
  independently, and selects the result with the lowest robot alignment loss.
  """
  OUTER_LOOPS = 5
  INNER_LOOPS = 100
  MAX_ROBOT_PTS = 2000
  MAX_ALIGN_FRAMES = 100

  print("\n🦾 Stage 2a: External camera-robot arm alignment (dual-base competition)...")

  ext_cams = [c for c in scene_constants["camera"].keys() if c != scene_constants["meta"]["wrist_serial"]]
  n_frames = len(scene_constants["robot"]["joint_positions"])

  has_init = init_scene_state is not None
  pybullet_scene_state = copy.deepcopy(init_scene_state if has_init else vggt_scene_state)

  def optimize_camera_with_base(cam_id, base_extrinsic):
    """Optimize a single camera from a given base extrinsic. Returns (T_final, loss, shift_mm)."""
    T_init_t = torch.tensor(base_extrinsic, dtype=torch.float32, device=device)
    K_t = torch.tensor(scene_constants["camera"][cam_id]["K_mat"], dtype=torch.float32, device=device)
    K_np = scene_constants["camera"][cam_id]["K_mat"]

    d_ext = torch.zeros(6, requires_grad=True, device=device)
    optimizer = optim.Adam([d_ext], lr=0.001)
    final_loss = float("inf")

    for outer_step in range(OUTER_LOOPS):
      with torch.no_grad():
        T_cur_np = (T_init_t @ make_delta_T(d_ext, device)).cpu().numpy()

      cache_X, cache_obs = [], []

      # 🌟 核心：每轮 outer_loop 都随机大洗牌，选取不重复的随机帧
      if n_frames > MAX_ALIGN_FRAMES:
        sampled_indices = np.random.choice(n_frames, MAX_ALIGN_FRAMES, replace=False)
      else:
        sampled_indices = np.arange(n_frames)

      # 🌟 遍历随机出来的帧（排个序能让读取略微连续一些）
      for t in sorted(sampled_indices):
        pb_renderer.update_robot_pose(scene_constants["robot"]["joint_positions"][t])
        d_obs = scene_constants["camera"][cam_id]["raw_depth"][t].astype(np.float32)
        r_pts_t = get_foreground_robot_points(T_cur_np, K_np, d_obs, pb_renderer, MAX_ROBOT_PTS, device)

        if r_pts_t is not None:
          cache_X.append(r_pts_t)
          cache_obs.append(torch.tensor(d_obs, dtype=torch.float32, device=device)[None, ...])

      if not cache_X:
        continue

      batch_X = torch.stack(cache_X)
      batch_obs = torch.stack(cache_obs)

      for inner_step in range(INNER_LOOPS):
        optimizer.zero_grad()
        loss_rob = compute_robot_loss_batched(batch_X, T_init_t @ make_delta_T(d_ext, device), K_t, batch_obs)
        loss_rob.backward()
        optimizer.step()
        final_loss = loss_rob.item()

        if inner_step % 50 == 0 or inner_step == INNER_LOOPS - 1:
          print(f"      Outer {outer_step+1}/{OUTER_LOOPS} | Inner {inner_step:03d} | Robot Loss: {final_loss:.4f}")

    with torch.no_grad():
      T_final_np = (T_init_t @ make_delta_T(d_ext, device)).cpu().numpy()
      shift_mm = np.linalg.norm(d_ext[3:].detach().cpu().numpy()) * 1000

    return T_final_np, final_loss, shift_mm

  for cam in ext_cams:
    print(f"\n  📷 Optimizing external camera: [{cam}] ...")

    # Build candidate pool
    base_candidates = [("VGGT", vggt_scene_state[cam]["base_extrinsic"])]
    if has_init and init_scene_state[cam]["base_extrinsic"] is not None:
      base_candidates.append(("Dataset Init", init_scene_state[cam]["base_extrinsic"]))

    best_loss = float("inf")
    best_T = None
    best_source_name = None
    best_shift = 0.0

    for source_name, base_ext in base_candidates:
      print(f"    → 🔄 Trying [{source_name}] base:")
      T_res, loss_res, shift_res = optimize_camera_with_base(cam, base_ext)

      if loss_res < best_loss:
        best_loss = loss_res
        best_T = T_res
        best_source_name = source_name
        best_shift = shift_res

    print(f"  ✅ [{cam}] Alignment done! Best base: {best_source_name}, Loss: {best_loss:.4f} (shift: {best_shift:.2f}mm)")

    pybullet_scene_state[cam]["base_extrinsic"] = best_T
    pybullet_scene_state[cam]["extrinsics"] = np.tile(best_T, (n_frames, 1, 1))

  return pybullet_scene_state


# ---------------------------------------------------------------------------
# 9. Stage 2b: Wrist Camera – Gripper Body Alignment
# ---------------------------------------------------------------------------
def get_foreground_gripper_points(T_cam_world, K, obs_depth, pb_renderer, max_pts):
  """Extract gripper-only point cloud via PyBullet segmentation mask."""
  h_img, w_img = obs_depth.shape
  cam_pos = T_cam_world[:3, 3]
  target_pos = T_cam_world[:3, 3] + T_cam_world[:3, 2]

  view_matrix = p.computeViewMatrix(cam_pos, target_pos, -T_cam_world[:3, 1])
  proj_matrix = pb_renderer._get_projection_matrix(K, w_img, h_img)

  _, _, _, depth_buffer, seg_buffer = p.getCameraImage(
      w_img, h_img, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
      renderer=p.ER_BULLET_HARDWARE_OPENGL,
      flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
  )

  metric_depth = 0.1 / (10.0 - 9.99 * np.reshape(depth_buffer, (h_img, w_img)))
  seg_array = np.reshape(seg_buffer, (h_img, w_img)).astype(np.int32)
  obj_ids = seg_array & 0xFFFFFF

  valid_ghost = obj_ids == pb_renderer.ghost_id
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


def compute_wrist_loss_batched(batch_P_ee, T_cam_ee_opt, K, batch_obs):
  """Batched depth re-projection loss for wrist gripper points anchored in EE frame."""
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
      batch_obs, grid, mode="bilinear", padding_mode="border", align_corners=True,
  ).squeeze(1).squeeze(1)

  valid_mask = (
      (Z_pred > 0.) & (Z_pred < 1.5) & (Z_obs_raw > 0.) & (Z_obs_raw < 1.5) &
      (u >= 0) & (u < w_img - 1) & (v >= 0) & (v < h_img - 1)
  )

  diff = torch.abs(Z_obs_raw[valid_mask] - Z_pred[valid_mask])
  return torch.nan_to_num(diff.mean(), nan=0.0)


def run_stage2_wrist_alignment(scene_constants, init_scene_state, pb_renderer, device):
  """Stage 2b: Optimize wrist camera hand-eye calibration via gripper body alignment."""
  OUTER_LOOPS = 5
  INNER_LOOPS = 100
  MAX_ROBOT_PTS = 2000
  MAX_ALIGN_FRAMES = 100

  print("\n🦾 Stage 2b: Wrist camera-gripper body alignment...")

  wrist_cam = scene_constants["meta"]["wrist_serial"]
  n_frames = len(scene_constants["robot"]["joint_positions"])

  pybullet_scene_state = copy.deepcopy(init_scene_state)

  T_cam_ee_init_np = init_scene_state[wrist_cam]["base_extrinsic"]
  T_cam_ee_init_t = torch.tensor(T_cam_ee_init_np, dtype=torch.float32, device=device)

  K_np = scene_constants["camera"][wrist_cam]["K_mat"]
  K_t = torch.tensor(K_np, dtype=torch.float32, device=device)
  T_ee_base_all = scene_constants["robot"]["T_ee_base_all"]

  d_ext = torch.zeros(6, requires_grad=True, device=device)
  optimizer = optim.Adam([d_ext], lr=0.001)

  for outer_step in range(OUTER_LOOPS):
    with torch.no_grad():
      T_cam_ee_np = (T_cam_ee_init_t @ make_delta_T(d_ext, device)).cpu().numpy()

    cache_P_ee, cache_obs = [], []

    # 🌟 核心：每轮 outer_loop 都随机选取不重复的随机帧
    if n_frames > MAX_ALIGN_FRAMES:
      sampled_indices = np.random.choice(n_frames, MAX_ALIGN_FRAMES, replace=False)
    else:
      sampled_indices = np.arange(n_frames)

    for t in sorted(sampled_indices):
      pb_renderer.update_robot_pose(
          scene_constants["robot"]["joint_positions"][t],
          gripper_state=scene_constants["robot"]["gripper_positions"][t],
      )

      T_cam_world_np = T_ee_base_all[t] @ T_cam_ee_np
      d_obs = scene_constants["camera"][wrist_cam]["raw_depth"][t].astype(np.float32)

      P_cam_r = get_foreground_gripper_points(T_cam_world_np, K_np, d_obs, pb_renderer, MAX_ROBOT_PTS)

      if P_cam_r is not None:
        P_ee_r = (T_cam_ee_np @ P_cam_r)[:3, :].T
        cache_P_ee.append(torch.tensor(P_ee_r, dtype=torch.float32, device=device))
        cache_obs.append(torch.tensor(d_obs, dtype=torch.float32, device=device)[None, ...])

    if not cache_P_ee:
      print(f"    ⚠️ No valid gripper points found!")
      break

    batch_P_ee = torch.stack(cache_P_ee)
    batch_obs = torch.stack(cache_obs)

    for inner_step in range(INNER_LOOPS):
      optimizer.zero_grad()

      T_cam_ee_opt = T_cam_ee_init_t @ make_delta_T(d_ext, device)
      loss_rob = compute_wrist_loss_batched(batch_P_ee, T_cam_ee_opt, K_t, batch_obs)
      loss_rob.backward()
      optimizer.step()

      if inner_step % 50 == 0 or inner_step == INNER_LOOPS - 1:
        with torch.no_grad():
          rot_deg = torch.norm(d_ext[:3]).item() * (180.0 / np.pi)
          shift_mm = torch.norm(d_ext[3:]).item() * 1000.0
        print(f"    Outer {outer_step+1}/{OUTER_LOOPS} | Inner {inner_step:03d} | Wrist Loss: {loss_rob.item():.4f} | Shift: {shift_mm:.2f}mm | Rot: {rot_deg:.2f}°")

  with torch.no_grad():
    T_cam_ee_final = (T_cam_ee_init_t @ make_delta_T(d_ext, device)).cpu().numpy()
    shift_mm = torch.norm(d_ext[3:]).item() * 1000.0
    rot_deg = torch.norm(d_ext[:3]).item() * (180.0 / np.pi)
    print(f"  ✅ [Wrist: {wrist_cam}] Hand-eye calibration converged! Shift: {shift_mm:.2f}mm, Rot: {rot_deg:.2f}°")

    pybullet_scene_state[wrist_cam]["base_extrinsic"] = T_cam_ee_final
    pybullet_scene_state[wrist_cam]["extrinsics"] = T_ee_base_all @ T_cam_ee_final

  return pybullet_scene_state


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


def get_cam_points_t(t, cam_data, device):
  """Extract full-scene point cloud from a single depth frame."""
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
  if P_cam.shape[1] > 2000:
    idx = np.random.choice(P_cam.shape[1], 2000, replace=False)
  else:
    idx = np.random.choice(P_cam.shape[1], 2000, replace=True)

  return torch.tensor(P_cam[:, idx], dtype=torch.float32, device=device)


def _run_joint_alignment(scene_constants, prev_scene_state, pb_renderer, device,
                         lr=0.001, n_steps=500, robot_weight=1.0, stage_name="Stage 3"):
  """Shared joint optimization engine for Stage 3 and Stage 4."""
  print(f"\n🌍 {stage_name}: Global joint optimization (Chamfer + Robot + Wrist, lr={lr})...")

  camera_ids = list(scene_constants["camera"].keys())
  wrist_cam = scene_constants["meta"]["wrist_serial"]
  ext_cams = [c for c in camera_ids if c != wrist_cam]
  cam1, cam2 = ext_cams[0], ext_cams[1]

  n_frames = len(scene_constants["robot"]["joint_positions"])
  robot_joints_seq = scene_constants["robot"]["joint_positions"]
  gripper_states_seq = scene_constants["robot"]["gripper_positions"]

  def to_t(arr):
    return torch.tensor(arr, dtype=torch.float32, device=device)

  T_ee_all = scene_constants["robot"]["T_ee_base_all"]

  T_cam_ee_init = prev_scene_state[wrist_cam]["base_extrinsic"]
  init_p1 = prev_scene_state[cam1]["base_extrinsic"]
  init_p2 = prev_scene_state[cam2]["base_extrinsic"]

  K_np1 = scene_constants["camera"][cam1]["K_mat"]
  K_np2 = scene_constants["camera"][cam2]["K_mat"]
  K_np_w = scene_constants["camera"][wrist_cam]["K_mat"]
  K_t1, K_t2, K_t_w = to_t(K_np1), to_t(K_np2), to_t(K_np_w)

  # Pre-compute caches
  cache_P1, cache_P2, cache_Pw, cache_Tee = [], [], [], []
  cache_X1, cache_obs1, cache_X2, cache_obs2 = [], [], [], []
  cache_P_ee, cache_obs_w = [], []

  print(f"  🔍 Pre-computing point clouds for {n_frames} frames...")
  for t in tqdm(range(n_frames), desc="Caching"):
    # Chamfer environment points
    p1 = get_cam_points_t(t, scene_constants["camera"][cam1], device)
    p2 = get_cam_points_t(t, scene_constants["camera"][cam2], device)
    pw = get_cam_points_t(t, scene_constants["camera"][wrist_cam], device)
    if p1 is not None and p2 is not None and pw is not None:
      cache_P1.append(p1)
      cache_P2.append(p2)
      cache_Pw.append(pw)
      cache_Tee.append(to_t(T_ee_all[t]))

    pb_renderer.update_robot_pose(robot_joints_seq[t], gripper_state=gripper_states_seq[t])

    # External camera robot points
    d_obs1 = scene_constants["camera"][cam1]["raw_depth"][t].astype(np.float32)
    r_pts1 = get_foreground_robot_points(init_p1, K_np1, d_obs1, pb_renderer, max_pts=2000, device=device)
    if r_pts1 is not None:
      cache_X1.append(r_pts1)
      cache_obs1.append(torch.tensor(d_obs1, dtype=torch.float32, device=device)[None, ...])

    d_obs2 = scene_constants["camera"][cam2]["raw_depth"][t].astype(np.float32)
    r_pts2 = get_foreground_robot_points(init_p2, K_np2, d_obs2, pb_renderer, max_pts=2000, device=device)
    if r_pts2 is not None:
      cache_X2.append(r_pts2)
      cache_obs2.append(torch.tensor(d_obs2, dtype=torch.float32, device=device)[None, ...])

    # Wrist gripper points (anchored to EE frame)
    T_cam_world_np = T_ee_all[t] @ T_cam_ee_init
    d_obs_w = scene_constants["camera"][wrist_cam]["raw_depth"][t].astype(np.float32)
    P_cam_r = get_foreground_gripper_points(T_cam_world_np, K_np_w, d_obs_w, pb_renderer, max_pts=2000)

    if P_cam_r is not None:
      P_ee_r = (T_cam_ee_init @ P_cam_r)[:3, :].T
      cache_P_ee.append(torch.tensor(P_ee_r, dtype=torch.float32, device=device))
      cache_obs_w.append(torch.tensor(d_obs_w, dtype=torch.float32, device=device)[None, ...])

  # Stack batches
  batch_P1 = torch.stack(cache_P1)
  batch_P2 = torch.stack(cache_P2)
  batch_Pw = torch.stack(cache_Pw)
  batch_Tee = torch.stack(cache_Tee)
  batch_X1, batch_obs1 = torch.stack(cache_X1), torch.stack(cache_obs1)
  batch_X2, batch_obs2 = torch.stack(cache_X2), torch.stack(cache_obs2)
  batch_P_ee = torch.stack(cache_P_ee) if cache_P_ee else None
  batch_obs_w = torch.stack(cache_obs_w) if cache_obs_w else None

  print(f"  ✅ Data ready! Launching GPU joint optimization engine...")

  d1 = torch.zeros(6, requires_grad=True, device=device)
  d2 = torch.zeros(6, requires_grad=True, device=device)
  dhe = torch.zeros(6, requires_grad=True, device=device)

  optimizer = optim.Adam([d1, d2, dhe], lr=lr)

  T1_init_t, T2_init_t, Tee_init_t = to_t(init_p1), to_t(init_p2), to_t(T_cam_ee_init)

  for step in range(n_steps):
    optimizer.zero_grad()

    # Chamfer
    bc1 = (T1_init_t @ make_delta_T(d1, device) @ batch_P1)[:, :3, :].transpose(1, 2)
    bc2 = (T2_init_t @ make_delta_T(d2, device) @ batch_P2)[:, :3, :].transpose(1, 2)
    T_wrist_world = batch_Tee @ (Tee_init_t @ make_delta_T(dhe, device))
    bcw = torch.bmm(T_wrist_world, batch_Pw)[:, :3, :].transpose(1, 2)

    l12, o12 = batched_chamfer_distance(bc1, bc2, device)
    l1w, o1w = batched_chamfer_distance(bc1, bcw, device)
    l2w, o2w = batched_chamfer_distance(bc2, bcw, device)
    loss_chamfer = l12 + l1w + l2w

    # Robot
    l_rob1 = compute_robot_loss_batched(batch_X1, T1_init_t @ make_delta_T(d1, device), K_t1, batch_obs1)
    l_rob2 = compute_robot_loss_batched(batch_X2, T2_init_t @ make_delta_T(d2, device), K_t2, batch_obs2)

    # Wrist
    l_wrist = torch.tensor(0.0, device=device)
    if batch_P_ee is not None:
      l_wrist = compute_wrist_loss_batched(batch_P_ee, Tee_init_t @ make_delta_T(dhe, device), K_t_w, batch_obs_w)

    loss_total = loss_chamfer + robot_weight * (l_rob1 + l_rob2 + l_wrist)
    loss_total.backward()
    optimizer.step()

    if step % 50 == 0 or step == n_steps - 1:
      bg_overlap = (o12 + o1w + o2w) / 3.0 * 100
      shift_c1 = torch.norm(d1[:3]).item() * 1000
      shift_c2 = torch.norm(d2[:3]).item() * 1000
      shift_w = torch.norm(dhe[:3]).item() * 1000
      print(f"    Step {step:03d} | "
            f"Chmf: {loss_chamfer.item():.4f} | "
            f"Rob1: {l_rob1.item():.4f} | Rob2: {l_rob2.item():.4f} | Wrst: {l_wrist.item():.4f} | "
            f"BG Overlap: {bg_overlap:.1f}% | "
            f"Shift → C1: {shift_c1:.2f}mm, C2: {shift_c2:.2f}mm, W: {shift_w:.2f}mm")

  with torch.no_grad():
    final_p1 = (T1_init_t @ make_delta_T(d1, device)).cpu().numpy()
    final_p2 = (T2_init_t @ make_delta_T(d2, device)).cpu().numpy()
    final_cam_ee = (Tee_init_t @ make_delta_T(dhe, device)).cpu().numpy()

  print(f"\n✅ {stage_name} complete! Joint shifts:")
  print(f"  📷 [Static {cam1}]: {np.linalg.norm(d1[:3].detach().cpu().numpy()) * 1000:.2f} mm")
  print(f"  📷 [Static {cam2}]: {np.linalg.norm(d2[:3].detach().cpu().numpy()) * 1000:.2f} mm")
  print(f"  🦾 [Kinematic Wrist]: {np.linalg.norm(dhe[:3].detach().cpu().numpy()) * 1000:.2f} mm")

  ultimate_scene_state = {cam: {} for cam in camera_ids}
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


def run_stage3_joint_alignment(scene_constants, stage2_scene_state, pb_renderer, device):
  """Stage 3: Joint optimization with lr=0.001 and robot_weight=1.0."""
  return _run_joint_alignment(
      scene_constants, stage2_scene_state, pb_renderer, device,
      lr=0.001, n_steps=500, robot_weight=1.0, stage_name="Stage 3",
  )


def run_stage4_joint_alignment(scene_constants, stage3_scene_state, pb_renderer, device):
  """Stage 4: Fine-tuning with lr=0.0001 and robot_weight=0.1."""
  return _run_joint_alignment(
      scene_constants, stage3_scene_state, pb_renderer, device,
      lr=0.0001, n_steps=500, robot_weight=0.1, stage_name="Stage 4",
  )


# ---------------------------------------------------------------------------
# 11. Export
# ---------------------------------------------------------------------------
def export_extrinsics(scene_constants, scene_state,
                      export_root="~/droid_data/output/mv-tap/droid/stage2"):
  """Save calibrated extrinsics for all cameras.

  Output layout:
    <episode_id>/extrinsics.npz
      Keys per camera:
        <cam_id>_base_extrinsic  → (4, 4) static extrinsic matrix
        <cam_id>_extrinsics      → (N, 4, 4) per-frame trajectory
  """
  ep_str = scene_constants["meta"]["episode_id"]
  out_dir = os.path.abspath(os.path.expanduser(os.path.join(export_root, ep_str)))
  os.makedirs(out_dir, exist_ok=True)

  save_dict = {}
  for cam_id, state in scene_state.items():
    save_dict[f"{cam_id}_base_extrinsic"] = state["base_extrinsic"].astype(np.float32)
    save_dict[f"{cam_id}_extrinsics"] = state["extrinsics"].astype(np.float32)

  # Also save wrist serial for downstream convenience
  save_dict["wrist_serial"] = np.array(scene_constants["meta"]["wrist_serial"])

  np.savez_compressed(os.path.join(out_dir, "extrinsics.npz"), **save_dict)
  print(f"  💾 Extrinsics saved to {out_dir}/extrinsics.npz")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="DROID Stage 2: Camera Extrinsics Calibration")
  parser.add_argument("--rank", type=int, default=0, help="Rank of the process")
  parser.add_argument("--world_size", type=int, default=1, help="Total number of processes")
  parser.add_argument("--ep_list", type=str, help="Comma-separated list of episode IDs")
  parser.add_argument("--stage1_root", type=str, default="~/droid_data/output/mv-tap/droid/stage1",
                       help="Root directory of Stage 1 outputs")
  args = parser.parse_args()

  print("🚀 DROID Stage 2: Camera Extrinsics Calibration Pipeline")
  device = get_accelerator()
  vggt_model, load_fn, pose_fn = init_stage2_models()
  serials_db, extrinsics_db = load_metadata()

  # Discover available episodes from stage1 output
  stage1_abs = os.path.abspath(os.path.expanduser(args.stage1_root))
  if args.ep_list:
    target_eps = [ep.strip() for ep in args.ep_list.split(",") if ep.strip()]
    print(f"📋 Selected via --ep_list: {target_eps}")
  else:
    available_eps = sorted([
        d for d in os.listdir(stage1_abs)
        if os.path.isdir(os.path.join(stage1_abs, d))
    ])
    import random
    random.seed(42)
    random.shuffle(available_eps)
    target_eps = available_eps[args.rank::args.world_size]
    print(f"📋 Selected via distributed rank {args.rank}/{args.world_size} targeting: {len(target_eps)} episodes")

  succeeded_eps = []

  for idx, ep_id in enumerate(target_eps):
    print(f"\n🎬 [{idx + 1}/{len(target_eps)}] Processing Episode: {ep_id}")

    try:
      # Load Stage 1 outputs
      scene_constants = load_stage1_data(ep_id, args.stage1_root)

      # Stage 0: Initialize from dataset extrinsics (if available)
      init_scene_state = init_camera_states(scene_constants, extrinsics_db)

      # Check if all cameras have pre-calibrated extrinsics
      all_extrinsics_exist = all(
          state["extrinsics"] is not None
          for state in init_scene_state.values()
      )

      # Stage 1: VGGT visual anchoring (always run for dual-base competition)
      vggt_scene_state = vggt_warmup_extrinsics(
          scene_constants, vggt_model, load_fn, pose_fn, device,
      )

      # Stage 2a: External camera-robot alignment (dual-base competition)
      pb_renderer = PyBulletRenderer_Robotiq()
      robot_aligned_state = run_stage2_robot_alignment(
          scene_constants, pb_renderer, device,
          init_scene_state=init_scene_state if all_extrinsics_exist else None,
          vggt_scene_state=vggt_scene_state,
      )

      # Stage 2b: Wrist camera-gripper alignment
      wrist_aligned_state = run_stage2_wrist_alignment(
          scene_constants, robot_aligned_state, pb_renderer, device,
      )

      # Stage 3: Joint optimization (coarse)
      joint_state = run_stage3_joint_alignment(
          scene_constants, wrist_aligned_state, pb_renderer, device,
      )

      # Stage 4: Fine-tuning (refined)
      final_state = run_stage4_joint_alignment(
          scene_constants, joint_state, pb_renderer, device,
      )

      # Export
      export_extrinsics(scene_constants, final_state)
      succeeded_eps.append(ep_id)
      print(f"  ✅ Episode {ep_id} completed successfully.")

    except Exception as e:
      print(f"  ❌ Episode {ep_id} failed: {e}")
      import traceback
      traceback.print_exc()
      continue

  # Multi-process safe append
  stage2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "episodes_stage2.txt")
  if succeeded_eps:
    batch = "".join(ep_id + "\n" for ep_id in succeeded_eps)
    with open(stage2_path, "a") as f:
      fcntl.flock(f, fcntl.LOCK_EX)
      f.write(batch)
      fcntl.flock(f, fcntl.LOCK_UN)
    print(f"\n📝 Appended {len(succeeded_eps)} episodes to {stage2_path}")

  print(f"\n🎉 Stage 2 complete! {len(succeeded_eps)}/{len(target_eps)} episodes succeeded.")
