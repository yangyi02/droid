"""DROID Episode Extraction and Vision Foundation Model Pipeline.

A minimalist, high-efficiency multi-stage processor for decoding raw ZED SVO
stereo video streams, extracting robot kinematics, and inferring metric depth.
"""

import argparse
import fcntl
import glob
import json
import os
import sys
import warnings

import cv2
import h5py
import numpy as np
import pybullet as p
import pyzed.sl as sl
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F
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
# 1. Foundation Models
# ---------------------------------------------------------------------------
def init_all_models():
  """Load vision foundation models and vendor dependencies dynamically."""
  device = get_accelerator()
  print(f"🚀 Launching models onto {device} | CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
  if not torch.cuda.is_available():
    print("⚠️ WARNING: PyTorch cannot find a valid CUDA device. Please ensure your CUDA_VISIBLE_DEVICES index is correct (e.g. 0 or 1) and NVIDIA drivers are running.")

  # Inject third-party repo paths just-in-time
  vendor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party")
  for pkg in ["s2m2/src", "vggt", "co-tracker"]:
    path = os.path.join(vendor_dir, pkg)
    if path not in sys.path:
      sys.path.append(path)

  # Lazy importing
  from cotracker.predictor import CoTrackerPredictor
  from s2m2.core.utils.model_utils import load_model
  from segment_anything import sam_model_registry, SamPredictor
  from vggt.models.vggt import VGGT

  s2m2_model = torch.compile(
      load_model(
          os.path.join(vendor_dir, "s2m2/weights/pretrain_weights"),
          "XL",
          True,
          3,
          device,
      ).eval()
  )
  cotracker_model = CoTrackerPredictor(
      checkpoint=os.path.join(
          vendor_dir, "co-tracker/weights/cotracker3_offline.pth"
      )
  ).to(device)
  vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
  sam = sam_model_registry["vit_h"](
      checkpoint=os.path.join(vendor_dir, "sam_weights/sam_vit_h_4b8939.pth")
  ).to(device)

  print("  ✅ All foundation models securely loaded.")
  return s2m2_model, cotracker_model, vggt_model, SamPredictor(sam)


# ---------------------------------------------------------------------------
# 2. Metadata Management
# ---------------------------------------------------------------------------
def load_metadata():
  """Securely fetch and load global dataset JSON mappings."""
  root_path = os.path.expanduser("~/droid_data/meta/1.0.1")
  os.makedirs(root_path, exist_ok=True)

  base_url = "https://huggingface.co/KarlP/droid/resolve/main"
  files = [
      "intrinsics.json",
      "camera_serials.json",
      "episode_id_to_path.json",
      "keep_ranges_1_0_1.json",
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
  id_to_path = load_json("episode_id_to_path.json")
  keep_ranges = load_json("keep_ranges_1_0_1.json")
  extrinsics_db = load_json("cam2base_extrinsic_superset.json")

  valid_ids = sorted(set(serials_db.keys()) & set(id_to_path.keys()) & set(extrinsics_db.keys()))
  print(f"✅ Metadata ready! Matched {len(valid_ids)} episodes with pre-calibrated extrinsics.")
  return serials_db, id_to_path, keep_ranges, extrinsics_db, valid_ids


# ---------------------------------------------------------------------------
# 3. SVO Decoding & Kinematics Extraction
# ---------------------------------------------------------------------------
def download_episode(episode_id, root_path, id_to_path, serials_db, keep_ranges_db):
  """Download episode data and initialize hierarchical scene_constants."""
  relative_path = id_to_path[episode_id]
  episode_path = os.path.join(root_path, relative_path)

  cam_info = serials_db[episode_id]
  wrist_serial = cam_info.get("wrist_cam_serial")

  # Extract all camera serials directly from the serial data
  valid_cams = sorted(set(cam_info.values()))

  # Reconstruct the official JSON absolute bucket path format
  base_prefix = "gs://xembodiment_data/r2d2/r2d2-data-full/"
  episode_key = f"{base_prefix}{relative_path}/recordings/MP4--{base_prefix}{relative_path}/trajectory.h5"

  valid_indices = None

  if episode_key in keep_ranges_db:
    ranges = keep_ranges_db[episode_key]
    indices = []
    for start, end in ranges:
      indices.extend(range(start, end))
    valid_indices = np.array(indices)
    print(f"  ✂️ Loaded action ranges, marked {len(valid_indices)} frames as valid keyframes.")
  else:
    print(f"  ⚠️ No idle filter info found for {episode_id}, keeping all frames.")

  # Structured into meta, robot, camera core modules
  return {
      "meta": {
          "episode_id": episode_id,
          "episode_path": episode_path,
          "wrist_serial": wrist_serial,
          "valid_indices": valid_indices,
      },
      "robot": {},  # Placeholder for kinematics parsing
      "camera": {
          cam: {
              "baseline": 0.063 if cam == wrist_serial else 0.120,
          }
          for cam in valid_cams
      },
  }


def extract_svo_video(scene_constants):
  """Decode SVO video, extract full stereo calibration and both rectified/unrectified frames."""
  print("  🎥 Decoding SVO video streams and physical calibration data...")
  episode_path = scene_constants["meta"]["episode_path"]

  for cam in scene_constants["camera"]:
    svo_files = glob.glob(os.path.join(episode_path, f"**/{cam}.svo"), recursive=True)
    if not svo_files:
      continue

    zed, init_params = sl.Camera(), sl.InitParameters()
    init_params.set_from_svo_file(svo_files[0])
    init_params.svo_real_time_mode = False
    zed.open(init_params)

    # Extract ZED native calibration data
    cam_info = zed.get_camera_information()

    # --- Rectified (Calibrated) ---
    calib = cam_info.camera_configuration.calibration_parameters
    K_calib_left = np.array([
        [calib.left_cam.fx, 0, calib.left_cam.cx],
        [0, calib.left_cam.fy, calib.left_cam.cy],
        [0, 0, 1],
    ], dtype=np.float32)
    disto_calib_left = np.array(calib.left_cam.disto, dtype=np.float32)

    K_calib_right = np.array([
        [calib.right_cam.fx, 0, calib.right_cam.cx],
        [0, calib.right_cam.fy, calib.right_cam.cy],
        [0, 0, 1],
    ], dtype=np.float32)
    disto_calib_right = np.array(calib.right_cam.disto, dtype=np.float32)

    # --- Raw (Distorted) ---
    calib_raw = cam_info.camera_configuration.calibration_parameters_raw
    K_raw_left = np.array([
        [calib_raw.left_cam.fx, 0, calib_raw.left_cam.cx],
        [0, calib_raw.left_cam.fy, calib_raw.left_cam.cy],
        [0, 0, 1],
    ], dtype=np.float32)
    disto_raw_left = np.array(calib_raw.left_cam.disto, dtype=np.float32)

    K_raw_right = np.array([
        [calib_raw.right_cam.fx, 0, calib_raw.right_cam.cx],
        [0, calib_raw.right_cam.fy, calib_raw.right_cam.cy],
        [0, 0, 1],
    ], dtype=np.float32)
    disto_raw_right = np.array(calib_raw.right_cam.disto, dtype=np.float32)

    # ==========================================
    # Frame extraction loop (left/right + raw)
    # ==========================================
    all_left, all_right, all_left_raw, all_right_raw = [], [], [], []
    left_mat, right_mat = sl.Mat(), sl.Mat()
    left_raw_mat, right_raw_mat = sl.Mat(), sl.Mat()

    for _ in tqdm(range(zed.get_svo_number_of_frames()), desc=f"Decoding {cam}"):
      zed.grab()
      # 1. Extract rectified stereo frames
      zed.retrieve_image(left_mat, sl.VIEW.LEFT)
      zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
      # 2. Extract raw unrectified stereo frames
      zed.retrieve_image(left_raw_mat, sl.VIEW.LEFT_UNRECTIFIED)
      zed.retrieve_image(right_raw_mat, sl.VIEW.RIGHT_UNRECTIFIED)

      all_left.append(cv2.cvtColor(left_mat.get_data(), cv2.COLOR_BGRA2RGB))
      all_right.append(cv2.cvtColor(right_mat.get_data(), cv2.COLOR_BGRA2RGB))
      all_left_raw.append(cv2.cvtColor(left_raw_mat.get_data(), cv2.COLOR_BGRA2RGB))
      all_right_raw.append(cv2.cvtColor(right_raw_mat.get_data(), cv2.COLOR_BGRA2RGB))

    zed.close()

    # Pack results into scene_constants
    scene_constants["camera"][cam].update({
        "K_mat": K_calib_left,
        "zed_calibration": {
            "calibrated": {
                "K": K_calib_left, "disto": disto_calib_left,
                "K_right": K_calib_right, "disto_right": disto_calib_right,
            },
            "raw": {
                "K": K_raw_left, "disto": disto_raw_left,
                "K_right": K_raw_right, "disto_right": disto_raw_right,
            },
        },
        "video_rgb": np.stack(all_left),
        "video_right": np.stack(all_right),
        "video_raw_rgb": np.stack(all_left_raw),
        "video_raw_right": np.stack(all_right_raw),
    })

  return scene_constants


def make_4x4(vec_6d):
  """Convert 6DoF vector [x, y, z, rx, ry, rz] to 4x4 homogeneous transform."""
  transform = np.eye(4)
  transform[:3, :3] = R.from_euler("xyz", vec_6d[3:]).as_matrix()
  transform[:3, 3] = vec_6d[:3]
  return transform


def parse_robot_kinematics(scene_constants):
  """Load H5 and JSON to extract robot kinematics and hand-eye matrices."""
  print("  🦾 Parsing robot H5 kinematics and hand-eye matrices...")
  ep_path = scene_constants["meta"]["episode_path"]

  h5_files = glob.glob(f"{ep_path}/trajectory.h5")
  json_files = glob.glob(f"{ep_path}/metadata_*.json")
  if not h5_files or not json_files:
    return scene_constants

  with h5py.File(h5_files[0], "r") as f, open(json_files[0]) as jf:
    # Extract full arrays from H5 using [:]
    ee_poses = f["observation/robot_state/cartesian_position"][:]
    joint_poses = f["observation/robot_state/joint_positions"][:]
    gripper_poses = f["observation/robot_state/gripper_position"][:]
    wrist_ext = json.load(jf)["wrist_cam_extrinsics"]
    wrist_ext = wrist_ext.get("extrinsics", wrist_ext) if isinstance(wrist_ext, dict) else wrist_ext

  # Get total frame count
  total_frames = len(ee_poses)

  # Vectorized batch construction of end-effector poses
  T_ee_all = np.tile(np.eye(4), (total_frames, 1, 1))
  T_ee_all[:, :3, :3] = R.from_euler("xyz", ee_poses[:, 3:]).as_matrix()
  T_ee_all[:, :3, 3] = ee_poses[:, :3]

  # Pack into robot dict
  scene_constants["robot"] = {
      "joint_positions": joint_poses,
      "gripper_positions": gripper_poses,
      "T_cam_ee_init": np.linalg.inv(make_4x4(ee_poses[0])) @ make_4x4(wrist_ext),
      "T_ee_base_all": T_ee_all,
  }
  return scene_constants


def align_temporal_streams(scene_constants):
  """Truncate all temporal streams to the shortest length for global alignment."""
  print("  ⏱️ Running global temporal alignment check...")
  if "T_ee_base_all" not in scene_constants["robot"]:
    return scene_constants

  # 1. Collect lengths of all temporal streams
  lengths = [
      len(scene_constants["robot"]["joint_positions"]),
      len(scene_constants["robot"]["gripper_positions"]),
      len(scene_constants["robot"]["T_ee_base_all"]),
  ]
  for cam_id, cam_data in scene_constants["camera"].items():
    lengths.append(len(cam_data["video_rgb"]))
    lengths.append(len(cam_data["video_right"]))

  # 2. Find the shortest stream (bottleneck)
  min_frames = min(lengths)
  max_frames = max(lengths)

  if min_frames == max_frames:
    print(f"    ✅ Temporal streams perfectly aligned at {min_frames} frames.")
    return scene_constants

  print(f"    ⚠️ Temporal mismatch detected (max {max_frames}, min {min_frames})!")
  print(f"    ✂️ Truncating all streams to {min_frames} frames...")

  # 3. Truncate all dimensions to min_frames
  scene_constants["robot"]["joint_positions"] = scene_constants["robot"]["joint_positions"][:min_frames]
  scene_constants["robot"]["gripper_positions"] = scene_constants["robot"]["gripper_positions"][:min_frames]
  scene_constants["robot"]["T_ee_base_all"] = scene_constants["robot"]["T_ee_base_all"][:min_frames]

  for cam_id, cam_data in scene_constants["camera"].items():
    cam_data["video_rgb"] = cam_data["video_rgb"][:min_frames]
    cam_data["video_right"] = cam_data["video_right"][:min_frames]

  print("    ✅ Global alignment complete. All dimensions are now consistent.")
  return scene_constants


# ---------------------------------------------------------------------------
# 4. Stereo Vision Inference
# ---------------------------------------------------------------------------
def decode_disparity_np(disp, fx, baseline):
  """Convert raw disparity to metric depth (NumPy)."""
  z = np.zeros_like(disp)
  valid_mask = disp > 0  # Filter invalid disparity
  z[valid_mask] = (fx * baseline) / disp[valid_mask]
  return z


@torch.inference_mode()
def get_s2m2_disparity(model, img_left, img_right, device, conf_thresh=0.95):
  """Pure disparity extractor: inference only, no premature depth conversion."""
  from s2m2.core.utils.model_utils import run_stereo_matching
  left_torch = torch.from_numpy(img_left).permute(2, 0, 1).unsqueeze(0).to(device)
  right_torch = torch.from_numpy(img_right).permute(2, 0, 1).unsqueeze(0).to(device)

  pred_disp, _, pred_conf, _, _ = run_stereo_matching(
      model, left_torch, right_torch, device, N_repeat=3
  )

  disp = pred_disp.cpu().numpy().squeeze()
  conf = pred_conf.cpu().numpy().squeeze()

  # Mask out low-confidence disparity pixels
  valid_mask = (disp > 0) & (conf >= conf_thresh)
  disp[~valid_mask] = 0.0

  return disp


def compute_stereo_depth(scene_constants, model, device):
  """Run S2M2 stereo inference and convert raw disparity to metric depth."""
  print("  🧠 Running S2M2 stereo depth inference...")
  for cam_id in scene_constants["camera"]:
    cam_data = scene_constants["camera"][cam_id]
    left_seq, right_seq = cam_data["video_rgb"], cam_data["video_right"]
    if "video_rgb" not in cam_data or "video_right" not in cam_data:
      continue

    disp_frames = [
        get_s2m2_disparity(model, left_img, right_img, device)
        for left_img, right_img in tqdm(
            zip(left_seq, right_seq),
            total=len(left_seq),
            desc=f"Depth [{cam_id}]",
        )
    ]
    raw_disp = np.stack(disp_frames)

    fx = cam_data["K_mat"][0, 0]
    baseline = cam_data["baseline"]
    cam_data["raw_depth"] = decode_disparity_np(raw_disp, fx, baseline)

  return scene_constants


# ---------------------------------------------------------------------------
# 5. Gripper Depth Refinement (SAM + Temporal Distillation)
# ---------------------------------------------------------------------------
def extract_single_frame_mask(img_rgb, predictor):
  """Extract a single-frame gripper mask using SAM with positive/negative prompts."""
  h, w = img_rgb.shape[:2]

  points = np.array([
      [w // 2 - 120, h - 110],
      [w // 2 + 500, h - 110],
      [w // 2 - 250, h - 25],
      [w // 2 + 450, h - 25],
      [w // 2 + 100, h - 15],
      [w // 2 + 100, h - 300],
  ])
  labels = np.array([1, 1, 1, 1, 1, 0])
  bbox = np.array([0, h // 2, w, h])

  predictor.set_image(img_rgb)
  masks, scores, _ = predictor.predict(
      point_coords=points, point_labels=labels, box=bbox, multimask_output=True
  )

  valid_masks = []
  for m, s in zip(masks, scores):
    area_ratio = np.sum(m) / (w * h)
    if 0.02 < area_ratio < 0.45:
      valid_masks.append((m, s * area_ratio))

  if valid_masks:
    best_mask = max(valid_masks, key=lambda x: x[1])[0]
  else:
    best_mask = masks[np.argmax(scores)]

  return best_mask


def compute_consensus_mask(masks_list, consensus_thresh=0.5):
  """Compute a consensus mask from multiple per-frame masks via voting."""
  vote_map = np.mean(masks_list, axis=0)
  consensus_mask = vote_map >= consensus_thresh

  num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
      consensus_mask.astype(np.uint8)
  )
  if num_labels > 1:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    consensus_mask = labels_map == largest_label

  return consensus_mask


def build_universal_gripper_mask(scene_constants, sam_predictor):
  """Build a universal static gripper mask from closed-gripper frames using SAM."""
  wrist_cam = scene_constants["meta"].get("wrist_serial")
  if wrist_cam is None or wrist_cam not in scene_constants["camera"]:
    print("  ⚠️ No wrist camera found, skipping gripper mask extraction.")
    return scene_constants

  cam_data = scene_constants["camera"][wrist_cam]
  gripper_states = scene_constants["robot"].get("gripper_positions")
  if gripper_states is None:
    print("  ⚠️ No gripper positions found, skipping gripper mask extraction.")
    return scene_constants

  closed_indices = np.where(gripper_states < 0.05)[0]
  if len(closed_indices) == 0:
    print("  ⚠️ No closed-gripper frames found, skipping mask extraction.")
    return scene_constants

  print(f"  🎭 Building consensus gripper mask from {len(closed_indices)} closed-gripper frames...")

  masks_list = []
  for idx in tqdm(closed_indices, desc="SAM mask"):
    img = cam_data["video_rgb"][idx].copy()
    mask = extract_single_frame_mask(img, sam_predictor)
    masks_list.append(mask)

  final_mask = compute_consensus_mask(masks_list)

  # Broadcast the static consensus mask to all closed-gripper frames
  n_frames = len(gripper_states)
  cam_data["sam_real_masks"] = np.zeros(
      (n_frames, *final_mask.shape), dtype=bool
  )
  cam_data["sam_real_masks"][closed_indices] = final_mask
  print(f"  ✅ Gripper consensus mask built and broadcast to {len(closed_indices)} frames.")

  return scene_constants


def distill_empirical_gripper_depth(scene_constants, max_depth_thresh=0.15):
  """Distill a clean gripper surface depth via temporal median of masked stereo depth."""
  wrist_cam = scene_constants["meta"].get("wrist_serial")
  if wrist_cam is None or wrist_cam not in scene_constants["camera"]:
    return scene_constants

  cam_data = scene_constants["camera"][wrist_cam]
  gripper_states = scene_constants["robot"].get("gripper_positions")
  if gripper_states is None:
    return scene_constants

  closed_indices = np.where(gripper_states < 0.05)[0]
  if len(closed_indices) == 0:
    print("  ⚠️ No closed-gripper frames for depth distillation.")
    return scene_constants

  if "sam_real_masks" not in cam_data:
    print("  ⚠️ No SAM masks found, skipping depth distillation.")
    return scene_constants

  h, w = cam_data["video_rgb"][0].shape[:2]
  num_frames = len(closed_indices)

  print(f"  🧪 Distilling gripper depth from {num_frames} closed-gripper frames...")
  depth_bank = np.full((num_frames, h, w), np.nan, dtype=np.float32)

  for i, idx in enumerate(tqdm(closed_indices, desc="Depth collect")):
    raw_depth = cam_data["raw_depth"][idx].astype(np.float32)
    mask = cam_data["sam_real_masks"][idx]
    valid_pixels = (mask > 0) & (raw_depth > 0) & (raw_depth < max_depth_thresh)
    depth_bank[i, valid_pixels] = raw_depth[valid_pixels]

  print("  🔨 Computing temporal median depth...")
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    median_depth = np.nanmedian(depth_bank, axis=0)

  median_depth = np.nan_to_num(median_depth, nan=0.0).astype(np.float32)
  cam_data["empirical_gripper_depth"] = median_depth
  print("  ✅ Gripper depth distillation complete.")

  return scene_constants


def inject_gripper_depth(scene_constants):
  """Inject distilled gripper depth into raw_depth for closed-gripper frames."""
  wrist_cam = scene_constants["meta"].get("wrist_serial")
  if wrist_cam is None or wrist_cam not in scene_constants["camera"]:
    return scene_constants

  cam_data = scene_constants["camera"][wrist_cam]
  gripper_states = scene_constants["robot"].get("gripper_positions")
  empirical_depth = cam_data.get("empirical_gripper_depth")

  if gripper_states is None or empirical_depth is None:
    return scene_constants

  closed_indices = np.where(gripper_states < 0.05)[0]
  if len(closed_indices) == 0:
    return scene_constants

  valid_mask = empirical_depth > 0

  print(f"  💉 Injecting distilled gripper depth into {len(closed_indices)} frames...")
  cam_data["raw_depth"][closed_indices] = np.where(
      valid_mask,
      empirical_depth,
      cam_data["raw_depth"][closed_indices],
  )

  replaced_pixels_per_frame = int(np.sum(valid_mask))
  total_replaced = len(closed_indices) * replaced_pixels_per_frame
  print(f"  ✅ Injection complete! {total_replaced} noisy depth pixels replaced.")

  return scene_constants


def _write_mp4(path, frames, fps=10.0):
  """Write a sequence of RGB frames to an mp4 file."""
  h, w = frames[0].shape[:2]
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
  for img in frames:
    writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
  writer.release()


def export_to_disk(scene_constants, export_root="~/droid_data/output/mv-tap/droid/stage1"):
  """Export all videos, depth maps, calibration, and robot kinematics to disk.

  Episode directory layout:
    robot.npz                         - Robot kinematics & metadata
    <cam_serial>/
      video_left.mp4                  - Rectified left eye
      video_right.mp4                 - Rectified right eye
      video_left_raw.mp4              - Unrectified (distorted) left eye
      video_right_raw.mp4             - Unrectified (distorted) right eye
      raw_depth.npz                   - uint16 best depth (refined for wrist, original for others)
      original_raw_depth.npz          - uint16 pre-injection stereo depth (wrist only)
      gripper_mask.npz                - bool consensus SAM mask (wrist only)
      gripper_depth.npz               - uint16 distilled gripper surface depth (wrist only)
      calibration.npz                 - Full intrinsics: calibrated & raw K + distortion
  """
  ep_str = scene_constants["meta"]["episode_id"]
  out_dir = os.path.abspath(os.path.expanduser(os.path.join(export_root, ep_str)))
  os.makedirs(out_dir, exist_ok=True)

  print(f"  💾 Exporting multi-view data to {out_dir}...")

  # --- Robot kinematics (episode-level) ---
  robot = scene_constants["robot"]
  if robot:
    robot_save = {}
    if "joint_positions" in robot:
      robot_save["joint_positions"] = robot["joint_positions"].astype(np.float32)
    if "gripper_positions" in robot:
      robot_save["gripper_positions"] = robot["gripper_positions"].astype(np.float32)
    if "T_ee_base_all" in robot:
      robot_save["T_ee_base_all"] = robot["T_ee_base_all"].astype(np.float32)
    if "T_cam_ee_init" in robot:
      robot_save["T_cam_ee_init"] = robot["T_cam_ee_init"].astype(np.float32)
    # Include metadata
    meta = scene_constants["meta"]
    if meta.get("valid_indices") is not None:
      robot_save["valid_indices"] = meta["valid_indices"]
    if meta.get("wrist_serial") is not None:
      robot_save["wrist_serial"] = np.array(meta["wrist_serial"])
    np.savez_compressed(os.path.join(out_dir, "robot.npz"), **robot_save)

  # --- Per-camera data ---
  for cam_id, data in scene_constants["camera"].items():
    cam_dir = os.path.join(out_dir, str(cam_id))
    os.makedirs(cam_dir, exist_ok=True)

    # Videos: save all 4 streams (rectified + raw, left + right)
    video_keys = {
        "video_rgb": "video_left.mp4",
        "video_right": "video_right.mp4",
        "video_raw_rgb": "video_left_raw.mp4",
        "video_raw_right": "video_right_raw.mp4",
    }
    for key, filename in video_keys.items():
      if key in data and len(data[key]) > 0:
        _write_mp4(os.path.join(cam_dir, filename), data[key])

    # Depth: uint16 in millimeters
    if "original_raw_depth" in data:
      # Wrist camera: save pre-injection backup as original_raw_depth.npz
      np.savez_compressed(
          os.path.join(cam_dir, "original_raw_depth.npz"),
          depth=(data["original_raw_depth"] * 1000).astype(np.uint16),
      )
    if "raw_depth" in data:
      # raw_depth.npz = best available depth (refined for wrist, original for others)
      np.savez_compressed(
          os.path.join(cam_dir, "raw_depth.npz"),
          depth=(data["raw_depth"] * 1000).astype(np.uint16),
      )

    # Gripper intermediate artifacts (wrist camera only)
    if "sam_real_masks" in data:
      np.savez_compressed(
          os.path.join(cam_dir, "gripper_mask.npz"),
          mask=data["sam_real_masks"],
      )
    if "empirical_gripper_depth" in data:
      gripper_uint16 = (data["empirical_gripper_depth"] * 1000).astype(
          np.uint16
      )
      np.savez_compressed(
          os.path.join(cam_dir, "gripper_depth.npz"),
          depth=gripper_uint16,
      )

    # Full calibration: all K matrices + distortion + baseline
    if "zed_calibration" in data:
      calib = data["zed_calibration"]
      np.savez(
          os.path.join(cam_dir, "calibration.npz"),
          # Rectified (calibrated)
          K_calib_left=calib["calibrated"]["K"],
          K_calib_right=calib["calibrated"]["K_right"],
          disto_calib_left=calib["calibrated"]["disto"],
          disto_calib_right=calib["calibrated"]["disto_right"],
          # Raw (distorted)
          K_raw_left=calib["raw"]["K"],
          K_raw_right=calib["raw"]["K_right"],
          disto_raw_left=calib["raw"]["disto"],
          disto_raw_right=calib["raw"]["disto_right"],
          # Stereo baseline
          baseline=np.array(data["baseline"], dtype=np.float32),
      )

  return True


# ---------------------------------------------------------------------------
# Execution & Batched Slicing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="DROID Flexible Pipeline Extractor")
  parser.add_argument("--rank", type=int, default=0, help="Rank of the process")
  parser.add_argument("--world_size", type=int, default=1, help="Total number of processes")
  parser.add_argument("--limit", type=int, default=-1, help="Limit total number of episodes to process")
  parser.add_argument("--ep_list", type=str, help="Optional comma-separated list of specific episode IDs")
  args = parser.parse_args()

  print("Environment setup verified. Initializing flexible multi-GPU extractor...")
  device = get_accelerator()
  s2m2_model, cotracker_model, vggt_model, sam_predictor = init_all_models()
  serials_db, id_to_path, keep_ranges, extrinsics_db, valid_ids = load_metadata()

  if args.ep_list:
    target_eps = [ep.strip() for ep in args.ep_list.split(",") if ep.strip()]
    print(f"📋 Selected via --ep_list targeting: {target_eps}")
  else:
    import random
    random.seed(42)
    random.shuffle(valid_ids)
    if args.limit > 0:
      valid_ids = valid_ids[:args.limit]
    target_eps = valid_ids[args.rank::args.world_size]
    print(f"📋 Selected via distributed rank {args.rank}/{args.world_size} targeting: {len(target_eps)} episodes")

  succeeded_eps = []

  for idx, ep_id in enumerate(target_eps):
    print(f"\n🎬 [{idx + 1}/{len(target_eps)}] Processing Episode: {ep_id}")
    if ep_id not in id_to_path:
      print(f"  ❌ Invalid episode ID: {ep_id}")
      continue

    try:
      scene_constants = download_episode(
          ep_id,
          os.path.expanduser("~/droid_data/input/robotics/droid_raw/1.0.1"),
          id_to_path,
          serials_db,
          keep_ranges,
      )
      scene_constants = extract_svo_video(scene_constants)
      if not any("video_rgb" in data for data in scene_constants["camera"].values()):
        print(f"  ⚠️ No valid video streams extracted for [{ep_id}]. Skipping processing.")
        continue
      scene_constants = parse_robot_kinematics(scene_constants)
      scene_constants = align_temporal_streams(scene_constants)
      scene_constants = compute_stereo_depth(scene_constants, s2m2_model, device)

      # --- Gripper depth refinement (wrist camera only) ---
      # Save original raw depth before injection
      wrist_serial = scene_constants["meta"].get("wrist_serial")
      if wrist_serial and wrist_serial in scene_constants["camera"]:
        wrist_data = scene_constants["camera"][wrist_serial]
        if "raw_depth" in wrist_data:
          wrist_data["original_raw_depth"] = wrist_data["raw_depth"].copy()

      scene_constants = build_universal_gripper_mask(scene_constants, sam_predictor)
      scene_constants = distill_empirical_gripper_depth(scene_constants)
      scene_constants = inject_gripper_depth(scene_constants)

      export_to_disk(scene_constants)
      succeeded_eps.append(ep_id)
      print(f"  ✅ Episode {ep_id} completed successfully.")
    except Exception as e:
      print(f"  ❌ Episode {ep_id} failed: {e}")
      continue

  # Append successfully processed episodes to episodes_stage1.txt (multi-process safe)
  stage1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "episodes_stage1.txt")
  if succeeded_eps:
    batch = "".join(ep_id + "\n" for ep_id in succeeded_eps)
    with open(stage1_path, "a") as f:
      fcntl.flock(f, fcntl.LOCK_EX)
      f.write(batch)
      fcntl.flock(f, fcntl.LOCK_UN)
    print(f"\n📝 Appended {len(succeeded_eps)} episodes to {stage1_path}")

  print(f"\n🎉 Pipeline complete! {len(succeeded_eps)}/{len(target_eps)} episodes succeeded.")
