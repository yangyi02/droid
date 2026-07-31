"""DROID Episode Extraction and Vision Foundation Model Pipeline.

A minimalist, high-efficiency multi-stage processor for decoding raw ZED SVO
stereo video streams, extracting robot kinematics, and inferring metric depth.
"""

import argparse
import glob
import json
import os
import random
import sys

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from tqdm import tqdm

from core.depth import (build_universal_gripper_mask, compute_stereo_depth,
                        distill_empirical_gripper_depth, inject_gripper_depth)
from core.geometry import make_4x4
from core.io import get_accelerator, load_metadata

# Foundation Models
def init_all_models():
  """Load vision foundation models and vendor dependencies dynamically."""
  device = get_accelerator()
  print(f"Launching models onto {device} | CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
  if not torch.cuda.is_available():
    print("[WARN] WARNING: PyTorch cannot find a valid CUDA device. Please ensure your CUDA_VISIBLE_DEVICES index is correct (e.g. 0 or 1) and NVIDIA drivers are running.")

  # Inject third-party repo paths just-in-time
  vendor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party")
  for pkg in ["s2m2/src"]:
    path = os.path.join(vendor_dir, pkg)
    if path not in sys.path:
      sys.path.append(path)

  # Lazy importing — these modules live under third_party/ paths added above
  from s2m2.core.utils.model_utils import load_model, run_stereo_matching
  from segment_anything import sam_model_registry, SamPredictor

  s2m2_model = torch.compile(
      load_model(
          os.path.join(vendor_dir, "s2m2/weights/pretrain_weights"),
          "XL",
          True,
          3,
          device,
      ).eval()
  )
  sam = sam_model_registry["vit_h"](
      checkpoint=os.path.join(vendor_dir, "sam_weights/sam_vit_h_4b8939.pth")
  ).to(device)

  print("  All foundation models loaded.")
  return s2m2_model, SamPredictor(sam), run_stereo_matching


# SVO Decoding & Kinematics Extraction
def init_episode(episode_id, root_path, id_to_path, serials_db, keep_ranges_db):
  """Build the hierarchical scene_constants dict for one episode."""
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
    print(f"  Loaded action ranges, marked {len(valid_indices)} frames as valid keyframes.")
  else:
    print(f"  [WARN] No idle filter info found for {episode_id}, keeping all frames.")

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


def extract_svo_video(scene_constants, min_frames=0, max_frames=250):
  """Decode SVO video, extract full stereo calibration, both rectified/unrectified frames, and timestamps."""
  import pyzed.sl as sl

  print("  Fast-decoding SVO video streams and physical calibration data (including timestamps)...")
  episode_path = scene_constants["meta"]["episode_path"]

  for cam in scene_constants["camera"]:
    svo_files = glob.glob(os.path.join(episode_path, f"**/{cam}.svo"), recursive=True)
    if not svo_files:
      continue

    zed, init_params = sl.Camera(), sl.InitParameters()
    init_params.set_from_svo_file(svo_files[0])
    init_params.svo_real_time_mode = False
    zed.open(init_params)

    # Frame count gate (early exit before expensive decoding)
    n_svo_frames = zed.get_svo_number_of_frames() - 2
    if max_frames > 0 and n_svo_frames > max_frames:
      print(f"  Skipping SVO [{cam}]: {n_svo_frames} frames exceeds --max_frames={max_frames}.")
      zed.close()
      continue
    if min_frames > 0 and n_svo_frames < min_frames:
      print(f"  Skipping SVO [{cam}]: {n_svo_frames} frames below --min_frames={min_frames}.")
      zed.close()
      continue

    # Extract ZED native calibration data
    cam_info = zed.get_camera_information()

    # Rectified (Calibrated)
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

    # Raw (Distorted)
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

    # Frame extraction loop (left/right + raw)
    all_left, all_right, all_left_raw, all_right_raw = [], [], [], []
    all_timestamps = []  # list for storing per-frame timestamps
    left_mat, right_mat = sl.Mat(), sl.Mat()
    left_raw_mat, right_raw_mat = sl.Mat(), sl.Mat()

    for _ in tqdm(range(zed.get_svo_number_of_frames()), desc=f"Decoding {cam}"):
      if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # capture hardware exposure timestamp for the current frame (ms)
        timestamp_ms = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        all_timestamps.append(timestamp_ms)

        # Extract rectified stereo frames
        zed.retrieve_image(left_mat, sl.VIEW.LEFT)
        zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
        # Extract raw unrectified stereo frames
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
        "timestamps": np.array(all_timestamps),  # timestamp array packed into scene dict
    })

  return scene_constants


def parse_robot_kinematics(scene_constants):
  """Load H5 and JSON to extract robot kinematics, hand-eye matrices, and timestamps."""
  print("  Parsing robot H5 kinematics and dynamic hand-eye matrices...")
  ep_path = scene_constants["meta"]["episode_path"]

  with h5py.File(f"{ep_path}/trajectory.h5", "r") as f:
    ee_poses = f["observation/robot_state/cartesian_position"][:]
    joint_poses = f["observation/robot_state/joint_positions"][:]
    gripper_poses = f["observation/robot_state/gripper_position"][:]

    # extract robot state timestamps from H5
    timestamps = f["observation/timestamp/robot_state/read_start"][:]

  with open(glob.glob(f"{ep_path}/metadata_*.json")[0]) as jf:
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
      "timestamps": timestamps,  # robot state timestamps
  }
  return scene_constants


def align_temporal_streams(scene_constants):
  """Truncate all temporal streams to the shortest length for global alignment."""
  print("  Running global temporal alignment check...")

  # Collect lengths of all temporal streams
  lengths = [
      len(scene_constants["robot"]["joint_positions"]),
      len(scene_constants["robot"]["gripper_positions"]),
      len(scene_constants["robot"]["T_ee_base_all"]),
  ]
  for cam_id, cam_data in scene_constants["camera"].items():
    lengths.append(len(cam_data["video_rgb"]))
    lengths.append(len(cam_data["video_right"]))

  # Find the shortest stream (bottleneck)
  min_frames = min(lengths)
  max_frames = max(lengths)

  if min_frames == max_frames:
    print(f"    Temporal streams perfectly aligned at {min_frames} frames.")
    return scene_constants

  print(f"    [WARN] Temporal mismatch detected (max {max_frames}, min {min_frames})!")
  print(f"    Truncating all streams to {min_frames} frames...")

  # Truncate all dimensions to min_frames (dynamic traversal)
  for key in ["joint_positions", "gripper_positions", "T_ee_base_all", "timestamps"]:
    if key in scene_constants["robot"]:
      scene_constants["robot"][key] = scene_constants["robot"][key][:min_frames]

  for cam_id, cam_data in scene_constants["camera"].items():
    for key, value in cam_data.items():
      # Truncate any array/list whose length matches the max frame count
      if isinstance(value, (list, np.ndarray)) and len(value) == max_frames:
        cam_data[key] = value[:min_frames]

  print("    Global alignment complete. All dimensions are now consistent.")
  return scene_constants


def _write_mp4(path, frames, fps=10.0):
  """Write a sequence of RGB frames to an mp4 file."""
  h, w = frames[0].shape[:2]
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
  for img in frames:
    writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
  writer.release()


def export_to_disk(scene_constants, export_root="~/droid_data/output/mv-tap/droid/depth"):
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

  print(f"  Exporting multi-view data to {out_dir}...")

  # Robot kinematics (episode-level)
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

  # Per-camera data
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


# Execution & Batched Slicing
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="DROID Flexible Pipeline Extractor")
  parser.add_argument("--rank", type=int, default=0, help="Rank of the process")
  parser.add_argument("--world_size", type=int, default=1, help="Total number of processes")
  parser.add_argument("--limit", type=int, default=-1, help="Limit total number of episodes to process")
  parser.add_argument("--min_frames", type=int, default=48, help="Skip episodes with fewer than this many frames (default: 48, -1 to disable)")
  parser.add_argument("--max_frames", type=int, default=250, help="Skip episodes with more than this many frames (default: 250, -1 to disable)")

  args = parser.parse_args()

  print("Environment setup verified. Initializing flexible multi-GPU extractor...")
  device = get_accelerator()
  s2m2_model, sam_predictor, run_stereo_matching = init_all_models()
  serials_db, id_to_path, keep_ranges, extrinsics_db, valid_ids = load_metadata()

  random.seed(42)
  random.shuffle(valid_ids)
  if args.limit > 0:
    valid_ids = valid_ids[:args.limit]
  target_eps = valid_ids[args.rank::args.world_size]
  print(f"Selected via distributed rank {args.rank}/{args.world_size} targeting: {len(target_eps)} episodes")

  succeeded_eps = []

  for idx, ep_id in enumerate(target_eps):
    print(f"\n[{idx + 1}/{len(target_eps)}] Processing Episode: {ep_id}")
    if ep_id not in id_to_path:
      print(f"  [FAIL] Invalid episode ID: {ep_id}")
      continue

    try:
      scene_constants = init_episode(
          ep_id,
          os.path.expanduser("~/droid_data/input/robotics/droid_raw/1.0.1"),
          id_to_path,
          serials_db,
          keep_ranges,
      )
      scene_constants = extract_svo_video(scene_constants, min_frames=args.min_frames, max_frames=args.max_frames)
      if not any("video_rgb" in data for data in scene_constants["camera"].values()):
        print(f"  [WARN] No valid video streams extracted for [{ep_id}]. Skipping processing.")
        continue

      scene_constants = parse_robot_kinematics(scene_constants)
      scene_constants = align_temporal_streams(scene_constants)
      scene_constants = compute_stereo_depth(
          scene_constants, s2m2_model, run_stereo_matching, device)

      # Gripper depth refinement (wrist camera only)
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
      print(f"  Episode {ep_id} completed successfully.")
    except Exception as e:
      print(f"  [FAIL] Episode {ep_id} failed: {e}")
      continue

  print(f"\nPipeline complete! {len(succeeded_eps)}/{len(target_eps)} episodes succeeded.")
