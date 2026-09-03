import argparse
import glob
import json
import os
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
from core.io import INPUT_ROOT, OUTPUT_ROOT, get_accelerator, load_metadata
from core.runner import add_sharding_args, run_episodes, shard_episodes

def init_all_models():
  device = get_accelerator()
  print(f"Launching models onto {device} | CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
  repo_dir = os.path.dirname(os.path.abspath(__file__))
  vendor_dir = os.path.join(repo_dir, "third_party")
  s2m2_src = os.path.join(vendor_dir, "s2m2/src")
  if not os.path.exists(s2m2_src):
    os.system(f"cd '{repo_dir}' && git submodule update --init --recursive")

  for pkg in ["s2m2/src"]:
    path = os.path.join(vendor_dir, pkg)
    if path not in sys.path:
      sys.path.append(path)

  from s2m2.core.utils.model_utils import load_model, run_stereo_matching
  from segment_anything import sam_model_registry, SamPredictor

  s2m2_model = torch.compile(
      load_model(
          os.path.join(vendor_dir, "s2m2", "weights"),
          "XL",
          True,
          3,
          device,
      ).eval()
  )
  sam = sam_model_registry["vit_h"](
      checkpoint=os.path.join(vendor_dir, "sam_weights", "sam_vit_h_4b8939.pth")
  ).to(device)

  print("  All foundation models loaded.")
  return s2m2_model, SamPredictor(sam), run_stereo_matching


def init_episode(episode_id, root_path, id_to_path, serials_db, keep_ranges_db):
  relative_path = id_to_path[episode_id]
  episode_path = os.path.join(root_path, relative_path)

  cam_info = serials_db[episode_id]
  wrist_serial = cam_info.get("wrist_cam_serial")

  valid_cams = sorted(set(cam_info.values()))

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

  return {
      "meta": {
          "episode_id": episode_id,
          "episode_path": episode_path,
          "wrist_serial": wrist_serial,
          "valid_indices": valid_indices,
      },
      "robot": {},
      "camera": {
          cam: {
              "baseline": 0.063 if cam == wrist_serial else 0.120,
          }
          for cam in valid_cams
      },
  }


def extract_svo_video(scene_constants, min_frames=0, max_frames=250):
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

    n_svo_frames = zed.get_svo_number_of_frames() - 2
    if max_frames > 0 and n_svo_frames > max_frames:
      print(f"  Skipping SVO [{cam}]: {n_svo_frames} frames exceeds --max_frames={max_frames}.")
      zed.close()
      continue
    if min_frames > 0 and n_svo_frames < min_frames:
      print(f"  Skipping SVO [{cam}]: {n_svo_frames} frames below --min_frames={min_frames}.")
      zed.close()
      continue

    cam_info = zed.get_camera_information()

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

    all_left, all_right, all_left_raw, all_right_raw = [], [], [], []
    all_timestamps = []
    left_mat, right_mat = sl.Mat(), sl.Mat()
    left_raw_mat, right_raw_mat = sl.Mat(), sl.Mat()

    for _ in tqdm(range(zed.get_svo_number_of_frames()), desc=f"Decoding {cam}"):
      if zed.grab() == sl.ERROR_CODE.SUCCESS:
        timestamp_ms = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        all_timestamps.append(timestamp_ms)

        zed.retrieve_image(left_mat, sl.VIEW.LEFT)
        zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
        zed.retrieve_image(left_raw_mat, sl.VIEW.LEFT_UNRECTIFIED)
        zed.retrieve_image(right_raw_mat, sl.VIEW.RIGHT_UNRECTIFIED)

        all_left.append(cv2.cvtColor(left_mat.get_data(), cv2.COLOR_BGRA2RGB))
        all_right.append(cv2.cvtColor(right_mat.get_data(), cv2.COLOR_BGRA2RGB))
        all_left_raw.append(cv2.cvtColor(left_raw_mat.get_data(), cv2.COLOR_BGRA2RGB))
        all_right_raw.append(cv2.cvtColor(right_raw_mat.get_data(), cv2.COLOR_BGRA2RGB))

    zed.close()

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
        "timestamps": np.array(all_timestamps),
    })

  return scene_constants


def parse_robot_kinematics(scene_constants):
  print("  Parsing robot H5 kinematics and dynamic hand-eye matrices...")
  ep_path = scene_constants["meta"]["episode_path"]

  with h5py.File(f"{ep_path}/trajectory.h5", "r") as f:
    ee_poses = f["observation/robot_state/cartesian_position"][:]
    joint_poses = f["observation/robot_state/joint_positions"][:]
    gripper_poses = f["observation/robot_state/gripper_position"][:]

    timestamps = f["observation/timestamp/robot_state/read_start"][:]

  with open(glob.glob(f"{ep_path}/metadata_*.json")[0]) as jf:
    wrist_ext = json.load(jf)["wrist_cam_extrinsics"]
    wrist_ext = wrist_ext.get("extrinsics", wrist_ext) if isinstance(wrist_ext, dict) else wrist_ext

  total_frames = len(ee_poses)

  T_ee_all = np.tile(np.eye(4), (total_frames, 1, 1))
  T_ee_all[:, :3, :3] = R.from_euler("xyz", ee_poses[:, 3:]).as_matrix()
  T_ee_all[:, :3, 3] = ee_poses[:, :3]

  scene_constants["robot"] = {
      "joint_positions": joint_poses,
      "gripper_positions": gripper_poses,
      "T_cam_ee_init": np.linalg.inv(make_4x4(ee_poses[0])) @ make_4x4(wrist_ext),
      "T_ee_base_all": T_ee_all,
      "timestamps": timestamps,
  }
  return scene_constants


def align_temporal_streams(scene_constants):
  print("  Running global temporal alignment check...")

  lengths = [
      len(scene_constants["robot"]["joint_positions"]),
      len(scene_constants["robot"]["gripper_positions"]),
      len(scene_constants["robot"]["T_ee_base_all"]),
  ]
  for cam_id, cam_data in scene_constants["camera"].items():
    lengths.append(len(cam_data["video_rgb"]))
    lengths.append(len(cam_data["video_right"]))

  min_frames = min(lengths)
  max_frames = max(lengths)

  print(f"    Truncating all streams to {min_frames} frames...")

  for key in ["joint_positions", "gripper_positions", "T_ee_base_all", "timestamps"]:
    if key in scene_constants["robot"]:
      scene_constants["robot"][key] = scene_constants["robot"][key][:min_frames]

  for cam_id, cam_data in scene_constants["camera"].items():
    for key, value in cam_data.items():
      if isinstance(value, (list, np.ndarray)) and len(value) == max_frames:
        cam_data[key] = value[:min_frames]

  print("    Global alignment complete. All dimensions are now consistent.")
  return scene_constants


def _write_mp4(path, frames, fps=10.0):
  h, w = frames[0].shape[:2]
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
  for img in frames:
    writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
  writer.release()


def export_depth(scene_constants, export_root=os.path.join(OUTPUT_ROOT, "depth")):
  ep_id = scene_constants["meta"]["episode_id"]
  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(export_root, ep_id)))
  os.makedirs(ep_dir, exist_ok=True)

  print(f"  Exporting multi-view data to {ep_dir}...")

  for cam_id, data in scene_constants["camera"].items():
    cam_dir = os.path.join(ep_dir, str(cam_id))
    os.makedirs(cam_dir, exist_ok=True)

    video_keys = {
        "video_rgb": "video_left.mp4",
        "video_right": "video_right.mp4",
        "video_raw_rgb": "video_left_raw.mp4",
        "video_raw_right": "video_right_raw.mp4",
    }
    for key, filename in video_keys.items():
      if key in data and len(data[key]) > 0:
        _write_mp4(os.path.join(cam_dir, filename), data[key])

    if "original_raw_depth" in data:
      np.savez_compressed(
          os.path.join(cam_dir, "original_raw_depth.npz"),
          depth=(data["original_raw_depth"] * 1000).astype(np.uint16),
      )
    if "raw_depth" in data:
      np.savez_compressed(
          os.path.join(cam_dir, "raw_depth.npz"),
          depth=(data["raw_depth"] * 1000).astype(np.uint16),
      )

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

    if "zed_calibration" in data:
      calib = data["zed_calibration"]
      np.savez(
          os.path.join(cam_dir, "calibration.npz"),
          K_calib_left=calib["calibrated"]["K"],
          K_calib_right=calib["calibrated"]["K_right"],
          disto_calib_left=calib["calibrated"]["disto"],
          disto_calib_right=calib["calibrated"]["disto_right"],
          K_raw_left=calib["raw"]["K"],
          K_raw_right=calib["raw"]["K_right"],
          disto_raw_left=calib["raw"]["disto"],
          disto_raw_right=calib["raw"]["disto_right"],
          baseline=np.array(data["baseline"], dtype=np.float32),
      )


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
    meta = scene_constants["meta"]
    if meta.get("valid_indices") is not None:
      robot_save["valid_indices"] = meta["valid_indices"]
    if meta.get("wrist_serial") is not None:
      robot_save["wrist_serial"] = np.array(meta["wrist_serial"])
    np.savez_compressed(os.path.join(ep_dir, "robot.npz"), **robot_save)

  return ep_dir


def process_episode(ep_id, models, dbs, raw_root, min_frames, max_frames,
                    export_root):
  s2m2_model, sam_predictor, run_stereo_matching, device = models
  id_to_path, serials_db, keep_ranges = dbs

  scene_constants = init_episode(
      ep_id, raw_root, id_to_path, serials_db, keep_ranges)
  scene_constants = extract_svo_video(
      scene_constants, min_frames=min_frames, max_frames=max_frames)
  scene_constants = parse_robot_kinematics(scene_constants)
  scene_constants = align_temporal_streams(scene_constants)
  scene_constants = compute_stereo_depth(
      scene_constants, s2m2_model, run_stereo_matching, device)

  wrist_serial = scene_constants["meta"].get("wrist_serial")
  wrist_data = scene_constants["camera"].get(wrist_serial or "", {})
  if "raw_depth" in wrist_data:
    wrist_data["original_raw_depth"] = wrist_data["raw_depth"].copy()

  scene_constants = build_universal_gripper_mask(scene_constants, sam_predictor)
  scene_constants = distill_empirical_gripper_depth(scene_constants)
  scene_constants = inject_gripper_depth(scene_constants)
  export_depth(scene_constants, export_root=export_root)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="DROID Stage 1: Stereo Depth")
  add_sharding_args(parser)
  parser.add_argument("--min_frames", type=int, default=48,
                      help="Skip episodes with fewer than this many frames "
                           "(default: 48, -1 to disable)")
  parser.add_argument("--max_frames", type=int, default=250,
                      help="Skip episodes with more than this many frames "
                           "(default: 250, -1 to disable)")
  parser.add_argument("--export_root", type=str,
                      default=os.path.join(OUTPUT_ROOT, "depth"),
                      help="Root directory for depth output")
  args = parser.parse_args()

  print("DROID Stage 1: Stereo Depth")
  device = get_accelerator()
  s2m2_model, sam_predictor, run_stereo_matching = init_all_models()
  serials_db, id_to_path, keep_ranges, _, valid_ids = load_metadata()
  raw_root = os.path.expanduser(
      os.path.join(INPUT_ROOT, "robotics", "droid_raw", "1.0.1"))

  target = shard_episodes(valid_ids, args.rank, args.world_size, args.limit)
  export_abs = os.path.abspath(os.path.expanduser(args.export_root))
  done = {ep for ep in target
          if os.path.exists(os.path.join(export_abs, ep, "robot.npz"))}

  run_episodes(
      target,
      lambda ep_id: process_episode(
          ep_id, (s2m2_model, sam_predictor, run_stereo_matching, device),
          (id_to_path, serials_db, keep_ranges),
          raw_root, args.min_frames, args.max_frames,
          args.export_root),
      rank=args.rank, world_size=args.world_size,
      done=done, stage="Stage 1")
