import glob
import json
import os
import sys

import cv2
import h5py
import mediapy as media
import numpy as np
from absl import app
from ml_collections import config_flags
from scipy.spatial.transform import Rotation as R
import torch
from tqdm import tqdm

import config
import core.depth
import core.geometry
import core.io
import core.runner


def init_all_models():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  vendor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party")
  sys.path.append(os.path.join(vendor_dir, "s2m2", "src"))

  from s2m2.core.utils.model_utils import load_model, run_stereo_matching
  from segment_anything import SamPredictor, sam_model_registry

  s2m2_model = torch.compile(load_model(os.path.join(vendor_dir, "s2m2", "weights"), "XL", True, 3, device).eval())
  sam_ckpt = os.path.join(vendor_dir, "segment_anything", "weights", "sam_vit_h_4b8939.pth")
  sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(device)

  return s2m2_model, SamPredictor(sam), run_stereo_matching


def stereo_intrinsics(params):
  def camera_matrix(cam):
    return np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], dtype=np.float32)

  return {
    "K": camera_matrix(params.left_cam),
    "disto": np.array(params.left_cam.disto, dtype=np.float32),
    "K_right": camera_matrix(params.right_cam),
    "disto_right": np.array(params.right_cam.disto, dtype=np.float32),
  }


def init_episode(episode_id, root_path, id_to_path, serials_db):
  cam_info = serials_db[episode_id]
  wrist_cam_id = cam_info["wrist_cam_serial"]

  return {
    "meta": {
      "episode_id": episode_id,
      "episode_path": os.path.join(root_path, id_to_path[episode_id]),
      "wrist_serial": wrist_cam_id,
    },
    "robot": {},
    "camera": {
      cam_id: {"baseline": 0.063 if cam_id == wrist_cam_id else 0.120} for cam_id in sorted(set(cam_info.values()))
    },
  }


def extract_svo_video(episode, min_frames, max_frames):
  import pyzed.sl as sl

  episode_path = episode["meta"]["episode_path"]

  for cam_id in episode["camera"]:
    svo_file = glob.glob(os.path.join(episode_path, f"**/{cam_id}.svo"), recursive=True)[0]

    zed, init_params = sl.Camera(), sl.InitParameters()
    init_params.set_from_svo_file(svo_file)
    init_params.svo_real_time_mode = False
    zed.open(init_params)

    n_svo_frames = zed.get_svo_number_of_frames() - 2
    if not min_frames <= n_svo_frames <= max_frames:
      zed.close()
      raise core.runner.SkipEpisode(f"{cam_id}: {n_svo_frames} frames outside [{min_frames}, {max_frames}]")

    cam_info = zed.get_camera_information()
    configuration = cam_info.camera_configuration

    views = {
      "video_rgb": sl.VIEW.LEFT,
      "video_right": sl.VIEW.RIGHT,
      "video_raw_rgb": sl.VIEW.LEFT_UNRECTIFIED,
      "video_raw_right": sl.VIEW.RIGHT_UNRECTIFIED,
    }
    mats = {key: sl.Mat() for key in views}
    frames = {key: [] for key in views}
    timestamps = []

    for _ in tqdm(range(n_svo_frames + 2), desc=f"Decoding {cam_id}"):
      if zed.grab() != sl.ERROR_CODE.SUCCESS:
        continue

      timestamps.append(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds())
      for key, view in views.items():
        zed.retrieve_image(mats[key], view)
        frames[key].append(cv2.cvtColor(mats[key].get_data(), cv2.COLOR_BGRA2RGB))

    zed.close()

    calibrated = stereo_intrinsics(configuration.calibration_parameters)
    episode["camera"][cam_id].update(
      {key: np.stack(images) for key, images in frames.items()},
      K=calibrated["K"],
      zed_calibration={"calibrated": calibrated, "raw": stereo_intrinsics(configuration.calibration_parameters_raw)},
      timestamps=np.array(timestamps),
    )

  return episode


def parse_robot_kinematics(episode):
  ep_path = episode["meta"]["episode_path"]

  with h5py.File(f"{ep_path}/trajectory.h5", "r") as f:
    ee_poses = f["observation/robot_state/cartesian_position"][:]
    joint_poses = f["observation/robot_state/joint_positions"][:]
    gripper_poses = f["observation/robot_state/gripper_position"][:]

    timestamps = f["observation/timestamp/robot_state/read_start"][:]

  with open(glob.glob(f"{ep_path}/metadata_*.json")[0]) as jf:
    wrist_ext = json.load(jf)["wrist_cam_extrinsics"]

  total_frames = len(ee_poses)

  T_ee2base = np.tile(np.eye(4), (total_frames, 1, 1))
  T_ee2base[:, :3, :3] = R.from_euler("xyz", ee_poses[:, 3:]).as_matrix()
  T_ee2base[:, :3, 3] = ee_poses[:, :3]

  episode["robot"] = {
    "joint_positions": joint_poses,
    "gripper_positions": gripper_poses,
    "T_cam_ee_init": (
      np.linalg.inv(core.geometry.pose_from_euler(ee_poses[0])) @ core.geometry.pose_from_euler(wrist_ext)
    ),
    "T_ee_base_all": T_ee2base,
    "timestamps": timestamps,
  }
  return episode


def align_temporal_streams(episode):
  robot_streams = ["joint_positions", "gripper_positions", "T_ee_base_all", "timestamps"]
  camera_streams = ["video_rgb", "video_right", "video_raw_rgb", "video_raw_right", "timestamps"]

  streams = [(episode["robot"], key) for key in robot_streams]
  streams += [(cam_data, key) for cam_data in episode["camera"].values() for key in camera_streams]

  n_frames = min(len(owner[key]) for owner, key in streams)
  for owner, key in streams:
    owner[key] = owner[key][:n_frames]

  return episode


def export_depth(episode, export_root):
  wrist_cam_id = episode["meta"]["wrist_serial"]
  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(export_root, episode["meta"]["episode_id"])))
  os.makedirs(ep_dir, exist_ok=True)

  for cam_id, data in episode["camera"].items():
    cam_dir = os.path.join(ep_dir, str(cam_id))
    os.makedirs(cam_dir, exist_ok=True)

    for key, filename in [
      ("video_rgb", "video_left.mp4"),
      ("video_right", "video_right.mp4"),
      ("video_raw_rgb", "video_left_raw.mp4"),
      ("video_raw_right", "video_right_raw.mp4"),
    ]:
      media.write_video(os.path.join(cam_dir, filename), data[key], fps=10)

    np.savez_compressed(os.path.join(cam_dir, "raw_depth.npz"), depth=(data["raw_depth"] * 1000).astype(np.uint16))

    calibrated, raw = data["zed_calibration"]["calibrated"], data["zed_calibration"]["raw"]
    np.savez(
      os.path.join(cam_dir, "calibration.npz"),
      K_calib_left=calibrated["K"],
      K_calib_right=calibrated["K_right"],
      disto_calib_left=calibrated["disto"],
      disto_calib_right=calibrated["disto_right"],
      K_raw_left=raw["K"],
      K_raw_right=raw["K_right"],
      disto_raw_left=raw["disto"],
      disto_raw_right=raw["disto_right"],
      baseline=np.array(data["baseline"], dtype=np.float32),
    )

    if cam_id == wrist_cam_id:
      np.savez_compressed(os.path.join(cam_dir, "gripper_mask.npz"), mask=data["sam_real_masks"])
      for key, filename in [
        ("original_raw_depth", "original_raw_depth.npz"),
        ("empirical_gripper_depth", "gripper_depth.npz"),
      ]:
        np.savez_compressed(os.path.join(cam_dir, filename), depth=(data[key] * 1000).astype(np.uint16))

  robot = episode["robot"]
  np.savez_compressed(
    os.path.join(ep_dir, "robot.npz"),
    wrist_serial=np.array(wrist_cam_id),
    joint_positions=robot["joint_positions"].astype(np.float32),
    gripper_positions=robot["gripper_positions"].astype(np.float32),
    T_ee_base_all=robot["T_ee_base_all"].astype(np.float32),
    T_cam_ee_init=robot["T_cam_ee_init"].astype(np.float32),
  )

  return ep_dir


def process_episode(episode_id, models, dbs, raw_root, config):
  s2m2_model, sam_predictor, run_stereo_matching, device = models
  id_to_path, serials_db = dbs

  episode = init_episode(episode_id, raw_root, id_to_path, serials_db)
  episode = extract_svo_video(episode, config.depth.min_frames, config.depth.max_frames)
  episode = parse_robot_kinematics(episode)
  episode = align_temporal_streams(episode)
  episode = core.depth.compute_stereo_depth(episode, s2m2_model, run_stereo_matching, device, config.depth.conf_thresh)

  wrist_data = episode["camera"][episode["meta"]["wrist_serial"]]
  wrist_data["original_raw_depth"] = wrist_data["raw_depth"].copy()

  episode = core.depth.build_universal_gripper_mask(
    episode,
    sam_predictor,
    config.depth.consensus_thresh,
    config.depth.gripper_closed_thresh,
    config.depth.mask_area_min,
    config.depth.mask_area_max,
  )
  episode = core.depth.distill_empirical_gripper_depth(
    episode, config.depth.max_depth_thresh, config.depth.gripper_closed_thresh
  )
  episode = core.depth.inject_gripper_depth(episode, config.depth.gripper_closed_thresh)
  export_depth(episode, export_root=config.paths.depth)


def main(_):
  config = config_flag.value
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  s2m2_model, sam_predictor, run_stereo_matching = init_all_models()
  serials_db, id_to_path, _, valid_ids = core.io.load_metadata(config)
  raw_root = os.path.expanduser(config.paths.raw)

  target = core.runner.shard_episodes(valid_ids, config.runner.rank, config.runner.world_size, config.runner.limit)
  export_root = os.path.abspath(os.path.expanduser(config.paths.depth))
  done = {e for e in target if os.path.exists(os.path.join(export_root, e, "robot.npz"))}

  def run_one(episode_id):
    process_episode(
      episode_id,
      (s2m2_model, sam_predictor, run_stereo_matching, device),
      (id_to_path, serials_db),
      raw_root,
      config,
    )

  core.runner.run_episodes(
    target,
    run_one,
    rank=config.runner.rank,
    world_size=config.runner.world_size,
    done=done,
    stage="Stage 1",
  )


if __name__ == "__main__":
  config_flag = config_flags.DEFINE_config_file("config", config.__file__)
  app.run(main)
