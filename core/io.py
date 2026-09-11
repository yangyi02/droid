import glob
import json
import os

import mediapy as media
import numpy as np


def load_metadata(config):
  root_path = os.path.expanduser(config.paths.meta)
  os.makedirs(root_path, exist_ok=True)

  def fetch(name):
    os.system(f"wget -q -nc -P {root_path} {config.urls.meta}/{name}")
    with open(os.path.join(root_path, name), "r") as fh:
      return json.load(fh)

  serials_db = fetch("camera_serials.json")
  id_to_path = fetch("episode_id_to_path.json")
  extrinsics_db = fetch("cam2base_extrinsic_superset.json")

  valid_ids = []
  for episode_id in sorted(set(serials_db) & set(id_to_path) & set(extrinsics_db)):
    cam_info = serials_db[episode_id]
    external_cams = set(cam_info.values()) - {cam_info["wrist_cam_serial"]}
    calibrated_cams = {cam_id for cam_id, entry in extrinsics_db[episode_id].items() if isinstance(entry, list)}
    if external_cams == calibrated_cams:
      valid_ids.append(episode_id)

  return serials_db, id_to_path, extrinsics_db, valid_ids


def load_depth_data(episode_id, depth_root, load_video=False):
  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(depth_root, episode_id)))

  robot_data = np.load(os.path.join(ep_dir, "robot.npz"))
  wrist_cam_id = str(robot_data["wrist_serial"])
  robot_keys = ("joint_positions", "gripper_positions", "T_ee_base_all", "T_cam_ee_init")

  camera = {}
  for cam_id in sorted(d for d in os.listdir(ep_dir) if os.path.isdir(os.path.join(ep_dir, d))):
    cam_path = os.path.join(ep_dir, cam_id)
    calib = np.load(os.path.join(cam_path, "calibration.npz"))

    cam_data = {
      "K": calib["K_calib_left"].astype(np.float32),
      "baseline": float(calib["baseline"]),
      "raw_depth": np.load(os.path.join(cam_path, "raw_depth.npz"))["depth"].astype(np.float32) / 1000.0,
    }

    if load_video:
      cam_data["video_rgb"] = media.read_video(os.path.join(cam_path, "video_left.mp4"))
      cam_data["video_right"] = media.read_video(os.path.join(cam_path, "video_right.mp4"))

    if cam_id == wrist_cam_id:
      cam_data["sam_real_masks"] = np.load(os.path.join(cam_path, "gripper_mask.npz"))["mask"]
      for filename, key in [
        ("original_raw_depth.npz", "original_raw_depth"),
        ("gripper_depth.npz", "empirical_gripper_depth"),
      ]:
        cam_data[key] = np.load(os.path.join(cam_path, filename))["depth"].astype(np.float32) / 1000.0

    camera[cam_id] = cam_data

  return {
    "meta": {"episode_id": episode_id, "wrist_serial": wrist_cam_id},
    "robot": {key: robot_data[key].astype(np.float32) for key in robot_keys},
    "camera": camera,
  }


def load_extrinsics(episode, extrinsics_root):
  episode_id = episode["meta"]["episode_id"]
  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(extrinsics_root, episode_id)))

  poses = {}
  for cam_id in episode["camera"]:
    with open(os.path.join(ep_dir, cam_id, "extrinsics.json"), "r") as f:
      payload = json.load(f)

    poses[cam_id] = {
      "base_extrinsic": np.array(payload["base_extrinsic"], dtype=np.float32),
      "extrinsics": np.array(payload["extrinsics"], dtype=np.float32),
    }

  return poses


def load_track_data(episode_id, tracks_root):
  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(tracks_root, episode_id)))
  meta_data = np.load(os.path.join(ep_dir, "track_metadata.npz"))

  uv, vis = [], []
  for tracks_path in sorted(glob.glob(os.path.join(ep_dir, "*", "tracks_2d.npz"))):
    cam_data = np.load(tracks_path)
    uv.append(cam_data["tracks_2d"])
    vis.append(cam_data["vis_2d"])

  return {
    "tracks_3d": np.load(os.path.join(ep_dir, "tracks_3d.npz"))["tracks_3d"],
    "uv": np.stack(uv),
    "vis": np.stack(vis),
    "query_view": meta_data["query_view"],
    "n_robot": int(meta_data["n_robot"]),
    "n_static": int(meta_data["n_static"]),
  }
