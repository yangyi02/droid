import json
import os

import cv2
import numpy as np
import torch

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
META_ROOT = os.path.join(DATA_ROOT, "meta", "1.0.1")
INPUT_ROOT = os.path.join(DATA_ROOT, "input")
OUTPUT_ROOT = os.path.join(DATA_ROOT, "output", "droid")


def get_accelerator(force_egl=True):
  if force_egl:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_metadata(meta_root=META_ROOT):
  root_path = os.path.expanduser(meta_root)
  os.makedirs(root_path, exist_ok=True)

  base_url = "https://huggingface.co/KarlP/droid/resolve/main"
  files = [
      "camera_serials.json",
      "episode_id_to_path.json",
      "keep_ranges_1_0_1.json",
      "cam2base_extrinsic_superset.json",
  ]

  print(f"Synchronizing metadata to {root_path}...")
  for f in files:
    dest = os.path.join(root_path, f)
    if not os.path.exists(dest):
      os.system(f"wget -q -nc -P {root_path} {base_url}/{f}")

  def _load(name):
    with open(os.path.join(root_path, name), "r") as fh:
      return json.load(fh)

  serials_db    = _load("camera_serials.json")
  id_to_path    = _load("episode_id_to_path.json")
  keep_ranges   = _load("keep_ranges_1_0_1.json")
  extrinsics_db = _load("cam2base_extrinsic_superset.json")

  valid_ids = sorted(
      set(serials_db.keys()) & set(id_to_path.keys()) & set(extrinsics_db.keys())
  )
  print(f"Metadata ready: {len(valid_ids)} episodes with pre-calibrated extrinsics.")
  return serials_db, id_to_path, keep_ranges, extrinsics_db, valid_ids


def _read_video(path):
  cap = cv2.VideoCapture(path)
  frames = []
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  cap.release()
  return np.array(frames, dtype=np.uint8) if frames else None


def load_depth_data(episode_id, depth_root=os.path.join(OUTPUT_ROOT, "depth"),
                    load_video="first_frame", inspection=False):
  ep_dir = os.path.abspath(
      os.path.expanduser(os.path.join(depth_root, episode_id))
  )
  print(f"  Loading depth data from {ep_dir}...")

  robot_data = np.load(os.path.join(ep_dir, "robot.npz"), allow_pickle=True)
  wrist_serial = (str(robot_data["wrist_serial"])
                  if "wrist_serial" in robot_data else None)

  robot = {
      "joint_positions": robot_data["joint_positions"].astype(np.float32),
      "gripper_positions": robot_data["gripper_positions"].astype(np.float32),
  }
  if "T_ee_base_all" in robot_data:
    robot["T_ee_base_all"] = robot_data["T_ee_base_all"].astype(np.float32)
  if "T_cam_ee_init" in robot_data:
    robot["T_cam_ee_init"] = robot_data["T_cam_ee_init"].astype(np.float32)

  valid_indices = robot_data.get("valid_indices")

  _NON_DIR_NAMES = {"robot.npz"}
  cam_dirs = [
      d for d in os.listdir(ep_dir)
      if d not in _NON_DIR_NAMES and not d.endswith((".npz", ".json", ".txt"))
  ]

  camera = {}
  for cam_id in sorted(cam_dirs):
    cam_path = os.path.join(ep_dir, cam_id)
    cam_data = {}


    try:
      calib = np.load(os.path.join(cam_path, "calibration.npz"))
      cam_data["K_mat"] = calib["K_calib_left"].astype(np.float32)
      if "baseline" in calib:
        cam_data["baseline"] = float(calib["baseline"])
    except FileNotFoundError:
      pass

    try:
      depth_uint16 = np.load(os.path.join(cam_path, "raw_depth.npz"))["depth"]
      cam_data["raw_depth"] = depth_uint16.astype(np.float32) / 1000.0
    except FileNotFoundError:
      pass

    try:
      cam_data["sam_real_masks"] = np.load(
          os.path.join(cam_path, "gripper_mask.npz"))["mask"]
    except FileNotFoundError:
      pass

    if inspection:
      for fname, key in [("original_raw_depth.npz", "original_raw_depth"),
                         ("gripper_depth.npz", "empirical_gripper_depth")]:
        try:
          cam_data[key] = (np.load(os.path.join(cam_path, fname))["depth"]
                           .astype(np.float32) / 1000.0)
        except FileNotFoundError:
          pass

    video_path = os.path.join(cam_path, "video_left.mp4")
    if load_video == "first_frame":
      cap = cv2.VideoCapture(video_path)
      ret, frame = cap.read()
      cap.release()
      if ret:
        cam_data["first_frame_rgb"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif load_video == "full":
      cam_data["video_rgb"] = _read_video(video_path)
      if inspection:
        right = _read_video(os.path.join(cam_path, "video_right.mp4"))
        if right is not None:
          cam_data["video_right"] = right

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
  print(f"  Loaded: {n_cams} cameras, {n_frames} frames.")
  return scene_constants


def load_extrinsics(scene_constants,
                    extrinsics_root=os.path.join(OUTPUT_ROOT, "extrinsics")):
  ep_id = scene_constants["meta"]["episode_id"]
  ep_dir = os.path.abspath(
      os.path.expanduser(os.path.join(extrinsics_root, ep_id))
  )

  scene_state = {}
  for cam_id in scene_constants["camera"]:
    cam_ext_path = os.path.join(ep_dir, cam_id, "extrinsics.json")

    with open(cam_ext_path, "r") as f:
      payload = json.load(f)

    scene_state[cam_id] = {
        "base_extrinsic": np.array(payload["base_extrinsic"],
                                   dtype=np.float32),
        "extrinsics": np.array(payload["extrinsics"], dtype=np.float32),
    }

  print(f"  Loaded extrinsics for {len(scene_state)} cameras.")
  return scene_state
