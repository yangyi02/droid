"""Data I/O helpers shared across pipeline stages.

Provides accelerator setup, metadata loading, and Stage 1/2 output reading.
"""

import json
import os
import sys

import cv2
import numpy as np
import torch


def get_accelerator(force_egl=True):
  """Configure headless GPU hardware acceleration and return active device."""
  if force_egl:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_depth_data(episode_id, depth_root="~/droid_data/output/mv-tap/droid/depth",
                    load_video="first_frame"):
  """Reconstruct scene_constants from Stage 1 disk outputs.

  Args:
    episode_id: str, episode identifier.
    depth_root: path to Stage 1 depth output root.
    load_video: "first_frame" (extrinsics) | "full" (tracks) | "none".

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
      if "baseline" in calib:
        cam_data["baseline"] = float(calib["baseline"])

    # Depth (uint16 mm → float32 meters)
    depth_path = os.path.join(cam_path, "raw_depth.npz")
    if os.path.exists(depth_path):
      depth_uint16 = np.load(depth_path)["depth"]
      cam_data["raw_depth"] = depth_uint16.astype(np.float32) / 1000.0

    # Video loading
    video_path = os.path.join(cam_path, "video_left.mp4")
    if os.path.exists(video_path):
      if load_video == "first_frame":
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
          cam_data["first_frame_rgb"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      elif load_video == "full":
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
          ret, frame = cap.read()
          if not ret:
            break
          frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if frames:
          cam_data["video_rgb"] = np.array(frames, dtype=np.uint8)

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


def load_extrinsics(scene_constants,
                    extrinsics_root="~/droid_data/output/mv-tap/droid/extrinsics"):
  """Load Stage 2 extrinsics outputs."""
  ep_id = scene_constants["meta"]["episode_id"]
  ep_dir = os.path.abspath(
      os.path.expanduser(os.path.join(extrinsics_root, ep_id))
  )

  scene_state = {}
  for cam_id in scene_constants["camera"]:
    cam_ext_path = os.path.join(ep_dir, cam_id, "extrinsics.json")
    if not os.path.exists(cam_ext_path):
      raise FileNotFoundError(f"Extrinsics not found: {cam_ext_path}")

    with open(cam_ext_path, "r") as f:
      payload = json.load(f)

    scene_state[cam_id] = {
        "base_extrinsic": np.array(payload["base_extrinsic"],
                                   dtype=np.float32),
        "extrinsics": np.array(payload["extrinsics"], dtype=np.float32),
    }

  print(f"  ✅ Loaded extrinsics for {len(scene_state)} cameras.")
  return scene_state
