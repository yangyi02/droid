"""Data I/O helpers shared across pipeline stages.

Provides accelerator setup, metadata loading, and Stage 1/2 output reading.

DATA CONTRACT -- scene_constants

The primary in-memory data structure shared across all three pipeline stages.
All keys are optional unless marked (required).

  scene_constants = {
    'meta': {
      'episode_id':    str              (required) e.g. "ILIAD+5e938e3b+2023-07-20"
      'wrist_serial':  str              serial number of the wrist camera
      'valid_indices': np.ndarray[int]  frame indices after idle filtering
      'episode_path':  str              local path to raw episode dir
    },

    'robot': {
      'joint_positions':    np.float32 (T, 7)   Franka joint angles
      'gripper_positions':  np.float32 (T,)     gripper width
      'T_ee_base_all':      np.float32 (T, 4,4) EE-to-base transform per frame
      'T_cam_ee_init':      np.float32 (4, 4)   hand-eye transform (wrist cam)
    },

    'camera': {
      '<cam_serial>': {
        # Calibration  (all cameras)
        'K_mat':            np.float32 (3, 3)    pinhole intrinsics
        'baseline':         float                stereo baseline in metres
        'zed_calibration':  dict                 raw ZED calibration payload

        # Video  (all cameras, added by extract_svo_video)
        'video_rgb':        np.uint8   (T,H,W,3) rectified left video
        'video_right':      np.uint8   (T,H,W,3) rectified right video
        'video_raw_rgb':    np.uint8   (T,H,W,3) raw (un-rectified) left video
        'video_raw_right':  np.uint8   (T,H,W,3) raw right video
        'first_frame_rgb':  np.uint8   (H,W,3)   first frame only (extrinsics mode)

        # Depth  (all cameras, added by compute_stereo_depth)
        'raw_depth':        np.float32 (T,H,W)   metric depth in metres

        # Wrist camera only  (added by gripper refinement steps)
        'original_raw_depth':       np.float32 (T,H,W)    depth before injection
        'sam_real_masks':           np.bool_   (T,H,W)    SAM gripper mask
        'empirical_gripper_depth':  np.float32 (T,H,W)    distilled gripper depth

        # Tracking  (all cameras, added by phase1_extract_2d_tracks)
        'tracks_2d':  np.float32 (T, N, 2)  CoTracker 2D point tracks
        'vis_2d':     np.bool_   (T, N)      CoTracker visibility mask
      }
    }
  }

DATA CONTRACT -- scene_state

Per-camera extrinsics produced by Stage 2 (compute_extrinsics.py).
Keyed by camera serial, same as scene_constants['camera'].

  scene_state = {
    '<cam_serial>': {
      'base_extrinsic':  np.float32 (4, 4)    static reference camera-to-world
      'extrinsics':      np.float32 (T, 4, 4) per-frame camera-to-world

  export_extrinsics also writes an 'is_wrist' flag into each JSON, but nothing
  reads it back: every caller derives it as cam_id == meta['wrist_serial'], so
  load_extrinsics does not carry it into scene_state.
    }
  }
"""

import json
import os

import cv2
import numpy as np
import torch

# Repository-relative data root. Every default path in the pipeline hangs off
# this, so a clone carries its own data layout and nothing depends on the
# invoking user's home directory. Override any individual root via the CLI.
DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
META_ROOT = os.path.join(DATA_ROOT, "meta", "1.0.1")
INPUT_ROOT = os.path.join(DATA_ROOT, "input")
OUTPUT_ROOT = os.path.join(DATA_ROOT, "output", "mv-tap", "droid")


def get_accelerator(force_egl=True):
  """Configure headless GPU hardware acceleration and return active device."""
  if force_egl:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_metadata(meta_root=META_ROOT):
  """Download (if needed) and load all global DROID dataset JSON mappings.

  This is the single canonical implementation shared by all pipeline stages.
  Files are cached locally in meta_root and only downloaded when absent.

  Args:
    meta_root: local directory to cache downloaded JSON files.

  Returns:
    serials_db:    dict  episode_id → {wrist_cam_serial, ext_cam_serials, ...}
    id_to_path:    dict  episode_id → relative path within raw data root
    keep_ranges:   dict  episode_key → list of (start, end) action frame ranges
    extrinsics_db: dict  episode_id → pre-calibrated extrinsics (may be empty)
    valid_ids:     list  episode IDs present in all three core databases
  """
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
  """Decode a whole mp4 to (T, H, W, 3) uint8 RGB, or None if it is not there."""
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
  """Reconstruct scene_constants from Stage 1 disk outputs.

  Args:
    episode_id: str, episode identifier.
    depth_root: path to Stage 1 depth output root.
    load_video: "first_frame" (extrinsics) | "full" (tracks) | "none".
    inspection: also load the streams only the inspection views read -- the
        right stereo video, the pre-injection depth, and the distilled gripper
        depth. No pipeline stage touches them and they roughly double what an
        episode costs in memory, so they stay off by default.

  Returns:
    scene_constants dict matching the Stage 1 in-memory format.
  """
  ep_dir = os.path.abspath(
      os.path.expanduser(os.path.join(depth_root, episode_id))
  )
  print(f"  Loading depth data from {ep_dir}...")

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

  # Discover camera subdirectories.
  # Avoid per-entry os.path.isdir (slow on gcsfuse); filter by known
  # non-directory filenames instead.
  _NON_DIR_NAMES = {"robot.npz"}
  cam_dirs = [
      d for d in os.listdir(ep_dir)
      if d not in _NON_DIR_NAMES and not d.endswith((".npz", ".json", ".txt"))
  ]

  camera = {}
  for cam_id in sorted(cam_dirs):
    cam_path = os.path.join(ep_dir, cam_id)
    cam_data = {}

    # Load files optimistically (try/except is faster than os.path.exists
    # on gcsfuse mounts where each stat is a GCS API call).

    # Calibration
    try:
      calib = np.load(os.path.join(cam_path, "calibration.npz"))
      cam_data["K_mat"] = calib["K_calib_left"].astype(np.float32)
      if "baseline" in calib:
        cam_data["baseline"] = float(calib["baseline"])
    except FileNotFoundError:
      pass

    # Depth (uint16 mm → float32 meters)
    try:
      depth_uint16 = np.load(os.path.join(cam_path, "raw_depth.npz"))["depth"]
      cam_data["raw_depth"] = depth_uint16.astype(np.float32) / 1000.0
    except FileNotFoundError:
      pass

    # SAM gripper mask (wrist camera only, bool (T, H, W))
    try:
      cam_data["sam_real_masks"] = np.load(
          os.path.join(cam_path, "gripper_mask.npz"))["mask"]
    except FileNotFoundError:
      pass

    if inspection:
      # Depth before the gripper surface was injected, and the distilled
      # surface itself: the two halves the refinement view compares.
      for fname, key in [("original_raw_depth.npz", "original_raw_depth"),
                         ("gripper_depth.npz", "empirical_gripper_depth")]:
        try:
          cam_data[key] = (np.load(os.path.join(cam_path, fname))["depth"]
                           .astype(np.float32) / 1000.0)
        except FileNotFoundError:
          pass

    # Video loading
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
  """Load Stage 2 extrinsics outputs."""
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
