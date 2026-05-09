"""Data loading — episode download, SVO decode, kinematics, temporal ops.

Test:
  python -c "from droid.data_loader import load_metadata; print('✅ data_loader OK')"
"""

import copy
import glob
import json
import os

import cv2
import h5py
import numpy as np
import pyzed.sl as sl
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from core.geometry import make_4x4


# ==========================================
# Metadata
# ==========================================

def load_metadata(data_root):
    """Load DROID dataset metadata JSONs.

    Returns:
        serials_db: episode_id -> camera serial mapping
        id_to_path: episode_id -> relative path
        keep_ranges: idle filtering ranges
        extrinsics_db: episode_id -> pre-calibrated cam2base extrinsics
        valid_ids: sorted list of episodes with all metadata
    """
    def load_json(name):
        path = os.path.join(data_root, name)
        with open(path, 'r') as f:
            return json.load(f)

    serials_db = load_json("camera_serials.json")
    id_to_path = load_json("episode_id_to_path.json")
    keep_ranges = load_json("keep_ranges_1_0_1.json")
    extrinsics_db = load_json("cam2base_extrinsic_superset.json")

    valid_ids = sorted(
        set(serials_db.keys()) & set(id_to_path.keys()) & set(extrinsics_db.keys())
    )
    print(f"✅ Metadata loaded. {len(valid_ids)} episodes with extrinsics.")
    return serials_db, id_to_path, keep_ranges, extrinsics_db, valid_ids


# ==========================================
# Episode Download
# ==========================================

def download_episode(episode_id, data_root, id_to_path, serials_db, keep_ranges_db):
    """Download episode data and initialize scene_constants structure.

    Returns:
        scene_constants dict with keys: meta, robot (empty), camera (per-serial).
    """
    relative_path = id_to_path[episode_id]
    episode_path = os.path.join(data_root, relative_path)

    if not os.path.exists(episode_path):
        os.makedirs(episode_path, exist_ok=True)
        os.system(
            f'gsutil -m -q cp -r "gs://gresearch/robotics/droid_raw/1.0.1/{relative_path}/*" '
            f'"{episode_path}/"'
        )
        print(f"  ⬇️ Downloaded: {episode_id}")
    else:
        print(f"  ⏭️ Already exists, skipping download.")

    cam_info = serials_db[episode_id]
    wrist_serial = cam_info.get('wrist_cam_serial')
    valid_cams = sorted(set(cam_info.values()))

    # Build keep_ranges key
    base_prefix = "gs://xembodiment_data/r2d2/r2d2-data-full/"
    episode_key = (
        f"{base_prefix}{relative_path}/recordings/MP4--"
        f"{base_prefix}{relative_path}/trajectory.h5"
    )

    valid_indices = None
    if episode_key in keep_ranges_db:
        ranges = keep_ranges_db[episode_key]
        indices = []
        for start, end in ranges:
            indices.extend(range(start, end))
        valid_indices = np.array(indices)
        print(f"  ✂️ Loaded action ranges: {len(valid_indices)} valid frames.")
    else:
        print(f"  ⚠️ No idle filter data found, keeping all frames.")

    return {
        'meta': {
            'episode_id': episode_id,
            'episode_path': episode_path,
            'wrist_serial': wrist_serial,
            'valid_indices': valid_indices,
        },
        'robot': {},
        'camera': {
            cam: {'baseline': 0.063 if cam == wrist_serial else 0.120}
            for cam in valid_cams
        },
    }


# ==========================================
# SVO Video Extraction
# ==========================================

def extract_svo_video(scene_constants):
    """Decode SVO stereo video files and extract calibration data."""
    print("  🎥 Decoding SVO video streams...")
    episode_path = scene_constants['meta']['episode_path']

    for cam in scene_constants['camera']:
        svo_path = glob.glob(
            os.path.join(episode_path, f"**/{cam}.svo"), recursive=True
        )[0]

        zed, init_params = sl.Camera(), sl.InitParameters()
        init_params.set_from_svo_file(svo_path)
        init_params.svo_real_time_mode = False
        zed.open(init_params)

        cam_info = zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters

        K_calib_left = np.array([
            [calib.left_cam.fx, 0, calib.left_cam.cx],
            [0, calib.left_cam.fy, calib.left_cam.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        all_left, all_right = [], []
        left_mat, right_mat = sl.Mat(), sl.Mat()

        for _ in tqdm(range(zed.get_svo_number_of_frames()), desc=f"  SVO [{cam}]"):
            zed.grab()
            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
            all_left.append(cv2.cvtColor(left_mat.get_data(), cv2.COLOR_BGRA2RGB))
            all_right.append(cv2.cvtColor(right_mat.get_data(), cv2.COLOR_BGRA2RGB))

        zed.close()

        scene_constants['camera'][cam].update({
            'K_mat': K_calib_left,
            'video_rgb': np.stack(all_left),
            'video_right': np.stack(all_right),
        })

    return scene_constants


# ==========================================
# Robot Kinematics
# ==========================================

def parse_robot_kinematics(scene_constants):
    """Read H5 trajectory and extract robot kinematics."""
    print("  🦾 Parsing robot kinematics...")
    ep_path = scene_constants['meta']['episode_path']

    with h5py.File(f"{ep_path}/trajectory.h5", 'r') as f, \
         open(glob.glob(f"{ep_path}/metadata_*.json")[0]) as jf:
        ee_poses = f["observation/robot_state/cartesian_position"][:]
        joint_poses = f["observation/robot_state/joint_positions"][:]
        gripper_poses = f["observation/robot_state/gripper_position"][:]
        wrist_ext = json.load(jf)["wrist_cam_extrinsics"]
        wrist_ext = (
            wrist_ext.get('extrinsics', wrist_ext)
            if isinstance(wrist_ext, dict) else wrist_ext
        )

    total_frames = len(ee_poses)
    T_ee_all = np.tile(np.eye(4), (total_frames, 1, 1))
    T_ee_all[:, :3, :3] = R.from_euler('xyz', ee_poses[:, 3:]).as_matrix()
    T_ee_all[:, :3, 3] = ee_poses[:, :3]

    scene_constants['robot'] = {
        'joint_positions': joint_poses,
        'gripper_positions': gripper_poses,
        'T_cam_ee_init': np.linalg.inv(make_4x4(ee_poses[0])) @ make_4x4(wrist_ext),
        'T_ee_base_all': T_ee_all,
    }
    return scene_constants


# ==========================================
# Temporal Operations
# ==========================================

def align_temporal_streams(scene_constants):
    """Truncate all temporal streams to the shortest length."""
    print("  ⏱️ Aligning temporal streams...")
    lengths = [
        len(scene_constants['robot']['joint_positions']),
        len(scene_constants['robot']['gripper_positions']),
        len(scene_constants['robot']['T_ee_base_all']),
    ]
    for cam_data in scene_constants['camera'].values():
        lengths.append(len(cam_data['video_rgb']))
        lengths.append(len(cam_data['video_right']))

    min_frames = min(lengths)
    if min_frames == max(lengths):
        print(f"    ✅ Already aligned at {min_frames} frames.")
        return scene_constants

    print(f"    ✂️ Truncating to {min_frames} frames (was {max(lengths)}).")
    for key in ['joint_positions', 'gripper_positions', 'T_ee_base_all']:
        scene_constants['robot'][key] = scene_constants['robot'][key][:min_frames]
    for cam_data in scene_constants['camera'].values():
        cam_data['video_rgb'] = cam_data['video_rgb'][:min_frames]
        cam_data['video_right'] = cam_data['video_right'][:min_frames]

    return scene_constants


def filter_idle_frames(scene_constants, scene_state=None):
    """Remove idle frames from both scene_constants AND scene_state.

    Fixed bug: previously only filtered scene_constants in the batch cell,
    leaving scene_state extrinsics misaligned.
    """
    print("  ✂️ Filtering idle frames...")
    valid_indices = scene_constants['meta'].get('valid_indices')
    if valid_indices is None:
        print("    ⚠️ No valid_indices found, skipping.")
        return scene_constants, scene_state

    n_original = len(scene_constants['robot']['joint_positions'])
    valid_indices = valid_indices[valid_indices < n_original]

    if len(valid_indices) == 0:
        print("    ❌ No valid frames after filtering!")
        return scene_constants, scene_state

    print(f"    Keeping {len(valid_indices)} / {n_original} action frames.")

    # Filter robot data
    for key in ['joint_positions', 'gripper_positions', 'T_ee_base_all']:
        if key in scene_constants['robot']:
            scene_constants['robot'][key] = scene_constants['robot'][key][valid_indices]

    # Filter camera data (auto-detect temporal arrays)
    for cam_data in scene_constants['camera'].values():
        for key, value in cam_data.items():
            if isinstance(value, np.ndarray) and len(value) == n_original:
                cam_data[key] = value[valid_indices]
            elif isinstance(value, list) and len(value) == n_original:
                cam_data[key] = [value[i] for i in valid_indices]

    # Filter scene_state extrinsics
    if scene_state is not None:
        for state_data in scene_state.values():
            if 'extrinsics' in state_data and state_data['extrinsics'] is not None:
                if len(state_data['extrinsics']) == n_original:
                    state_data['extrinsics'] = state_data['extrinsics'][valid_indices]

    return scene_constants, scene_state


def extract_last_n_frames(scene_constants, scene_state, n=48):
    """Extract the last N frames from all temporal streams."""
    print(f"  ✂️ Extracting last {n} frames...")
    new_constants = copy.deepcopy(scene_constants)
    new_state = copy.deepcopy(scene_state)

    current_frames = len(new_constants['robot']['joint_positions'])
    if current_frames <= n:
        print(f"    ✅ Already {current_frames} frames (<= {n}), no slicing needed.")
        return new_constants, new_state

    print(f"    Slicing: {current_frames} -> {n}")

    # Robot
    for key, value in new_constants['robot'].items():
        if isinstance(value, (list, np.ndarray)) and len(value) == current_frames:
            new_constants['robot'][key] = value[-n:]

    # Camera
    for cam_data in new_constants['camera'].values():
        for key, value in list(cam_data.items()):
            if isinstance(value, (list, np.ndarray)) and len(value) == current_frames:
                cam_data[key] = value[-n:]
        cam_data.pop('tracks_2d', None)
        cam_data.pop('vis_2d', None)

    # State
    if new_state is not None:
        for state_data in new_state.values():
            if 'extrinsics' in state_data and state_data['extrinsics'] is not None:
                if len(state_data['extrinsics']) == current_frames:
                    state_data['extrinsics'] = state_data['extrinsics'][-n:]

    return new_constants, new_state
