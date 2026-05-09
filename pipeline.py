"""DROID Dataset Processing Pipeline.

Converts raw DROID episodes (SVO stereo video + robot kinematics) into
TAPVid-format 3D tracking ground truth for multi-view evaluation.

Usage:
  # Step 1: Sequential on GPU 0, process 2 episodes
  python pipeline.py --gpu 0 --start 0 --count 2

  # Step 2: Parallel on 2 GPUs
  python pipeline.py --gpu 0 --start 0 --count 1 &
  python pipeline.py --gpu 1 --start 1 --count 1 &

  # Step 3: 10 episodes split across 2 GPUs
  python pipeline.py --gpu 0 --start 0 --count 5 &
  python pipeline.py --gpu 1 --start 5 --count 5 &
"""

import argparse
import copy
import gc
import glob
import importlib.util
import inspect
import json
import os
import sys
import traceback
import warnings

import cv2
import h5py
import numpy as np
import pybullet as p
import pybullet_data
import pyzed.sl as sl
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ==========================================
# 1. Environment & Device Configuration
# ==========================================
os.environ['PYOPENGL_PLATFORM'] = 'egl'

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY_DIR = os.path.join(os.path.dirname(REPO_DIR), "third_party")

# Inject third-party repo paths (pre-cloned on VM)
sys.path.append(os.path.join(THIRD_PARTY_DIR, "s2m2/src"))
sys.path.append(os.path.join(THIRD_PARTY_DIR, "vggt"))
sys.path.append(os.path.join(THIRD_PARTY_DIR, "co-tracker"))

from cotracker.predictor import CoTrackerPredictor
from s2m2.core.utils.model_utils import load_model, run_stereo_matching
from segment_anything import SamPredictor, sam_model_registry
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ==========================================
# 2. 3D Geometry Utility Functions
# ==========================================

def decode_disparity_np(disp, fx, baseline):
    """Convert raw disparity to physical depth (NumPy)."""
    z = np.zeros_like(disp)
    valid_mask = disp > 0
    z[valid_mask] = (fx * baseline) / disp[valid_mask]
    return z


def unproject_points_np(u, v, z, K, T_cam2world=None):
    """3D ray back-projection (NumPy)."""
    x_cam = (u - K[0, 2]) * z / K[0, 0]
    y_cam = (v - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=0)
    if T_cam2world is None:
        return pts_cam[:3, :].T
    return (T_cam2world @ pts_cam)[:3, :].T


def project_points_np(pts_world, K, T_cam2world):
    """Project 3D world points back to 2D pixel plane (NumPy)."""
    T_world2cam = np.linalg.inv(T_cam2world)
    pts_homo = np.hstack([pts_world, np.ones((len(pts_world), 1))]).T
    pts_cam = T_world2cam @ pts_homo
    z_cam = pts_cam[2, :]
    u = np.zeros_like(pts_cam[0, :])
    v = np.zeros_like(pts_cam[1, :])
    valid_mask = z_cam > 0
    u[valid_mask] = (pts_cam[0, valid_mask] / z_cam[valid_mask]) * K[0, 0] + K[0, 2]
    v[valid_mask] = (pts_cam[1, valid_mask] / z_cam[valid_mask]) * K[1, 1] + K[1, 2]
    return u, v, z_cam


def unproject_to_3d(depth, color_img, K_mat, T_cam2world=None, min_depth=0., max_depth=1.5):
    """Unproject depth map to 3D point cloud with spatial truncation."""
    mask = (depth > min_depth) & (depth < max_depth)
    v, u = np.where(mask)
    if T_cam2world is None:
        T_cam2world = np.eye(4)
    pts_world = unproject_points_np(u, v, depth[mask], K_mat, T_cam2world)
    return pts_world, color_img[mask]


def make_4x4(vec_6d):
    """Convert 6DoF vector [x, y, z, rx, ry, rz] to 4x4 homogeneous matrix."""
    transform = np.eye(4)
    transform[:3, :3] = R.from_euler('xyz', vec_6d[3:]).as_matrix()
    transform[:3, 3] = vec_6d[:3]
    return transform


def axis_angle_to_matrix(rot_vec):
    """Differentiable Rodrigues rotation formula (PyTorch)."""
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


# ==========================================
# 3. Model Loading
# ==========================================

def init_all_models(device):
    """Load all ML models onto the specified GPU device."""
    print(f"🚀 Loading models onto {device}...")

    # S2M2 stereo depth
    s2m2_model = load_model(
        os.path.join(WORKSPACE_DIR, "s2m2/weights/pretrain_weights"), "XL", True, 3, device
    )
    s2m2_model.eval()
    s2m2_model = torch.compile(s2m2_model)
    print("  ✅ S2M2 loaded")

    # CoTracker3 dense tracking
    cotracker_model = CoTrackerPredictor(
        checkpoint=os.path.join(WORKSPACE_DIR, "co-tracker/weights/cotracker3_offline.pth")
    ).to(device)
    print("  ✅ CoTracker3 loaded")

    # VGGT-1B visual extrinsics
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    vggt_model.eval()
    print("  ✅ VGGT-1B loaded")

    # SAM ViT-H segmentation
    sam = sam_model_registry["vit_h"](
        checkpoint=os.path.join(WORKSPACE_DIR, "sam_weights/sam_vit_h_4b8939.pth")
    ).to(device)
    sam_predictor = SamPredictor(sam)
    print("  ✅ SAM loaded")

    return s2m2_model, cotracker_model, vggt_model, sam_predictor


# ==========================================
# 4. Depth Extraction
# ==========================================

@torch.inference_mode()
def get_s2m2_disparity(s2m2_model, img_left, img_right, device, conf_thresh=0.95):
    """Pure disparity extractor using S2M2 stereo network."""
    left_torch = torch.from_numpy(img_left).permute(2, 0, 1).unsqueeze(0).to(device)
    right_torch = torch.from_numpy(img_right).permute(2, 0, 1).unsqueeze(0).to(device)
    pred_disp, _, pred_conf, _, _ = run_stereo_matching(
        s2m2_model, left_torch, right_torch, device, N_repeat=3
    )
    disp = pred_disp.cpu().numpy().squeeze()
    conf = pred_conf.cpu().numpy().squeeze()
    valid_mask = (disp > 0) & (conf >= conf_thresh)
    disp[~valid_mask] = 0.0
    return disp


def compute_stereo_depth(scene_constants, s2m2_model, device):
    """Run S2M2 stereo depth on all cameras."""
    print("  🧠 Running S2M2 stereo depth inference...")
    for cam_id in scene_constants['camera']:
        cam_data = scene_constants['camera'][cam_id]
        left_seq, right_seq = cam_data['video_rgb'], cam_data['video_right']
        disp_frames = [
            get_s2m2_disparity(s2m2_model, left_img, right_img, device=device)
            for left_img, right_img in tqdm(
                zip(left_seq, right_seq), total=len(left_seq), desc=f"Depth [{cam_id}]"
            )
        ]
        raw_disp = np.stack(disp_frames)
        fx = cam_data['K_mat'][0, 0]
        baseline = cam_data['baseline']
        cam_data['raw_depth'] = decode_disparity_np(raw_disp, fx, baseline)
    return scene_constants


# ==========================================
# 5. Data Loading & Episode Management
# ==========================================

def load_metadata(data_root):
    """Load DROID dataset metadata JSONs."""
    meta_dir = os.path.expanduser("~/droid_data/meta/1.0.1")
    os.makedirs(meta_dir, exist_ok=True)
    base_url = "https://huggingface.co/KarlP/droid/resolve/main"
    files = [
        "intrinsics.json",
        "camera_serials.json",
        "episode_id_to_path.json",
        "keep_ranges_1_0_1.json",
        "cam2base_extrinsic_superset.json"
    ]
    print(f"⬇️ Checking and downloading metadata JSON files to {meta_dir}...")
    for f in files:
        if not os.path.exists(os.path.join(meta_dir, f)):
            os.system(f"wget -q -nc -P {meta_dir} {base_url}/{f}")

    def load_json(name):
        with open(os.path.join(meta_dir, name), 'r') as f:
            return json.load(f)

    serials_db = load_json("camera_serials.json")
    id_to_path = load_json("episode_id_to_path.json")
    keep_ranges = load_json("keep_ranges_1_0_1.json")
    extrinsics_db = load_json("cam2base_extrinsic_superset.json")

    # Only episodes with all metadata AND pre-calibrated extrinsics
    valid_ids = sorted(
        set(serials_db.keys()) & set(id_to_path.keys()) & set(extrinsics_db.keys())
    )
    print(f"✅ Metadata loaded. {len(valid_ids)} episodes with extrinsics.")
    return serials_db, id_to_path, keep_ranges, extrinsics_db, valid_ids


def download_episode(episode_id, data_root, id_to_path, serials_db, keep_ranges_db):
    """Download episode and initialize scene_constants structure."""
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


def extract_svo_video(scene_constants):
    """Decode SVO stereo video files and extract calibration data."""
    print("  🎥 Decoding SVO video streams...")
    episode_path = scene_constants['meta']['episode_path']

    for cam in scene_constants['camera']:
        svo_path = glob.glob(os.path.join(episode_path, f"**/{cam}.svo"), recursive=True)[0]

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

        for _ in tqdm(range(zed.get_svo_number_of_frames()), desc=f"Decoding {cam}"):
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
        wrist_ext = wrist_ext.get('extrinsics', wrist_ext) if isinstance(wrist_ext, dict) else wrist_ext

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
    scene_constants['robot']['joint_positions'] = scene_constants['robot']['joint_positions'][:min_frames]
    scene_constants['robot']['gripper_positions'] = scene_constants['robot']['gripper_positions'][:min_frames]
    scene_constants['robot']['T_ee_base_all'] = scene_constants['robot']['T_ee_base_all'][:min_frames]

    for cam_data in scene_constants['camera'].values():
        cam_data['video_rgb'] = cam_data['video_rgb'][:min_frames]
        cam_data['video_right'] = cam_data['video_right'][:min_frames]

    return scene_constants


def filter_idle_frames(scene_constants, scene_state=None):
    """Remove idle frames from both scene_constants AND scene_state."""
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
        return new_constants, new_state

    # Robot
    for key, value in new_constants['robot'].items():
        if isinstance(value, (list, np.ndarray)) and len(value) == current_frames:
            new_constants['robot'][key] = value[-n:]

    # Camera
    for cam_data in new_constants['camera'].values():
        for key, value in list(cam_data.items()):
            if isinstance(value, (list, np.ndarray)) and len(value) == current_frames:
                cam_data[key] = value[-n:] if isinstance(value, np.ndarray) else value[-n:]
        cam_data.pop('tracks_2d', None)
        cam_data.pop('vis_2d', None)

    # State
    if new_state is not None:
        for state_data in new_state.values():
            if 'extrinsics' in state_data and state_data['extrinsics'] is not None:
                if len(state_data['extrinsics']) == current_frames:
                    state_data['extrinsics'] = state_data['extrinsics'][-n:]

    return new_constants, new_state


# ==========================================
# 6. Camera Calibration
# ==========================================

def init_camera_states(scene_constants, extrinsics_db):
    """Initialize multi-camera extrinsic states from metadata."""
    print("  🌐 Initializing camera extrinsic states...")
    wrist_serial = scene_constants['meta']['wrist_serial']
    robot_data = scene_constants['robot']
    n_frames = len(robot_data['T_ee_base_all'])
    episode_id = scene_constants['meta']['episode_id']
    episode_extrinsics = extrinsics_db.get(episode_id, {})

    scene_state = {}
    for cam_id in scene_constants['camera'].keys():
        if cam_id == wrist_serial:
            base_ext = robot_data['T_cam_ee_init']
            cam_trajectory = robot_data['T_ee_base_all'] @ base_ext
        elif cam_id in episode_extrinsics:
            ext_data = episode_extrinsics[cam_id]
            ext_vec = ext_data.get('extrinsics', ext_data) if isinstance(ext_data, dict) else ext_data
            base_ext = make_4x4(ext_vec)
            cam_trajectory = np.tile(base_ext, (n_frames, 1, 1))
        else:
            print(f"    ⚠️ No extrinsics for camera [{cam_id}], setting to None.")
            base_ext = None
            cam_trajectory = None

        scene_state[cam_id] = {
            'base_extrinsic': base_ext,
            'extrinsics': cam_trajectory,
        }
    return scene_state


def estimate_multi_camera_vggt(vggt_model, img_list, device):
    """Use VGGT to estimate relative camera poses from images."""
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    filenames = []
    for i, img in enumerate(img_list):
        fname = f"/tmp/tmp_vggt_{i}.png"
        cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        filenames.append(fname)

    images = load_and_preprocess_images(filenames).to(device)
    images_input = images.unsqueeze(0)

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=dtype):
            aggregated_tokens_list, ps_idx = vggt_model.aggregator(images_input)
            pose_enc = vggt_model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_input.shape[-2:])

    T_ref_to_tgts = []
    for i in range(1, len(img_list)):
        ext_mat = extrinsic[0, i].cpu().numpy()
        T = np.eye(4)
        T[:3, :] = ext_mat
        T_ref_to_tgts.append(T)

    # Cleanup temp files
    for fname in filenames:
        if os.path.exists(fname):
            os.remove(fname)

    return T_ref_to_tgts


def vggt_warmup_extrinsics(scene_constants, vggt_model, device):
    """Use VGGT to estimate camera extrinsics from first frame."""
    print("  🌍 Running VGGT visual extrinsic estimation...")
    wrist_serial = scene_constants['meta']['wrist_serial']
    ext_cams = [cam for cam in scene_constants['camera'].keys() if cam != wrist_serial]
    ref_cam = ext_cams[0]
    other_cams = ext_cams[1:]

    new_scene_state = {}
    robot_data = scene_constants['robot']
    n_frames_total = len(scene_constants['camera'][ref_cam]['video_rgb'])

    new_scene_state[wrist_serial] = {
        'base_extrinsic': robot_data['T_cam_ee_init'],
        'extrinsics': robot_data['T_ee_base_all'] @ robot_data['T_cam_ee_init'],
    }

    img_ref = scene_constants['camera'][ref_cam]['video_rgb'][0]
    img_others = [scene_constants['camera'][cam]['video_rgb'][0] for cam in other_cams]
    img_wrist = scene_constants['camera'][wrist_serial]['video_rgb'][0]
    img_list = [img_ref] + img_others + [img_wrist]

    T_rel_list = estimate_multi_camera_vggt(vggt_model, img_list, device)
    T_ref_to_others = T_rel_list[:-1]
    T_ref_to_wrist = T_rel_list[-1]

    T_wrist_to_base_first = scene_constants['robot']['T_ee_base_all'][0] @ scene_constants['robot']['T_cam_ee_init']
    T_ref_to_base = T_wrist_to_base_first @ T_ref_to_wrist

    new_scene_state[ref_cam] = {
        'base_extrinsic': T_ref_to_base,
        'extrinsics': np.tile(T_ref_to_base, (n_frames_total, 1, 1)),
    }

    for tgt_cam, T_ref_to_tgt in zip(other_cams, T_ref_to_others):
        T_tgt_to_base = T_ref_to_base @ np.linalg.inv(T_ref_to_tgt)
        new_scene_state[tgt_cam] = {
            'base_extrinsic': T_tgt_to_base,
            'extrinsics': np.tile(T_tgt_to_base, (n_frames_total, 1, 1)),
        }

    return new_scene_state


# ==========================================
# 7. PyBullet Physics Renderer
# ==========================================

class PyBulletRenderer_Robotiq:
    """Dual-body physics renderer: Franka arm + Robotiq 2F-85 gripper."""

    def __init__(self, ghost_urdf=None):
        if ghost_urdf is None:
            ghost_urdf = os.path.join(
                WORKSPACE_DIR, "PointWorld/assets/franka_description/franka_panda_robotiq_2f85_og.urdf"
            )

        if p.isConnected():
            p.disconnect()
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        if importlib.util.find_spec('eglRendererPlugin'):
            p.loadPlugin(importlib.util.find_spec('eglRendererPlugin').origin, "_eglRendererPlugin")

        # Real body (thin arm)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.arm_joints = [
            i for i in range(p.getNumJoints(self.robot_id))
            if "panda_joint" in p.getJointInfo(self.robot_id, i)[1].decode('utf-8')
            and p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED
        ]

        self.hidden_robot_links = []
        for i in range(-1, p.getNumJoints(self.robot_id)):
            name = (p.getBodyInfo(self.robot_id)[0].decode('utf-8') if i == -1
                    else p.getJointInfo(self.robot_id, i)[12].decode('utf-8'))
            if "hand" in name or "finger" in name:
                p.changeVisualShape(self.robot_id, i, rgbaColor=[0, 0, 0, 0])
                self.hidden_robot_links.append(i)

        # Ghost body (with gripper)
        self.ghost_id = p.loadURDF(ghost_urdf, useFixedBase=True)
        self.ghost_arm_joints = [
            i for i in range(p.getNumJoints(self.ghost_id))
            if "panda_joint" in p.getJointInfo(self.ghost_id, i)[1].decode('utf-8')
            and p.getJointInfo(self.ghost_id, i)[2] != p.JOINT_FIXED
        ]

        self.gripper_joints = []
        self.gripper_signs = []
        for i in range(p.getNumJoints(self.ghost_id)):
            info = p.getJointInfo(self.ghost_id, i)
            joint_name = info[1].decode('utf-8')
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
            name = (p.getBodyInfo(self.ghost_id)[0].decode('utf-8') if i == -1
                    else p.getJointInfo(self.ghost_id, i)[12].decode('utf-8'))
            if "panda_link" in name:
                p.changeVisualShape(self.ghost_id, i, rgbaColor=[0, 0, 0, 0])
                self.hidden_ghost_links.append(i)

    def _get_projection_matrix(self, intrinsics, width, height):
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        near, far = 0.01, 10.0
        return [
            2.0 * fx / width, 0.0, 0.0, 0.0,
            0.0, 2.0 * fy / height, 0.0, 0.0,
            1.0 - 2.0 * cx / width, 2.0 * cy / height - 1.0, (far + near) / (near - far), -1.0,
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
        cam_pos, target_pos = extrinsics[:3, 3], extrinsics[:3, 3] + extrinsics[:3, 2]
        view_matrix = p.computeViewMatrix(cam_pos, target_pos, -extrinsics[:3, 1])
        proj_matrix = self._get_projection_matrix(intrinsics, width, height)
        _, _, _, depth_buffer, _ = p.getCameraImage(
            width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        metric_depth = 0.1 / (10.0 - 9.99 * np.reshape(depth_buffer, (height, width)))
        return np.where(metric_depth < 9.9, metric_depth, 0.0)

    def render_mask(self, extrinsics, intrinsics, width, height):
        cam_pos, target_pos = extrinsics[:3, 3], extrinsics[:3, 3] + extrinsics[:3, 2]
        view_matrix = p.computeViewMatrix(cam_pos, target_pos, -extrinsics[:3, 1])
        proj_matrix = self._get_projection_matrix(intrinsics, width, height)
        _, _, _, _, seg_buffer = p.getCameraImage(
            width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        )
        seg_array = np.reshape(seg_buffer, (height, width)).astype(np.int32)
        obj_ids, link_ids = seg_array & 0xFFFFFF, (seg_array >> 24) - 1
        valid_robot = (obj_ids == self.robot_id) & ~np.isin(link_ids, self.hidden_robot_links)
        valid_ghost = (obj_ids == self.ghost_id) & ~np.isin(link_ids, self.hidden_ghost_links)
        return valid_robot | valid_ghost


# ==========================================
# 8. Camera Calibration Optimization
# ==========================================

def get_foreground_robot_points(T_init, K, obs_depth, pb_renderer, max_pts, device):
    """Extract robot surface point cloud from rendered depth."""
    h_img, w_img = obs_depth.shape
    render_d = pb_renderer.render_depth(T_init, K, w_img, h_img)
    v_r, u_r = np.where(render_d > 0)
    z_r = render_d[v_r, u_r]
    if len(z_r) < max_pts:
        return None
    P_cam_r = np.stack([
        (u_r - K[0, 2]) * z_r / K[0, 0],
        (v_r - K[1, 2]) * z_r / K[1, 1],
        z_r, np.ones_like(z_r)
    ])
    pts_robot_world = (T_init @ P_cam_r)[:3, :].T
    idx = np.random.choice(len(pts_robot_world), max_pts, replace=(len(pts_robot_world) < max_pts))
    return torch.tensor(pts_robot_world[idx], dtype=torch.float32, device=device)


def get_foreground_gripper_points(T_cam_world, K, obs_depth, pb_renderer, max_pts):
    """Extract only gripper surface points (not arm links)."""
    h_img, w_img = obs_depth.shape
    cam_pos, target_pos = T_cam_world[:3, 3], T_cam_world[:3, 3] + T_cam_world[:3, 2]
    view_matrix = p.computeViewMatrix(cam_pos, target_pos, -T_cam_world[:3, 1])
    proj_matrix = pb_renderer._get_projection_matrix(K, w_img, h_img)
    _, _, _, depth_buffer, seg_buffer = p.getCameraImage(
        w_img, h_img, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    )
    metric_depth = 0.1 / (10.0 - 9.99 * np.reshape(depth_buffer, (h_img, w_img)))
    seg_array = np.reshape(seg_buffer, (h_img, w_img)).astype(np.int32)
    obj_ids = seg_array & 0xFFFFFF
    valid_ghost = (obj_ids == pb_renderer.ghost_id)
    v_r, u_r = np.where((metric_depth < 9.9) & valid_ghost)
    z_r = metric_depth[v_r, u_r]
    if len(z_r) < 100:
        return None
    P_cam_r = np.stack([
        (u_r - K[0, 2]) * z_r / K[0, 0],
        (v_r - K[1, 2]) * z_r / K[1, 1],
        z_r, np.ones_like(z_r)
    ])
    idx = np.random.choice(P_cam_r.shape[1], max_pts, replace=(P_cam_r.shape[1] < max_pts))
    return P_cam_r[:, idx]


def compute_robot_loss_batched(batch_X, T_opt, K, batch_obs, device):
    """GPU batched depth alignment loss for robot points."""
    B, _, h_img, w_img = batch_obs.shape
    P_c = (batch_X - T_opt[:3, 3]) @ T_opt[:3, :3]
    Z_pred = P_c[..., 2]
    u = K[0, 0] * P_c[..., 0] / Z_pred + K[0, 2]
    v = K[1, 1] * P_c[..., 1] / Z_pred + K[1, 2]
    grid = torch.stack([(u / (w_img - 1)) * 2 - 1, (v / (h_img - 1)) * 2 - 1], dim=-1).unsqueeze(1)
    Z_obs_raw = F.grid_sample(batch_obs, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(1).squeeze(1)
    valid_mask = (
        (Z_pred > 0.) & (Z_pred < 1.5) & (Z_obs_raw > 0.) & (Z_obs_raw < 1.5) &
        (u >= 0) & (u < w_img - 1) & (v >= 0) & (v < h_img - 1)
    )
    diff = torch.abs(Z_obs_raw[valid_mask] - Z_pred[valid_mask])
    return torch.nan_to_num(diff.mean(), nan=0.0)


def compute_wrist_loss_batched(batch_P_ee, T_cam_ee_opt, K, batch_obs, device):
    """GPU batched depth alignment loss for wrist gripper points."""
    B, _, h_img, w_img = batch_obs.shape
    T_ee_cam = torch.linalg.inv(T_cam_ee_opt)
    P_c = batch_P_ee @ T_ee_cam[:3, :3].T + T_ee_cam[:3, 3]
    Z_pred = P_c[..., 2]
    u = K[0, 0] * P_c[..., 0] / Z_pred + K[0, 2]
    v = K[1, 1] * P_c[..., 1] / Z_pred + K[1, 2]
    grid = torch.stack([(u / (w_img - 1)) * 2 - 1, (v / (h_img - 1)) * 2 - 1], dim=-1).unsqueeze(1)
    Z_obs_raw = F.grid_sample(batch_obs, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(1).squeeze(1)
    valid_mask = (
        (Z_pred > 0.) & (Z_pred < 1.5) & (Z_obs_raw > 0.) & (Z_obs_raw < 1.5) &
        (u >= 0) & (u < w_img - 1) & (v >= 0) & (v < h_img - 1)
    )
    diff = torch.abs(Z_obs_raw[valid_mask] - Z_pred[valid_mask])
    return torch.nan_to_num(diff.mean(), nan=0.0)


def run_stage2_robot_alignment(scene_constants, init_scene_state, pb_renderer, device):
    """Stage 2: Align external cameras to robot body via depth matching."""
    OUTER_LOOPS, INNER_LOOPS, MAX_ROBOT_PTS = 5, 100, 2000
    print("  🦾 Stage 2: External camera-robot alignment...")

    ext_cams = [c for c in scene_constants['camera'].keys() if c != scene_constants['meta']['wrist_serial']]
    n_frames = len(scene_constants['camera'][ext_cams[0]]['video_rgb'])

    def make_T(delta):
        T = torch.eye(4, device=device)
        T[:3, :3], T[:3, 3] = axis_angle_to_matrix(delta[:3]), delta[3:]
        return T

    pybullet_scene_state = copy.deepcopy(init_scene_state)

    for cam in ext_cams:
        print(f"    📷 Optimizing [{cam}]...")
        T_init_t = torch.tensor(init_scene_state[cam]['base_extrinsic'], dtype=torch.float32, device=device)
        K_t = torch.tensor(scene_constants['camera'][cam]['K_mat'], dtype=torch.float32, device=device)
        K_np = scene_constants['camera'][cam]['K_mat']
        d_ext = torch.zeros(6, requires_grad=True, device=device)
        optimizer = optim.Adam([d_ext], lr=0.001)

        for outer_step in range(OUTER_LOOPS):
            with torch.no_grad():
                T_cur_np = (T_init_t @ make_T(d_ext)).cpu().numpy()

            cache_X, cache_obs = [], []
            for t in range(n_frames):
                pb_renderer.update_robot_pose(scene_constants['robot']['joint_positions'][t])
                d_obs = scene_constants['camera'][cam]['raw_depth'][t].astype(np.float32)
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
                loss = compute_robot_loss_batched(batch_X, T_init_t @ make_T(d_ext), K_t, batch_obs, device)
                loss.backward()
                optimizer.step()

            print(f"      Outer {outer_step+1}/{OUTER_LOOPS} | Loss: {loss.item():.4f}")

        with torch.no_grad():
            T_final_np = (T_init_t @ make_T(d_ext)).cpu().numpy()
            pybullet_scene_state[cam]['base_extrinsic'] = T_final_np
            pybullet_scene_state[cam]['extrinsics'] = np.tile(T_final_np, (n_frames, 1, 1))

    return pybullet_scene_state


def run_stage3_joint_alignment(scene_constants, stage2_scene_state, pb_renderer, device):
    """Stage 3: Joint environment stitching + robot + wrist alignment."""
    print("  🌍 Stage 3: Joint unified alignment...")

    camera_ids = list(scene_constants['camera'].keys())
    wrist_cam = scene_constants['meta']['wrist_serial']
    ext_cams = [c for c in camera_ids if c != wrist_cam]
    cam1, cam2 = ext_cams[0], ext_cams[1]
    n_frames = len(scene_constants['camera'][cam1]['video_rgb'])

    def to_t(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    def make_T(delta):
        T = torch.eye(4, device=device)
        T[:3, :3], T[:3, 3] = axis_angle_to_matrix(delta[:3]), delta[3:]
        return T

    T_ee_all = scene_constants['robot']['T_ee_base_all']
    T_cam_ee_init = stage2_scene_state[wrist_cam]['base_extrinsic']
    init_p1 = stage2_scene_state[cam1]['base_extrinsic']
    init_p2 = stage2_scene_state[cam2]['base_extrinsic']

    K_np1 = scene_constants['camera'][cam1]['K_mat']
    K_np2 = scene_constants['camera'][cam2]['K_mat']
    K_np_w = scene_constants['camera'][wrist_cam]['K_mat']
    K_t1, K_t2, K_t_w = to_t(K_np1), to_t(K_np2), to_t(K_np_w)

    # Precompute point clouds
    def get_cam_points_t(t, cam_data):
        depth = cam_data['raw_depth'][t].astype(np.float32)
        K_mat_np = cam_data['K_mat']
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
        n_sample = min(P_cam.shape[1], 2000)
        idx = np.random.choice(P_cam.shape[1], n_sample, replace=(P_cam.shape[1] < n_sample))
        return torch.tensor(P_cam[:, idx], dtype=torch.float32, device=device)

    cache_P1, cache_P2, cache_Pw, cache_Tee = [], [], [], []
    cache_X1, cache_obs1, cache_X2, cache_obs2 = [], [], [], []
    cache_P_ee, cache_obs_w = [], []

    for t in range(n_frames):
        p1 = get_cam_points_t(t, scene_constants['camera'][cam1])
        p2 = get_cam_points_t(t, scene_constants['camera'][cam2])
        pw = get_cam_points_t(t, scene_constants['camera'][wrist_cam])
        if p1 is not None and p2 is not None and pw is not None:
            cache_P1.append(p1)
            cache_P2.append(p2)
            cache_Pw.append(pw)
            cache_Tee.append(to_t(T_ee_all[t]))

        pb_renderer.update_robot_pose(
            scene_constants['robot']['joint_positions'][t],
            gripper_state=scene_constants['robot']['gripper_positions'][t]
        )

        d_obs1 = scene_constants['camera'][cam1]['raw_depth'][t].astype(np.float32)
        r_pts1 = get_foreground_robot_points(init_p1, K_np1, d_obs1, pb_renderer, 2000, device)
        if r_pts1 is not None:
            cache_X1.append(r_pts1)
            cache_obs1.append(torch.tensor(d_obs1, dtype=torch.float32, device=device)[None, ...])

        d_obs2 = scene_constants['camera'][cam2]['raw_depth'][t].astype(np.float32)
        r_pts2 = get_foreground_robot_points(init_p2, K_np2, d_obs2, pb_renderer, 2000, device)
        if r_pts2 is not None:
            cache_X2.append(r_pts2)
            cache_obs2.append(torch.tensor(d_obs2, dtype=torch.float32, device=device)[None, ...])

        T_cam_world_np = T_ee_all[t] @ T_cam_ee_init
        d_obs_w = scene_constants['camera'][wrist_cam]['raw_depth'][t].astype(np.float32)
        P_cam_r = get_foreground_gripper_points(T_cam_world_np, K_np_w, d_obs_w, pb_renderer, 2000)
        if P_cam_r is not None:
            P_ee_r = (T_cam_ee_init @ P_cam_r)[:3, :].T
            cache_P_ee.append(torch.tensor(P_ee_r, dtype=torch.float32, device=device))
            cache_obs_w.append(torch.tensor(d_obs_w, dtype=torch.float32, device=device)[None, ...])

    batch_P1 = torch.stack(cache_P1)
    batch_P2 = torch.stack(cache_P2)
    batch_Pw = torch.stack(cache_Pw)
    batch_Tee = torch.stack(cache_Tee)
    batch_X1 = torch.stack(cache_X1)
    batch_obs1 = torch.stack(cache_obs1)
    batch_X2 = torch.stack(cache_X2)
    batch_obs2 = torch.stack(cache_obs2)
    batch_P_ee = torch.stack(cache_P_ee) if cache_P_ee else None
    batch_obs_w = torch.stack(cache_obs_w) if cache_obs_w else None

    d1 = torch.zeros(6, requires_grad=True, device=device)
    d2 = torch.zeros(6, requires_grad=True, device=device)
    dhe = torch.zeros(6, requires_grad=True, device=device)
    optimizer = optim.Adam([d1, d2, dhe], lr=0.001)

    T1_init_t, T2_init_t, Tee_init_t = to_t(init_p1), to_t(init_p2), to_t(T_cam_ee_init)

    def batched_chamfer_distance(p1, p2):
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
        return loss

    for step in range(500):
        optimizer.zero_grad()

        bc1 = (T1_init_t @ make_T(d1) @ batch_P1)[:, :3, :].transpose(1, 2)
        bc2 = (T2_init_t @ make_T(d2) @ batch_P2)[:, :3, :].transpose(1, 2)
        T_wrist_world = batch_Tee @ (Tee_init_t @ make_T(dhe))
        bcw = torch.bmm(T_wrist_world, batch_Pw)[:, :3, :].transpose(1, 2)

        l12 = batched_chamfer_distance(bc1, bc2)
        l1w = batched_chamfer_distance(bc1, bcw)
        l2w = batched_chamfer_distance(bc2, bcw)
        loss_chamfer = l12 + l1w + l2w

        l_rob1 = compute_robot_loss_batched(batch_X1, T1_init_t @ make_T(d1), K_t1, batch_obs1, device)
        l_rob2 = compute_robot_loss_batched(batch_X2, T2_init_t @ make_T(d2), K_t2, batch_obs2, device)

        l_wrist = torch.tensor(0.0, device=device)
        if batch_P_ee is not None:
            l_wrist = compute_wrist_loss_batched(batch_P_ee, Tee_init_t @ make_T(dhe), K_t_w, batch_obs_w, device)

        loss_total = loss_chamfer + 1.0 * (l_rob1 + l_rob2 + l_wrist)
        loss_total.backward()
        optimizer.step()

        if step % 100 == 0 or step == 499:
            print(f"    Step {step:03d} | Chamfer: {loss_chamfer.item():.4f} | "
                  f"Rob1: {l_rob1.item():.4f} | Rob2: {l_rob2.item():.4f} | Wrist: {l_wrist.item():.4f}")

    with torch.no_grad():
        final_p1 = (T1_init_t @ make_T(d1)).cpu().numpy()
        final_p2 = (T2_init_t @ make_T(d2)).cpu().numpy()
        final_cam_ee = (Tee_init_t @ make_T(dhe)).cpu().numpy()

    ultimate_scene_state = {cam: {} for cam in camera_ids}
    ultimate_scene_state[cam1] = {'base_extrinsic': final_p1, 'extrinsics': np.tile(final_p1, (n_frames, 1, 1))}
    ultimate_scene_state[cam2] = {'base_extrinsic': final_p2, 'extrinsics': np.tile(final_p2, (n_frames, 1, 1))}
    ultimate_scene_state[wrist_cam] = {'base_extrinsic': final_cam_ee, 'extrinsics': T_ee_all @ final_cam_ee}

    return ultimate_scene_state


# ==========================================
# 9. 2D Tracking
# ==========================================

def extract_2d_tracks(cotracker_model, scene_constants, device):
    """Run CoTracker3 dense 2D tracking on all camera views."""
    print("  🎯 Running CoTracker3 dense tracking...")
    for cam_id in scene_constants['camera']:
        cam_data = scene_constants['camera'][cam_id]
        video_tensor = torch.from_numpy(cam_data['video_rgb']).permute(0, 3, 1, 2)[None].float().to(device)
        with torch.no_grad():
            pred_tracks, pred_vis = cotracker_model(video_tensor, grid_size=30, grid_query_frame=0, backward_tracking=False)
        cam_data.update({
            'tracks_2d': pred_tracks[0].cpu().numpy(),
            'vis_2d': pred_vis[0].cpu().numpy(),
        })
    return scene_constants


# ==========================================
# 10. URDF Kinematics Tracker
# ==========================================

class URDFKinematicsTracker:
    """Forward kinematics-based robot 3D trajectory generator."""

    def __init__(self, pb_renderer):
        self.pb_renderer = pb_renderer

    def get_link_transform(self, obj_id, link_id):
        if link_id == -1:
            pos, orn = p.getBasePositionAndOrientation(obj_id)
        else:
            state = p.getLinkState(obj_id, link_id)
            pos, orn = state[0], state[1]
        T = np.eye(4)
        T[:3, :3] = R.from_quat(orn).as_matrix()
        T[:3, 3] = pos
        return T

    def extract_robot_tracks(self, src_cam, scene_constants, scene_state, safe_margin=7):
        print(f"    🦾 URDF tracking for [{src_cam}]")
        src_data = scene_constants['camera'][src_cam]
        src_state = scene_state[src_cam]
        K_mat = src_data['K_mat']
        extrinsics = src_state['extrinsics']
        h_img, w_img = src_data['video_rgb'][0].shape[:2]
        n_frames = len(src_data['video_rgb'])
        tracks_2d_t0 = src_data['tracks_2d'][0]

        # Frame 0: find seed points on robot
        joint_angles_t0 = scene_constants['robot']['joint_positions'][0]
        gripper_state_t0 = scene_constants['robot']['gripper_positions'][0]
        self.pb_renderer.update_robot_pose(joint_angles_t0, gripper_state=gripper_state_t0)

        cam_pos = extrinsics[0][:3, 3]
        target_pos = cam_pos + extrinsics[0][:3, 2]
        view_matrix = p.computeViewMatrix(cam_pos, target_pos, -extrinsics[0][:3, 1])
        proj_matrix = self.pb_renderer._get_projection_matrix(K_mat, w_img, h_img)

        _, _, _, depth_buffer, seg_buffer = p.getCameraImage(
            w_img, h_img, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        )

        urdf_depth_t0 = 0.1 / (10.0 - 9.99 * np.reshape(depth_buffer, (h_img, w_img)))
        urdf_depth_t0 = np.where(urdf_depth_t0 < 9.9, urdf_depth_t0, 0.0)

        seg_array = np.reshape(seg_buffer, (h_img, w_img)).astype(np.int32)
        obj_ids = seg_array & 0xFFFFFF
        link_ids = (seg_array >> 24) - 1
        is_robot = (obj_ids == self.pb_renderer.robot_id) | (obj_ids == self.pb_renderer.ghost_id)

        kernel = np.ones((safe_margin, safe_margin), np.uint8)
        is_robot_safe = cv2.erode(is_robot.astype(np.uint8), kernel, iterations=1) > 0

        u0 = np.clip(np.round(tracks_2d_t0[:, 0]).astype(int), 0, w_img - 1)
        v0 = np.clip(np.round(tracks_2d_t0[:, 1]).astype(int), 0, h_img - 1)
        on_robot_mask = is_robot_safe[v0, u0]
        robot_indices = np.where(on_robot_mask)[0]

        if len(robot_indices) == 0:
            return None, None, None, None

        robot_objs = obj_ids[v0[robot_indices], u0[robot_indices]]
        robot_links = link_ids[v0[robot_indices], u0[robot_indices]]
        z0 = urdf_depth_t0[v0[robot_indices], u0[robot_indices]]

        pts_world_t0 = unproject_points_np(
            tracks_2d_t0[robot_indices, 0], tracks_2d_t0[robot_indices, 1],
            z0, K_mat, extrinsics[0]
        )

        unique_parts = set(zip(robot_objs, robot_links))
        local_pts_dict = {}
        for obj_id_val, link_id_val in unique_parts:
            part_mask = (robot_objs == obj_id_val) & (robot_links == link_id_val)
            part_pts = pts_world_t0[part_mask]
            T_link_world_t0 = self.get_link_transform(obj_id_val, link_id_val)
            T_world_link_t0 = np.linalg.inv(T_link_world_t0)
            P_homo = np.hstack([part_pts, np.ones((len(part_pts), 1))]).T
            local_pts_dict[(obj_id_val, link_id_val)] = (part_mask, T_world_link_t0 @ P_homo)

        # Forward kinematics
        traj_3d = np.zeros((n_frames, len(robot_indices), 3), dtype=np.float32)
        traj_2d = np.zeros((n_frames, len(robot_indices), 2), dtype=np.float32)
        vis_2d = np.zeros((n_frames, len(robot_indices)), dtype=bool)

        for t in range(n_frames):
            self.pb_renderer.update_robot_pose(
                scene_constants['robot']['joint_positions'][t],
                gripper_state=scene_constants['robot']['gripper_positions'][t]
            )
            for obj_id_val, link_id_val in unique_parts:
                part_mask, P_local_homo = local_pts_dict[(obj_id_val, link_id_val)]
                T_link_world_t = self.get_link_transform(obj_id_val, link_id_val)
                P_world_t = T_link_world_t @ P_local_homo
                traj_3d[t, part_mask, :] = P_world_t[:3, :].T

            u_t, v_t, z_pred_t = project_points_np(traj_3d[t], K_mat, extrinsics[t])
            traj_2d[t, :, 0] = u_t
            traj_2d[t, :, 1] = v_t

            urdf_depth_t = self.pb_renderer.render_depth(extrinsics[t], K_mat, w_img, h_img)
            raw_depth_t = src_data['raw_depth'][t]

            ui = np.clip(np.round(u_t).astype(int), 0, w_img - 1)
            vi = np.clip(np.round(v_t).astype(int), 0, h_img - 1)

            in_bounds = (u_t >= 0) & (u_t < w_img) & (v_t >= 0) & (v_t < h_img) & (z_pred_t > 0)
            z_urdf = urdf_depth_t[vi, ui]
            not_self_occ = (z_urdf > 0) & (z_pred_t <= z_urdf + 0.015)
            z_sensor = raw_depth_t[vi, ui]
            not_env_occ = ~((z_sensor > 0) & (z_pred_t > z_sensor + 0.02))
            vis_2d[t] = in_bounds & not_self_occ & not_env_occ

        return traj_3d, traj_2d, vis_2d, robot_indices


# ==========================================
# 11. Consensus Visual Tracker (Track A)
# ==========================================

class ConsensusVisualTracker:
    """15-vote multi-view spatio-temporal consensus tracker for environment points."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def extract_env_tracks(self, src_cam, env_indices, scene_constants, scene_state, max_env_depth=5.0, occ_margin=0.10):
        print(f"    👁️ Consensus tracking [{src_cam}] ({len(env_indices)} pts)")

        src_data = scene_constants['camera'][src_cam]
        src_state = scene_state[src_cam]
        T_total = len(src_data['video_rgb'])
        N_env = len(env_indices)

        keyframes = [0, T_total // 4, T_total // 2, 3 * T_total // 4, T_total - 1]
        camera_ids = list(scene_constants['camera'].keys())

        src_tracks = src_data['tracks_2d'][:, env_indices, :]
        src_vis = src_data['vis_2d'][:, env_indices]

        vote_traj_dict = {}
        vote_vis_dict = {}

        for tgt_cam in camera_ids:
            tgt_data = scene_constants['camera'][tgt_cam]
            tgt_state = scene_state[tgt_cam]
            h_img, w_img = tgt_data['video_rgb'][0].shape[:2]

            video_tensor = torch.from_numpy(tgt_data['video_rgb']).permute(0, 3, 1, 2)[None].float().to(self.device)

            for t_k in keyframes:
                vote_name = f"{tgt_cam}_kf{t_k}"

                u_src, v_src = src_tracks[t_k, :, 0], src_tracks[t_k, :, 1]
                vis_src = src_vis[t_k]
                ui_src = np.clip(np.round(u_src).astype(int), 0, w_img - 1)
                vi_src = np.clip(np.round(v_src).astype(int), 0, h_img - 1)
                z_src = src_data['raw_depth'][t_k, vi_src, ui_src]
                valid_src = vis_src & (z_src > 0.05) & (z_src < max_env_depth)

                pts_3d_tk = unproject_points_np(u_src, v_src, z_src, src_data['K_mat'], src_state['extrinsics'][t_k])
                u_tgt, v_tgt, z_pred = project_points_np(pts_3d_tk, tgt_data['K_mat'], tgt_state['extrinsics'][t_k])
                ui_tgt = np.clip(np.round(u_tgt).astype(int), 0, w_img - 1)
                vi_tgt = np.clip(np.round(v_tgt).astype(int), 0, h_img - 1)
                z_sensor = tgt_data['raw_depth'][t_k, vi_tgt, ui_tgt]

                in_bounds = (u_tgt >= 0) & (u_tgt < w_img) & (v_tgt >= 0) & (v_tgt < h_img)
                is_visible = (z_sensor > 0) & (z_sensor >= z_pred - occ_margin)
                valid_query_mask = valid_src & in_bounds & is_visible
                valid_indices_arr = np.where(valid_query_mask)[0]

                if len(valid_indices_arr) == 0:
                    vote_traj_dict[vote_name] = np.zeros((T_total, N_env, 3), dtype=np.float32)
                    vote_vis_dict[vote_name] = np.zeros((T_total, N_env), dtype=bool)
                    continue

                queries_np = np.stack([
                    np.full_like(u_tgt[valid_indices_arr], t_k),
                    u_tgt[valid_indices_arr],
                    v_tgt[valid_indices_arr]
                ], axis=-1)
                queries_tensor = torch.tensor(queries_np, dtype=torch.float32, device=self.device)[None, ...]

                with torch.no_grad():
                    pred_tracks, pred_vis = self.model(video_tensor, queries=queries_tensor, backward_tracking=True)

                tgt_tracks_N = np.zeros((T_total, N_env, 2), dtype=np.float32)
                tgt_vis_N = np.zeros((T_total, N_env), dtype=bool)
                tgt_tracks_N[:, valid_indices_arr, :] = pred_tracks[0].cpu().numpy()
                tgt_vis_N[:, valid_indices_arr] = pred_vis[0].cpu().numpy()

                traj_3d_K = np.zeros((T_total, N_env, 3), dtype=np.float32)
                vis_3d_K = np.zeros((T_total, N_env), dtype=bool)

                for t in range(T_total):
                    u_t, v_t = tgt_tracks_N[t, :, 0], tgt_tracks_N[t, :, 1]
                    vis_t = tgt_vis_N[t]
                    ui_t = np.clip(np.round(u_t).astype(int), 0, w_img - 1)
                    vi_t = np.clip(np.round(v_t).astype(int), 0, h_img - 1)
                    z_t = tgt_data['raw_depth'][t, vi_t, ui_t]
                    valid_t = vis_t & (z_t > 0.05) & (z_t < max_env_depth) & (u_t >= 0) & (u_t < w_img) & (v_t >= 0) & (v_t < h_img)
                    vis_3d_K[t] = valid_t
                    if valid_t.any():
                        traj_3d_K[t, valid_t] = unproject_points_np(
                            u_t[valid_t], v_t[valid_t], z_t[valid_t],
                            tgt_data['K_mat'], tgt_state['extrinsics'][t]
                        )

                vote_traj_dict[vote_name] = traj_3d_K
                vote_vis_dict[vote_name] = vis_3d_K

        # Fusion: NaN median + smoothing
        stacked_traj = np.stack(list(vote_traj_dict.values()), axis=0)
        stacked_vis = np.stack(list(vote_vis_dict.values()), axis=0)
        stacked_traj_nan = np.where(stacked_vis[..., None], stacked_traj, np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            unified_traj_3d = np.nanmedian(stacked_traj_nan, axis=0)

        view_counts = np.sum(stacked_vis, axis=0)
        unified_vis = view_counts >= 3

        import pandas as pd
        smoothed_traj_3d = unified_traj_3d.copy()
        df = pd.DataFrame(smoothed_traj_3d.reshape(T_total, -1))
        df = df.interpolate(method='linear', limit_direction='both')
        interpolated_traj = df.to_numpy().reshape(T_total, N_env, 3)
        smoothed_traj_3d = gaussian_filter1d(interpolated_traj, sigma=2.0, axis=0)
        smoothed_traj_3d[~unified_vis] = np.nan

        refined_2d_traj = np.zeros((T_total, N_env, 3), dtype=np.float32)
        for t in range(T_total):
            u_r, v_r, z_r = project_points_np(smoothed_traj_3d[t], src_data['K_mat'], src_state['extrinsics'][t])
            refined_2d_traj[t, :, 0] = u_r
            refined_2d_traj[t, :, 1] = v_r
            refined_2d_traj[t, :, 2] = z_r

        return smoothed_traj_3d, refined_2d_traj, unified_vis


# ==========================================
# 12. TAPVid Export
# ==========================================

def extract_and_export_tapvid(export_root, scene_constants, scene_state,
                               urdf_tracker, consensus_tracker, pb_renderer):
    """Extract dual-track 3D ground truth and serialize to TAPVid format."""
    print("  📦 Extracting and exporting TAPVid data...")
    episode_id = scene_constants['meta']['episode_id']
    camera_ids = list(scene_constants['camera'].keys())
    T_frames = len(scene_constants['camera'][camera_ids[0]]['video_rgb'])

    all_final_tracks = []
    all_queries = []
    all_visibility = {cam: [] for cam in camera_ids}
    total_survivors = 0

    for v_source_idx, src_cam in enumerate(camera_ids):
        cam_data = scene_constants['camera'][src_cam]
        cam_state = scene_state[src_cam]
        h_img, w_img = cam_data['video_rgb'][0].shape[:2]
        tracks_2d = cam_data['tracks_2d']
        raw_depth = cam_data['raw_depth']

        orig_2d_all, traj_3d_all, vis_all = [], [], []

        # Track B: URDF robot points
        traj_3d_rob, proj_2d_rob, vis_rob, robot_indices = urdf_tracker.extract_robot_tracks(
            src_cam, scene_constants, scene_state
        )
        if proj_2d_rob is not None and len(robot_indices) > 0:
            orig_2d_all.append(tracks_2d[:, robot_indices, :])
            traj_3d_all.append(traj_3d_rob)
            vis_all.append(vis_rob)

        # Track A: Environment points with consensus filtering
        joint_angles_t0 = scene_constants['robot']['joint_positions'][0]
        gripper_state_t0 = scene_constants['robot']['gripper_positions'][0]
        pb_renderer.update_robot_pose(joint_angles_t0, gripper_state=gripper_state_t0)

        robot_mask = pb_renderer.render_mask(cam_state['extrinsics'][0], cam_data['K_mat'], w_img, h_img) > 0
        kernel = np.ones((15, 15), np.uint8)
        robot_mask_dilated = cv2.dilate(robot_mask.astype(np.uint8), kernel, iterations=1) > 0

        u0 = np.clip(np.round(tracks_2d[0, :, 0]).astype(int), 0, w_img - 1)
        v0 = np.clip(np.round(tracks_2d[0, :, 1]).astype(int), 0, h_img - 1)
        is_env_point = ~robot_mask_dilated[v0, u0]
        has_valid_depth = (raw_depth[0, v0, u0] > 0.05) & (raw_depth[0, v0, u0] < 5.0)
        test_env_indices = np.where(is_env_point & has_valid_depth)[0]

        if len(test_env_indices) > 0:
            smoothed_3d, refined_2d, unified_vis = consensus_tracker.extract_env_tracks(
                src_cam, test_env_indices, scene_constants, scene_state
            )
            original_2d_env_raw = tracks_2d[:, test_env_indices, :]
            valid_counts = np.sum(unified_vis, axis=0)
            D_2D = np.linalg.norm(original_2d_env_raw - refined_2d[:, :, :2], axis=-1)
            max_drift = np.max(np.where(unified_vis, D_2D, 0), axis=0)
            flicker_counts = np.sum(unified_vis[:-1] != unified_vis[1:], axis=0)

            survivor_mask = (valid_counts >= 5) & (max_drift < 10.0) & (flicker_counts <= T_frames * 0.10)
            golden_indices = np.where(survivor_mask)[0]

            if len(golden_indices) > 0:
                traj_3d_env = smoothed_3d[:, golden_indices, :]
                vis_env = unified_vis[:, golden_indices].copy()
                orig_2d_env = original_2d_env_raw[:, golden_indices, :]
                proj_2d_env = refined_2d[:, golden_indices, :2]
                z_pred_env = refined_2d[:, golden_indices, 2]

                for t in range(T_frames):
                    ui = np.clip(np.round(proj_2d_env[t, :, 0]).astype(int), 0, w_img - 1)
                    vi = np.clip(np.round(proj_2d_env[t, :, 1]).astype(int), 0, h_img - 1)
                    z_sensor = raw_depth[t, vi, ui]
                    xray_occluded = (z_sensor > 0) & (z_sensor < z_pred_env[t] - 0.05)
                    vis_env[t] = vis_env[t] & (~xray_occluded)

                orig_2d_all.append(orig_2d_env)
                traj_3d_all.append(traj_3d_env)
                vis_all.append(vis_env)

        if not orig_2d_all:
            continue

        combined_orig_2d = np.concatenate(orig_2d_all, axis=1)
        combined_traj_3d = np.concatenate(traj_3d_all, axis=1)
        combined_vis = np.concatenate(vis_all, axis=1)

        N_survivors = combined_traj_3d.shape[1]
        total_survivors += N_survivors
        all_final_tracks.append(combined_traj_3d)

        queries = np.zeros((N_survivors, 4), dtype=np.float32)
        queries[:, 0] = combined_orig_2d[0, :, 0]
        queries[:, 1] = combined_orig_2d[0, :, 1]
        queries[:, 2] = 0
        queries[:, 3] = v_source_idx
        all_queries.append(queries)

        # Cross-view visibility
        for tgt_cam in camera_ids:
            if tgt_cam == src_cam:
                all_visibility[tgt_cam].append(combined_vis)
                continue

            tgt_data = scene_constants['camera'][tgt_cam]
            tgt_state = scene_state[tgt_cam]
            tgt_raw_depth = tgt_data['raw_depth']
            tgt_K = tgt_data['K_mat']
            tgt_ext = tgt_state['extrinsics']
            tgt_vis = np.zeros((T_frames, N_survivors), dtype=bool)

            for t in range(T_frames):
                valid_3d = ~np.isnan(combined_traj_3d[t, :, 2])
                if valid_3d.any():
                    u_v, v_v, z_p = project_points_np(combined_traj_3d[t, valid_3d], tgt_K, tgt_ext[t])
                    ui = np.clip(np.round(u_v).astype(int), 0, w_img - 1)
                    vi = np.clip(np.round(v_v).astype(int), 0, h_img - 1)
                    in_bounds = (u_v >= 0) & (u_v < w_img) & (v_v >= 0) & (v_v < h_img) & (z_p > 0)
                    z_sensor = tgt_raw_depth[t, vi, ui]
                    xray_occluded = (z_sensor > 0) & (z_sensor < z_p - 0.05)
                    tgt_vis_t = np.zeros(N_survivors, dtype=bool)
                    tgt_vis_t[valid_3d] = in_bounds & (~xray_occluded)
                    tgt_vis[t] = tgt_vis_t

            all_visibility[tgt_cam].append(tgt_vis)

    if total_survivors == 0:
        print(f"  ⚠️ No valid points! Skipping export.")
        return

    final_tracks_xyz = np.concatenate(all_final_tracks, axis=1)
    queries_xytv = np.concatenate(all_queries, axis=0)

    seq_dir = os.path.join(export_root, episode_id)
    os.makedirs(seq_dir, exist_ok=True)
    np.save(os.path.join(seq_dir, "tracks_xyz.npy"), final_tracks_xyz.astype(np.float32))
    np.save(os.path.join(seq_dir, "queries_xytv.npy"), queries_xytv)

    for v_idx, cam in enumerate(camera_ids):
        cam_dir = os.path.join(seq_dir, str(v_idx))
        os.makedirs(cam_dir, exist_ok=True)
        cam_data = scene_constants['camera'][cam]

        jpeg_bytes_list = [
            cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()
            for img in cam_data['video_rgb']
        ]
        np.save(os.path.join(cam_dir, "images_jpeg_bytes.npy"), np.array(jpeg_bytes_list, dtype=object))

        final_vis = np.concatenate(all_visibility[cam], axis=1)
        np.save(os.path.join(cam_dir, "visibility.npy"), final_vis)

        K = cam_data['K_mat']
        np.save(os.path.join(cam_dir, "intrinsics.npy"), np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32))
        np.save(os.path.join(cam_dir, "extrinsics_w2c.npy"), np.linalg.inv(scene_state[cam]['extrinsics']).astype(np.float32))

    print(f"  🎉 Exported {total_survivors} points to {seq_dir}")


# ==========================================
# 13. Per-Episode Processing Orchestrator
# ==========================================

def process_single_episode(episode_id, data_root, export_root, extrinsics_db,
                           id_to_path, serials_db, keep_ranges,
                           s2m2_model, cotracker_model, vggt_model, sam_predictor,
                           pb_renderer, urdf_tracker, consensus_tracker,
                           device, slice_frames=48):
    """Full processing pipeline for a single DROID episode."""
    print(f"\n{'='*60}")
    print(f"🎬 Processing: {episode_id}")
    print(f"{'='*60}")

    # 1. Download & parse
    scene_constants = download_episode(episode_id, data_root, id_to_path, serials_db, keep_ranges)

    if scene_constants['meta']['valid_indices'] is None or len(scene_constants['meta']['valid_indices']) < slice_frames:
        print("  ⏭️ Too few valid action frames, skipping.")
        return

    scene_constants = extract_svo_video(scene_constants)
    scene_constants = parse_robot_kinematics(scene_constants)
    scene_constants = align_temporal_streams(scene_constants)

    # 2. Initialize extrinsics
    init_scene_state = init_camera_states(scene_constants, extrinsics_db)

    # 3. Filter idle frames (both constants AND state)
    scene_constants, init_scene_state = filter_idle_frames(scene_constants, init_scene_state)

    # 4. Slice to last N frames
    scene_constants, init_scene_state = extract_last_n_frames(scene_constants, init_scene_state, n=slice_frames)

    # 5. Compute stereo depth
    scene_constants = compute_stereo_depth(scene_constants, s2m2_model, device)

    # 6. Extract 2D tracks
    scene_constants = extract_2d_tracks(cotracker_model, scene_constants, device)

    # 7. Check/fix extrinsics
    all_extrinsics_exist = all(
        state['extrinsics'] is not None for state in init_scene_state.values()
    )
    if not all_extrinsics_exist:
        init_scene_state = vggt_warmup_extrinsics(scene_constants, vggt_model, device)

    # 8. Stage 2: Robot alignment
    pybullet_scene_state = run_stage2_robot_alignment(
        scene_constants, init_scene_state, pb_renderer, device
    )

    # 9. Stage 3: Joint alignment
    ultimate_scene_state = run_stage3_joint_alignment(
        scene_constants, pybullet_scene_state, pb_renderer, device
    )

    # 10. Export
    extract_and_export_tapvid(
        export_root, scene_constants, ultimate_scene_state,
        urdf_tracker, consensus_tracker, pb_renderer
    )


# ==========================================
# 14. Main Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="DROID Dataset Processing Pipeline")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (0 or 1)")
    parser.add_argument("--start", type=int, default=0, help="Start index in valid_ids")
    parser.add_argument("--count", type=int, default=2, help="Number of episodes to process")
    parser.add_argument("--data_root", type=str,
                        default=os.path.expanduser("~/droid_data/input/robotics/droid_raw/1.0.1"),
                        help="Root path to DROID raw data")
    parser.add_argument("--export_root", type=str,
                        default=os.path.expanduser("~/droid_data/output"),
                        help="Output directory for TAPVid exports")
    parser.add_argument("--slice_frames", type=int, default=48,
                        help="Number of frames to extract from end of episode")
    args = parser.parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    # Load metadata
    serials_db, id_to_path, keep_ranges, extrinsics_db, valid_ids = load_metadata(args.data_root)

    # Select episode range
    target_episodes = valid_ids[args.start:args.start + args.count]
    print(f"📋 Processing {len(target_episodes)} episodes: [{args.start}:{args.start + args.count}]")

    # Load models
    s2m2_model, cotracker_model, vggt_model, sam_predictor = init_all_models(device)

    # Initialize physics engine
    pb_renderer = PyBulletRenderer_Robotiq()
    urdf_tracker = URDFKinematicsTracker(pb_renderer)
    consensus_tracker = ConsensusVisualTracker(cotracker_model, device)

    os.makedirs(args.export_root, exist_ok=True)

    # Process episodes
    for ep_idx, episode_id in enumerate(target_episodes):
        print(f"\n🎬 [{ep_idx + 1}/{len(target_episodes)}] Episode: {episode_id}")

        try:
            process_single_episode(
                episode_id=episode_id,
                data_root=args.data_root,
                export_root=args.export_root,
                extrinsics_db=extrinsics_db,
                id_to_path=id_to_path,
                serials_db=serials_db,
                keep_ranges=keep_ranges,
                s2m2_model=s2m2_model,
                cotracker_model=cotracker_model,
                vggt_model=vggt_model,
                sam_predictor=sam_predictor,
                pb_renderer=pb_renderer,
                urdf_tracker=urdf_tracker,
                consensus_tracker=consensus_tracker,
                device=device,
                slice_frames=args.slice_frames,
            )
        except Exception as e:
            print(f"  ❌ Error: {e}")
            traceback.print_exc()
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n🏆 All {len(target_episodes)} episodes processed!")


if __name__ == "__main__":
    main()
