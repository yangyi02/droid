"""DROID Episode Extraction and Vision Foundation Model Pipeline.

A minimalist, high-efficiency multi-stage processor for decoding raw ZED SVO
stereo video streams, extracting robot kinematics, and inferring metric depth.
"""

import argparse
import glob
import json
import os
import sys

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
  vendor_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")
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
  meta_dir = os.path.expanduser("~/droid_data/meta/1.0.1")
  os.makedirs(meta_dir, exist_ok=True)

  base_url = "https://huggingface.co/KarlP/droid/resolve/main"
  files = [
      "intrinsics.json",
      "camera_serials.json",
      "episode_id_to_path.json",
      "keep_ranges_1_0_1.json",
      "cam2base_extrinsic_superset.json",
  ]

  print(f"⬇️ Synchronizing metadata to {meta_dir}...")
  for f in files:
    if not os.path.exists(os.path.join(meta_dir, f)):
      os.system(f"wget -q -nc -P {meta_dir} {base_url}/{f}")

  def load_json(name):
    with open(os.path.join(meta_dir, name), "r") as f:
      return json.load(f)

  serials = load_json("camera_serials.json")
  paths = load_json("episode_id_to_path.json")
  keep = load_json("keep_ranges_1_0_1.json")
  extrinsics = load_json("cam2base_extrinsic_superset.json")

  valid_ids = sorted(set(serials) & set(paths) & set(extrinsics))
  print(f"✅ Matched {len(valid_ids)} completely calibrated episodes.")
  return serials, paths, keep, extrinsics, valid_ids


# ---------------------------------------------------------------------------
# 3. SVO Decoding & Kinematics Extraction
# ---------------------------------------------------------------------------
def download_episode(episode_id, data_root, id_to_path, serials, keep_ranges):
  """Initialize structure for single episode processing."""
  rel_path = id_to_path[episode_id]
  ep_path = os.path.join(data_root, rel_path)
  wrist_serial = serials[episode_id].get("wrist_cam_serial")

  base_prefix = "gs://xembodiment_data/r2d2/r2d2-data-full/"
  episode_key = (
      f"{base_prefix}{rel_path}/recordings/MP4--"
      f"{base_prefix}{rel_path}/trajectory.h5"
  )
  indices = None
  if episode_key in keep_ranges:
    indices = np.array([
        i for start, end in keep_ranges[episode_key]
        for i in range(start, end)
    ])

  return {
      "meta": {
          "episode_id": episode_id,
          "episode_path": ep_path,
          "wrist_serial": wrist_serial,
          "valid_indices": indices,
      },
      "robot": {},
      "camera": {
          cam: {"baseline": 0.063 if cam == wrist_serial else 0.120}
          for cam in sorted(set(serials[episode_id].values()))
      },
  }


def extract_svo_video(scene_constants):
  """Decode twin streams: calibrated (rectified) & physical (unrectified)."""
  print("  🎥 Decoding full-frame stereo SVO sequences and intrinsic matrices...")
  ep_path = scene_constants["meta"]["episode_path"]

  for cam, cam_data in scene_constants["camera"].items():
    svo_files = glob.glob(f"{ep_path}/**/{cam}.svo", recursive=True)
    if not svo_files:
      continue

    zed, param = sl.Camera(), sl.InitParameters()
    param.set_from_svo_file(svo_files[0])
    param.svo_real_time_mode = False
    zed.open(param)

    info = zed.get_camera_information()
    c_par = info.camera_configuration.calibration_parameters
    r_par = info.camera_configuration.calibration_parameters_raw

    def get_k(cam_param):
      return np.array([
          [cam_param.fx, 0, cam_param.cx],
          [0, cam_param.fy, cam_param.cy],
          [0, 0, 1],
      ], dtype=np.float32)

    mat_l, mat_r = sl.Mat(), sl.Mat()
    mat_l_raw, mat_r_raw = sl.Mat(), sl.Mat()
    imgs_l, imgs_r, imgs_l_raw, imgs_r_raw = [], [], [], []

    for _ in tqdm(range(zed.get_svo_number_of_frames()), desc=f"Dec [{cam}]"):
      zed.grab()
      zed.retrieve_image(mat_l, sl.VIEW.LEFT)
      zed.retrieve_image(mat_r, sl.VIEW.RIGHT)
      zed.retrieve_image(mat_l_raw, sl.VIEW.LEFT_UNRECTIFIED)
      zed.retrieve_image(mat_r_raw, sl.VIEW.RIGHT_UNRECTIFIED)

      imgs_l.append(cv2.cvtColor(mat_l.get_data(), cv2.COLOR_BGRA2RGB))
      imgs_r.append(cv2.cvtColor(mat_r.get_data(), cv2.COLOR_BGRA2RGB))
      imgs_l_raw.append(cv2.cvtColor(mat_l_raw.get_data(), cv2.COLOR_BGRA2RGB))
      imgs_r_raw.append(cv2.cvtColor(mat_r_raw.get_data(), cv2.COLOR_BGRA2RGB))

    zed.close()

    cam_data.update({
        "K_mat": get_k(c_par.left_cam),
        "zed_calibration": {
            "calibrated": {
                "K": get_k(c_par.left_cam), "disto": np.float32(c_par.left_cam.disto),
                "K_right": get_k(c_par.right_cam), "disto_right": np.float32(c_par.right_cam.disto),
            },
            "raw": {
                "K": get_k(r_par.left_cam), "disto": np.float32(r_par.left_cam.disto),
                "K_right": get_k(r_par.right_cam), "disto_right": np.float32(r_par.right_cam.disto),
            },
        },
        "video_rgb": np.stack(imgs_l),
        "video_right": np.stack(imgs_r),
        "video_raw_rgb": np.stack(imgs_l_raw),
        "video_raw_right": np.stack(imgs_r_raw),
    })

  return scene_constants


def parse_robot_kinematics(scene_constants):
  """Load HDF5 joint trajectories and high-precision Cartesian kinematics."""
  print("  🦾 Parsing robot kinematic observations...")
  ep_path = scene_constants["meta"]["episode_path"]

  if not (h5 := glob.glob(f"{ep_path}/trajectory.h5")) or not (
      js := glob.glob(f"{ep_path}/metadata_*.json")
  ):
    return scene_constants

  with h5py.File(h5[0], "r") as f, open(js[0]) as jf:
    ee = f["observation/robot_state/cartesian_position"][:]
    joints = f["observation/robot_state/joint_positions"][:]
    grippers = f["observation/robot_state/gripper_position"][:]
    ext = json.load(jf)["wrist_cam_extrinsics"]
    ext = ext.get("extrinsics", ext) if isinstance(ext, dict) else ext

  T_all = np.tile(np.eye(4), (len(ee), 1, 1))
  T_all[:, :3, :3] = R.from_euler("xyz", ee[:, 3:]).as_matrix()
  T_all[:, :3, 3] = ee[:, :3]

  def make_4x4(vec):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", vec[3:]).as_matrix()
    T[:3, 3] = vec[:3]
    return T

  scene_constants["robot"] = {
      "joint_positions": joints,
      "gripper_positions": grippers,
      "T_cam_ee_init": np.linalg.inv(make_4x4(ee[0])) @ make_4x4(ext),
      "T_ee_base_all": T_all,
  }
  return scene_constants


def align_temporal_streams(scene_constants):
  """Guarantee absolute temporal length parity across vision & movement."""
  print("  ⏱️ Harmonizing temporal lengths...")
  if "T_ee_base_all" not in scene_constants["robot"]:
    return scene_constants

  min_frames = min(
      len(v) for d in scene_constants.values()
      if isinstance(d, dict)
      for k, v in d.items()
      if isinstance(v, np.ndarray) and len(v.shape) > 1 and k in (
          "video_rgb", "video_right", "video_raw_rgb", "video_raw_right",
          "joint_positions", "gripper_positions", "T_ee_base_all"
      )
  )

  for d in (scene_constants["robot"], *scene_constants["camera"].values()):
    for k, v in d.items():
      if isinstance(v, np.ndarray) and len(v) > min_frames:
        d[k] = v[:min_frames]

  return scene_constants


# ---------------------------------------------------------------------------
# 4. Stereo Vision Inference
# ---------------------------------------------------------------------------
@torch.inference_mode()
def get_s2m2_disparity(model, img_l, img_r, device, conf_thresh=0.95):
  from s2m2.core.utils.model_utils import run_stereo_matching
  T_l = torch.from_numpy(img_l).permute(2, 0, 1).unsqueeze(0).to(device)
  T_r = torch.from_numpy(img_r).permute(2, 0, 1).unsqueeze(0).to(device)

  disp, _, conf, _, _ = run_stereo_matching(model, T_l, T_r, device, N_repeat=3)
  d_np, c_np = disp.cpu().numpy().squeeze(), conf.cpu().numpy().squeeze()

  valid = (d_np > 0) & (c_np >= conf_thresh)
  return np.where(valid, d_np, 0.0)


def compute_stereo_depth(scene, model, device):
  """Convert stereo correspondences into metric depth maps."""
  print("  🧠 Running S2M2 multi-view stereo depth inference...")
  for cam, cam_data in scene["camera"].items():
    if "video_rgb" not in cam_data or "video_right" not in cam_data:
      continue

    disp = np.stack([
        get_s2m2_disparity(model, l, r, device)
        for l, r in tqdm(
            zip(cam_data["video_rgb"], cam_data["video_right"]),
            total=len(cam_data["video_rgb"]),
            desc=f"S2M [{cam}]",
        )
    ])

    fx, base = cam_data["K_mat"][0, 0], cam_data["baseline"]
    mask = disp > 0
    cam_data["raw_depth"] = np.where(mask, (fx * base) / disp.clip(1e-8), 0.0)

  return scene


def export_to_disk(scene, export_root="~/droid_data/output/mv-tap/droid_stereo_depth"):
  """Export uint16-quantized compressed depth maps and standard mp4 videos."""
  ep_str = scene["meta"]["episode_id"]
  out_dir = os.path.abspath(os.path.expanduser(os.path.join(export_root, ep_str)))
  os.makedirs(out_dir, exist_ok=True)

  print(f"  💾 Exporting ultra-compressed multi-view constants to {out_dir}...")
  for cam_id, data in scene["camera"].items():
    cam_dir = os.path.join(out_dir, str(cam_id))
    os.makedirs(cam_dir, exist_ok=True)

    if "raw_depth" in data:
      # Quantize metric depth to uint16 (1 unit = 0.1 millimeter)
      depth_u16 = (np.clip(data["raw_depth"], 0, 6.5) * 10000).astype(np.uint16)
      np.savez_compressed(
          os.path.join(cam_dir, "raw_depth.npz"),
          depth=depth_u16,
      )
    if "K_mat" in data:
      np.save(os.path.join(cam_dir, "intrinsics.npy"), data["K_mat"].astype(np.float32))
    if "video_rgb" in data:
      h, w = data["video_rgb"][0].shape[:2]
      mp4_path = os.path.join(cam_dir, "video.mp4")
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      out = cv2.VideoWriter(mp4_path, fourcc, 10.0, (w, h))
      for img in data["video_rgb"]:
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
      out.release()

  return True


# ---------------------------------------------------------------------------
# Execution & Batched Slicing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="DROID Flexible Pipeline Extractor")
  parser.add_argument("--start", type=int, default=0, help="Start index in the valid episodes list")
  parser.add_argument("--count", type=int, default=2, help="Number of sequential episodes to process")
  parser.add_argument("--ep_list", type=str, help="Optional comma-separated list of specific episode IDs")
  args = parser.parse_args()

  print("Environment setup verified. Initializing flexible multi-GPU extractor...")
  device = get_accelerator()
  s2m2, cotracker, vggt, sam = init_all_models()
  serials, paths, keep, extrinsics, valid_list = load_metadata()

  if args.ep_list:
    target_eps = [ep.strip() for ep in args.ep_list.split(",") if ep.strip()]
    print(f"📋 Selected via --ep_list targeting: {target_eps}")
  else:
    target_eps = valid_list[args.start : args.start + args.count]
    print(f"📋 Selected via index range [{args.start}:{args.start + args.count}] targeting: {target_eps}")

  for idx, ep_id in enumerate(target_eps):
    print(f"\n🎬 [{idx + 1}/{len(target_eps)}] Processing Episode: {ep_id}")
    if ep_id not in paths:
      print(f"  ❌ Invalid episode ID: {ep_id}")
      continue

    constants = download_episode(
        ep_id,
        os.path.expanduser("~/droid_data/input/robotics/droid_raw/1.0.1"),
        paths,
        serials,
        keep,
    )
    constants = extract_svo_video(constants)
    if not any("video_rgb" in data for data in constants["camera"].values()):
      print(f"  ⚠️ No valid video streams extracted for [{ep_id}]. Skipping processing.")
      continue
    constants = parse_robot_kinematics(constants)
    constants = align_temporal_streams(constants)
    constants = compute_stereo_depth(constants, s2m2, device)
    export_to_disk(constants)

  print("\n🎉 Pipeline successfully completed processing all assigned episodes on this accelerator!")
