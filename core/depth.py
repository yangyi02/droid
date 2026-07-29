"""Stereo depth estimation and gripper depth refinement.

Contains S2M2 disparity inference, SAM-based gripper segmentation, and
temporal depth distillation. All model dependencies are passed as arguments
rather than referenced as globals.
"""

import warnings

import cv2
import numpy as np
import torch
from tqdm import tqdm

from core.geometry import decode_disparity


# ===========================================================================
# S2M2 Stereo Depth
# ===========================================================================

@torch.inference_mode()
def get_s2m2_disparity(img_left, img_right, s2m2_model, run_stereo_matching,
                       device, conf_thresh=0.95):
  """Single-frame disparity extractor (FP32).

  Args:
    img_left: np.ndarray of shape [H, W, 3] (uint8).
    img_right: np.ndarray of shape [H, W, 3] (uint8).
    s2m2_model: the compiled S2M2 model.
    run_stereo_matching: the S2M2 inference function.
    device: torch.device to run inference on.
    conf_thresh: Confidence threshold; pixels below are zeroed out.
  Returns:
    Disparity array of shape [H, W] (float32).
  """
  # [H, W, 3] -> [1, 3, H, W] on GPU
  left_torch = torch.from_numpy(img_left).permute(2, 0, 1).unsqueeze(0).to(device)
  right_torch = torch.from_numpy(img_right).permute(2, 0, 1).unsqueeze(0).to(device)

  pred_disp, _, pred_conf, _, _ = run_stereo_matching(
      s2m2_model, left_torch, right_torch, device, N_repeat=3
  )

  disp = pred_disp.cpu().numpy().squeeze()
  conf = pred_conf.cpu().numpy().squeeze()

  # Mask out low-confidence disparity pixels
  valid_mask = (disp > 0) & (conf >= conf_thresh)
  disp[~valid_mask] = 0.0

  return disp


def compute_stereo_depth(scene_constants, s2m2_model, run_stereo_matching,
                         device):
  """S2M2 stereo depth inference over the full video pool."""
  print("  Running S2M2 stereo depth inference (frame-by-frame)...")

  for cam_id in scene_constants["camera"]:
    cam_data = scene_constants["camera"][cam_id]
    left_seq, right_seq = cam_data["video_rgb"], cam_data["video_right"]

    disp_frames = [
        get_s2m2_disparity(left_img, right_img, s2m2_model,
                           run_stereo_matching, device=device)
        for left_img, right_img in tqdm(
            zip(left_seq, right_seq), total=len(left_seq), desc=f"Depth [{cam_id}]"
        )
    ]
    raw_disp = np.stack(disp_frames)

    fx = cam_data["K_mat"][0, 0]
    baseline = cam_data["baseline"]
    cam_data["raw_depth"] = decode_disparity(raw_disp, fx, baseline)

  return scene_constants


# ===========================================================================
# SAM Gripper Mask Extraction
# ===========================================================================

def extract_single_frame_mask(img_rgb, predictor):
  """Extract a single-frame gripper mask using SAM with positive/negative prompts."""
  h, w = img_rgb.shape[:2]

  points = np.array([
      [w // 2 - 120, h - 110],
      [w // 2 + 500, h - 110],
      [w // 2 - 250, h - 25],
      [w // 2 + 450, h - 25],
      [w // 2 + 100, h - 15],
      [w // 2 + 100, h - 300],
  ])
  labels = np.array([1, 1, 1, 1, 1, 0])
  bbox = np.array([0, h // 2, w, h])

  predictor.set_image(img_rgb)
  masks, scores, _ = predictor.predict(
      point_coords=points, point_labels=labels, box=bbox, multimask_output=True
  )

  valid_masks = []
  for m, s in zip(masks, scores):
    area_ratio = np.sum(m) / (w * h)
    if 0.02 < area_ratio < 0.45:
      valid_masks.append((m, s * area_ratio))

  if valid_masks:
    best_mask = max(valid_masks, key=lambda x: x[1])[0]
  else:
    best_mask = masks[np.argmax(scores)]

  return best_mask


def compute_consensus_mask(masks_list, consensus_thresh=0.5):
  """Compute a consensus mask from multiple per-frame masks via voting."""
  vote_map = np.mean(masks_list, axis=0)
  consensus_mask = vote_map >= consensus_thresh

  num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
      consensus_mask.astype(np.uint8)
  )
  if num_labels > 1:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    consensus_mask = labels_map == largest_label

  return consensus_mask


def build_universal_gripper_mask(scene_constants, sam_predictor):
  """Build a universal static gripper mask from closed-gripper frames using SAM."""
  wrist_cam = scene_constants["meta"].get("wrist_serial")
  if wrist_cam is None or wrist_cam not in scene_constants["camera"]:
    print("  ⚠️ No wrist camera found, skipping gripper mask extraction.")
    return scene_constants

  cam_data = scene_constants["camera"][wrist_cam]
  gripper_states = scene_constants["robot"].get("gripper_positions")
  if gripper_states is None:
    print("  ⚠️ No gripper positions found, skipping gripper mask extraction.")
    return scene_constants

  closed_indices = np.where(gripper_states < 0.05)[0]
  if len(closed_indices) == 0:
    print("  ⚠️ No closed-gripper frames found, skipping mask extraction.")
    return scene_constants

  print(f"  Building consensus gripper mask from {len(closed_indices)} closed-gripper frames...")

  masks_list = []
  for idx in tqdm(closed_indices, desc="SAM mask"):
    img = cam_data["video_rgb"][idx].copy()
    mask = extract_single_frame_mask(img, sam_predictor)
    masks_list.append(mask)

  final_mask = compute_consensus_mask(masks_list)

  # Broadcast the static consensus mask to all closed-gripper frames
  n_frames = len(gripper_states)
  cam_data["sam_real_masks"] = np.zeros(
      (n_frames, *final_mask.shape), dtype=bool
  )
  cam_data["sam_real_masks"][closed_indices] = final_mask
  print(f"  ✅ Gripper consensus mask built and broadcast to {len(closed_indices)} frames.")

  return scene_constants


# ===========================================================================
# Gripper Depth Distillation & Injection
# ===========================================================================

def distill_empirical_gripper_depth(scene_constants, max_depth_thresh=0.15):
  """Distill a clean gripper surface depth via temporal median of masked stereo depth."""
  wrist_cam = scene_constants["meta"].get("wrist_serial")
  if wrist_cam is None or wrist_cam not in scene_constants["camera"]:
    return scene_constants

  cam_data = scene_constants["camera"][wrist_cam]
  gripper_states = scene_constants["robot"].get("gripper_positions")
  if gripper_states is None:
    return scene_constants

  closed_indices = np.where(gripper_states < 0.05)[0]
  if len(closed_indices) == 0:
    print("  ⚠️ No closed-gripper frames for depth distillation.")
    return scene_constants

  if "sam_real_masks" not in cam_data:
    print("  ⚠️ No SAM masks found, skipping depth distillation.")
    return scene_constants

  h, w = cam_data["video_rgb"][0].shape[:2]
  num_frames = len(closed_indices)

  print(f"  Distilling gripper depth from {num_frames} closed-gripper frames...")
  depth_bank = np.full((num_frames, h, w), np.nan, dtype=np.float32)

  for i, idx in enumerate(tqdm(closed_indices, desc="Depth collect")):
    raw_depth = cam_data["raw_depth"][idx].astype(np.float32)
    mask = cam_data["sam_real_masks"][idx]
    valid_pixels = (mask > 0) & (raw_depth > 0) & (raw_depth < max_depth_thresh)
    depth_bank[i, valid_pixels] = raw_depth[valid_pixels]

  print("  Computing temporal median depth...")
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    median_depth = np.nanmedian(depth_bank, axis=0)

  median_depth = np.nan_to_num(median_depth, nan=0.0).astype(np.float32)
  cam_data["empirical_gripper_depth"] = median_depth
  print("  ✅ Gripper depth distillation complete.")

  return scene_constants


def inject_gripper_depth(scene_constants):
  """Inject distilled gripper depth into raw_depth for closed-gripper frames."""
  wrist_cam = scene_constants["meta"].get("wrist_serial")
  if wrist_cam is None or wrist_cam not in scene_constants["camera"]:
    return scene_constants

  cam_data = scene_constants["camera"][wrist_cam]
  gripper_states = scene_constants["robot"].get("gripper_positions")
  empirical_depth = cam_data.get("empirical_gripper_depth")

  if gripper_states is None or empirical_depth is None:
    return scene_constants

  closed_indices = np.where(gripper_states < 0.05)[0]
  if len(closed_indices) == 0:
    return scene_constants

  valid_mask = empirical_depth > 0

  print(f"  Injecting distilled gripper depth into {len(closed_indices)} frames...")
  cam_data["raw_depth"][closed_indices] = np.where(
      valid_mask,
      empirical_depth,
      cam_data["raw_depth"][closed_indices],
  )

  replaced_pixels_per_frame = int(np.sum(valid_mask))
  total_replaced = len(closed_indices) * replaced_pixels_per_frame
  print(f"  ✅ Injection complete! {total_replaced} noisy depth pixels replaced.")

  return scene_constants
