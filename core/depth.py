import cv2
import numpy as np
import torch
from tqdm import tqdm

import core.geometry


@torch.inference_mode()
def get_s2m2_disparity(
  img_left, img_right, s2m2_model, run_stereo_matching, device, conf_thresh=0.95
):
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


def compute_stereo_depth(
  scene_constants, s2m2_model, run_stereo_matching, device, conf_thresh=0.95
):

  for cam_id in scene_constants["camera"]:
    cam_data = scene_constants["camera"][cam_id]
    left_seq, right_seq = cam_data["video_rgb"], cam_data["video_right"]

    disp_frames = [
      get_s2m2_disparity(
        left_img, right_img, s2m2_model, run_stereo_matching, device=device, conf_thresh=conf_thresh
      )
      for left_img, right_img in tqdm(
        zip(left_seq, right_seq), total=len(left_seq), desc=f"Depth [{cam_id}]"
      )
    ]
    raw_disp = np.stack(disp_frames)

    fx = cam_data["K_mat"][0, 0]
    baseline = cam_data["baseline"]
    cam_data["raw_depth"] = core.geometry.decode_disparity(raw_disp, fx, baseline)

  return scene_constants


def extract_single_frame_mask(img_rgb, predictor):
  h, w = img_rgb.shape[:2]

  points = np.array(
    [
      [w // 2 - 120, h - 110],
      [w // 2 + 500, h - 110],
      [w // 2 - 250, h - 25],
      [w // 2 + 450, h - 25],
      [w // 2 + 100, h - 15],
      [w // 2 + 100, h - 300],
    ]
  )
  labels = np.array([1, 1, 1, 1, 1, 0])
  bbox = np.array([0, h // 2, w, h])

  predictor.set_image(img_rgb)
  masks, scores, _ = predictor.predict(
    point_coords=points, point_labels=labels, box=bbox, multimask_output=True
  )

  valid_masks, valid_scores = [], []
  for m, s in zip(masks, scores):
    area_ratio = np.sum(m) / (w * h)
    if 0.02 < area_ratio < 0.45:
      valid_masks.append(m)
      valid_scores.append(s * area_ratio)

  if valid_masks:
    best_mask = valid_masks[np.argmax(valid_scores)]
  else:
    best_mask = masks[np.argmax(scores)]

  return best_mask


def compute_consensus_mask(masks_list, consensus_thresh=0.5):
  vote_map = np.mean(masks_list, axis=0)
  consensus_mask = vote_map >= consensus_thresh

  num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
    consensus_mask.astype(np.uint8)
  )
  if num_labels > 1:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    consensus_mask = labels_map == largest_label

  return consensus_mask


def build_universal_gripper_mask(scene_constants, sam_predictor, consensus_thresh=0.5):
  cam_data = scene_constants["camera"][scene_constants["meta"]["wrist_serial"]]
  gripper_states = scene_constants["robot"]["gripper_positions"]
  closed_indices = np.where(gripper_states < 0.05)[0]

  masks_list = []
  for idx in tqdm(closed_indices, desc="SAM mask"):
    img = cam_data["video_rgb"][idx].copy()
    mask = extract_single_frame_mask(img, sam_predictor)
    masks_list.append(mask)

  final_mask = compute_consensus_mask(masks_list, consensus_thresh=consensus_thresh)

  n_frames = len(gripper_states)
  cam_data["sam_real_masks"] = np.zeros((n_frames, *final_mask.shape), dtype=bool)
  cam_data["sam_real_masks"][closed_indices] = final_mask

  return scene_constants


def distill_empirical_gripper_depth(scene_constants, max_depth_thresh=0.15):
  cam_data = scene_constants["camera"][scene_constants["meta"]["wrist_serial"]]
  gripper_states = scene_constants["robot"]["gripper_positions"]
  closed_indices = np.where(gripper_states < 0.05)[0]
  h, w = cam_data["video_rgb"][0].shape[:2]
  num_frames = len(closed_indices)

  depth_bank = np.full((num_frames, h, w), np.nan, dtype=np.float32)

  for i, idx in enumerate(tqdm(closed_indices, desc="Depth collect")):
    raw_depth = cam_data["raw_depth"][idx].astype(np.float32)
    mask = cam_data["sam_real_masks"][idx]
    valid_pixels = (mask > 0) & (raw_depth > 0) & (raw_depth < max_depth_thresh)
    depth_bank[i, valid_pixels] = raw_depth[valid_pixels]

  observed = ~np.isnan(depth_bank).all(axis=0)
  median_depth = np.zeros((h, w), dtype=np.float32)
  median_depth[observed] = np.nanmedian(depth_bank[:, observed], axis=0)
  cam_data["empirical_gripper_depth"] = median_depth

  return scene_constants


def inject_gripper_depth(scene_constants):
  cam_data = scene_constants["camera"][scene_constants["meta"]["wrist_serial"]]
  gripper_states = scene_constants["robot"]["gripper_positions"]
  empirical_depth = cam_data["empirical_gripper_depth"]

  closed_indices = np.where(gripper_states < 0.05)[0]
  valid_mask = empirical_depth > 0

  cam_data["raw_depth"][closed_indices] = np.where(
    valid_mask, empirical_depth, cam_data["raw_depth"][closed_indices]
  )

  return scene_constants
