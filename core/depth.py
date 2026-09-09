import cv2
import numpy as np
import torch
from tqdm import tqdm

import core.geometry


@torch.inference_mode()
def get_s2m2_disparity(img_left, img_right, s2m2_model, run_stereo_matching, device, conf_thresh):
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


def compute_stereo_depth(episode, s2m2_model, run_stereo_matching, device, conf_thresh):

  for cam_id in episode["camera"]:
    cam_data = episode["camera"][cam_id]
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

    fx = cam_data["K"][0, 0]
    baseline = cam_data["baseline"]
    cam_data["raw_depth"] = core.geometry.decode_disparity(raw_disp, fx, baseline)

  return episode


def extract_single_frame_mask(img_rgb, predictor, mask_area_min, mask_area_max):
  height, width = img_rgb.shape[:2]

  points = np.array(
    [
      [width // 2 - 120, height - 110],
      [width // 2 + 500, height - 110],
      [width // 2 - 250, height - 25],
      [width // 2 + 450, height - 25],
      [width // 2 + 100, height - 15],
      [width // 2 + 100, height - 300],
    ]
  )
  labels = np.array([1, 1, 1, 1, 1, 0])
  bbox = np.array([0, height // 2, width, height])

  predictor.set_image(img_rgb)
  masks, scores, _ = predictor.predict(
    point_coords=points, point_labels=labels, box=bbox, multimask_output=True
  )

  valid_masks, valid_scores = [], []
  for m, s in zip(masks, scores):
    area_ratio = np.sum(m) / (width * height)
    if mask_area_min < area_ratio < mask_area_max:
      valid_masks.append(m)
      valid_scores.append(s * area_ratio)

  if valid_masks:
    best_mask = valid_masks[np.argmax(valid_scores)]
  else:
    best_mask = masks[np.argmax(scores)]

  return best_mask


def compute_consensus_mask(masks_list, consensus_thresh):
  vote_map = np.mean(masks_list, axis=0)
  consensus_mask = vote_map >= consensus_thresh

  n_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(consensus_mask.astype(np.uint8))
  if n_labels > 1:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    consensus_mask = labels_map == largest_label

  return consensus_mask


def build_universal_gripper_mask(
  episode, sam_predictor, consensus_thresh, gripper_closed_thresh, mask_area_min, mask_area_max
):
  cam_data = episode["camera"][episode["meta"]["wrist_serial"]]
  gripper_states = episode["robot"]["gripper_positions"]
  closed_indices = np.where(gripper_states < gripper_closed_thresh)[0]

  masks_list = []
  for idx in tqdm(closed_indices, desc="SAM mask"):
    img = cam_data["video_rgb"][idx].copy()
    mask = extract_single_frame_mask(img, sam_predictor, mask_area_min, mask_area_max)
    masks_list.append(mask)

  final_mask = compute_consensus_mask(masks_list, consensus_thresh)

  n_frames = len(gripper_states)
  cam_data["sam_real_masks"] = np.zeros((n_frames, *final_mask.shape), dtype=bool)
  cam_data["sam_real_masks"][closed_indices] = final_mask

  return episode


def distill_empirical_gripper_depth(episode, max_depth_thresh, gripper_closed_thresh):
  cam_data = episode["camera"][episode["meta"]["wrist_serial"]]
  gripper_states = episode["robot"]["gripper_positions"]
  closed_indices = np.where(gripper_states < gripper_closed_thresh)[0]
  height, width = cam_data["video_rgb"][0].shape[:2]
  n_frames = len(closed_indices)

  depth_bank = np.full((n_frames, height, width), np.nan, dtype=np.float32)

  for i, idx in enumerate(tqdm(closed_indices, desc="Depth collect")):
    raw_depth = cam_data["raw_depth"][idx].astype(np.float32)
    mask = cam_data["sam_real_masks"][idx]
    valid_pixels = (mask > 0) & (raw_depth > 0) & (raw_depth < max_depth_thresh)
    depth_bank[i, valid_pixels] = raw_depth[valid_pixels]

  observed = ~np.isnan(depth_bank).all(axis=0)
  median_depth = np.zeros((height, width), dtype=np.float32)
  median_depth[observed] = np.nanmedian(depth_bank[:, observed], axis=0)
  cam_data["empirical_gripper_depth"] = median_depth

  return episode


def inject_gripper_depth(episode, gripper_closed_thresh):
  cam_data = episode["camera"][episode["meta"]["wrist_serial"]]
  gripper_states = episode["robot"]["gripper_positions"]
  empirical_depth = cam_data["empirical_gripper_depth"]

  closed_indices = np.where(gripper_states < gripper_closed_thresh)[0]
  valid_mask = empirical_depth > 0

  cam_data["raw_depth"][closed_indices] = np.where(
    valid_mask, empirical_depth, cam_data["raw_depth"][closed_indices]
  )

  return episode
