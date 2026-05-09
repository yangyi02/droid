"""Stereo depth estimation via S2M2.

Test:
  python -c "from droid.depth import compute_stereo_depth; print('✅ depth OK')"
"""

import numpy as np
import torch
from tqdm import tqdm

from core.geometry import decode_disparity_np


@torch.inference_mode()
def get_s2m2_disparity(s2m2_model, img_left, img_right, device, conf_thresh=0.95):
    """Run S2M2 stereo matching on a single image pair.

    Args:
        s2m2_model: compiled S2M2 model
        img_left, img_right: (H, W, 3) uint8 RGB arrays
        device: torch device
        conf_thresh: confidence threshold for valid disparity

    Returns:
        disp: (H, W) float32 disparity map (0 = invalid)
    """
    # Lazy import to avoid loading at module level
    from s2m2.core.utils.model_utils import run_stereo_matching

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
    """Run S2M2 stereo depth on all cameras in scene_constants.

    Adds 'raw_depth' key to each camera's data dict.
    """
    print("  🧠 Running S2M2 stereo depth inference...")
    for cam_id in scene_constants['camera']:
        cam_data = scene_constants['camera'][cam_id]
        left_seq = cam_data['video_rgb']
        right_seq = cam_data['video_right']

        disp_frames = [
            get_s2m2_disparity(s2m2_model, left_img, right_img, device=device)
            for left_img, right_img in tqdm(
                zip(left_seq, right_seq), total=len(left_seq),
                desc=f"    Depth [{cam_id}]"
            )
        ]

        raw_disp = np.stack(disp_frames)
        fx = cam_data['K_mat'][0, 0]
        baseline = cam_data['baseline']
        cam_data['raw_depth'] = decode_disparity_np(raw_disp, fx, baseline)

    return scene_constants
