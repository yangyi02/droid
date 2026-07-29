"""Visualization utilities for the DROID processing pipeline.

Collected from pipeline.ipynb.
Provides point cloud rendering, 2D tracking overlays, mask inspection,
robot segmentation video, camera axes visualization, and 4D orbit video.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from core.geometry import unproject_to_3d


# ===========================================================================
# Data Inspection
# ===========================================================================

def inspect_dict_structure(data, name="scene_constants", indent=0):
  """Recursively print shape / length / type of every value in a dict."""
  import torch
  spacing = "  " * indent
  if isinstance(data, dict):
    print(f"{spacing}{name} (dict, {len(data)} keys)")
    for k, v in data.items():
      inspect_dict_structure(v, name=str(k), indent=indent + 1)
  elif isinstance(data, np.ndarray):
    print(f"{spacing}{name}: ndarray, shape={data.shape}, dtype={data.dtype}")
  elif torch.is_tensor(data):
    print(f"{spacing}{name}: Tensor, shape={tuple(data.shape)}, dtype={data.dtype}")
  elif isinstance(data, (list, tuple)):
    print(f"{spacing}{name}: {type(data).__name__}, len={len(data)}")
  else:
    val_str = str(data)
    if len(val_str) > 50:
      val_str = val_str[:47] + "..."
    print(f"{spacing}{name}: {type(data).__name__} = {val_str}")


# ===========================================================================
# Static Point Cloud Visualization (Plotly)
# ===========================================================================

def show_plotly_point_cloud(pts, cols, title="3D Point Cloud",
                            max_points=150000, eye_pos=(-1.5, -1.5, 1.0),
                            height=600, width=1000, renderer=None):
  """Interactive 3D point cloud rendering via Plotly."""
  import plotly.graph_objects as go  # lazy import
  idx = np.random.permutation(len(pts))[:max_points]
  p, c = pts[idx], cols[idx]
  fig = go.Figure(
      data=[go.Scatter3d(
          x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',
          marker=dict(size=1.5,
                      color=[f'rgb({r},{g},{b})' for r, g, b in c]))],
      layout=go.Layout(
          title=title, margin=dict(l=0, r=0, b=0, t=40),
          width=width, height=height, showlegend=False,
          scene=dict(aspectmode='data',
                     camera=dict(eye=dict(x=eye_pos[0], y=eye_pos[1],
                                          z=eye_pos[2]))))
  )
  if renderer is not None:
    fig.show(renderer=renderer)
  else:
    fig.show()



def render_fused_point_cloud(scene_constants, scene_state, frame_idx=0,
                             max_render_points=150000,
                             eye_pos=(-1.2, -1.2, 0.8), use_tint=False,
                             height=600, width=1000):
  """Fuse multi-view depth into a single 3D point cloud and render."""
  camera_ids = sorted(scene_constants['camera'].keys())
  tint_colors = np.array([[0, 50, 0], [50, 0, 0], [0, 0, 50]])
  fused_points, fused_colors = [], []

  for idx, cam_id in enumerate(camera_ids):
    cam_data = scene_constants['camera'][cam_id]
    cam_state = scene_state[cam_id]
    raw_depth = cam_data['raw_depth'][frame_idx].astype(np.float32)
    points_3d, colors_rgb = unproject_to_3d(
        raw_depth, cam_data['video_rgb'][frame_idx],
        cam_data['K_mat'], T_cam2world=cam_state['extrinsics'][frame_idx])
    if use_tint:
      colors_rgb = np.clip(
          colors_rgb.astype(int) + tint_colors[idx % len(tint_colors)],
          0, 255).astype(np.uint8)
    fused_points.append(points_3d)
    fused_colors.append(colors_rgb)

  title = f"Fused Point Cloud (Frame {frame_idx})"
  if use_tint:
    title += " [Tinted Debug Mode]"
  show_plotly_point_cloud(
      pts=np.vstack(fused_points), cols=np.vstack(fused_colors),
      title=title, max_points=max_render_points, eye_pos=eye_pos,
      height=height, width=width)


def render_distilled_gripper_3d(median_depth, K_mat, rgb_img):
  """Render the distilled gripper surface as an interactive 3D point cloud."""
  import plotly.graph_objects as go  # lazy import
  v, u = np.where(median_depth > 0)
  z = median_depth[v, u]
  x = (u - K_mat[0, 2]) * z / K_mat[0, 0]
  y = (v - K_mat[1, 2]) * z / K_mat[1, 1]
  pts_3d = np.stack([x, y, z], axis=-1)
  fig = go.Figure(data=[go.Scatter3d(
      x=pts_3d[:, 0], y=pts_3d[:, 1], z=pts_3d[:, 2],
      mode='markers',
      marker=dict(size=2, color=rgb_img[v, u], opacity=0.8))])
  fig.update_layout(
      title="Distilled Gripper Surface",
      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Depth (Z)',
                 aspectmode='data',
                 camera=dict(eye=dict(x=0, y=-0.5, z=-1.5),
                             up=dict(x=0, y=-1, z=0))),
      margin=dict(l=0, r=0, b=0, t=40))
  fig.show()


def render_gripper_refinement_inspection(scene_constants, frame_idx=0):
  """2D inspection of wrist gripper mask and depth refinement."""
  wrist_serial = scene_constants['meta'].get('wrist_serial')
  if not wrist_serial or wrist_serial not in scene_constants['camera']:
    print("⚠️ No wrist camera found in scene_constants.")
    return

  cam_data = scene_constants['camera'][wrist_serial]
  rgb = cam_data['video_rgb'][frame_idx]

  orig_depth = cam_data.get('original_raw_depth')
  raw_depth = cam_data.get('raw_depth')
  gripper_mask = cam_data.get('sam_real_masks')
  emp_depth = cam_data.get('empirical_gripper_depth')

  d_orig = orig_depth[frame_idx] if orig_depth is not None and len(orig_depth) > frame_idx else None
  d_final = raw_depth[frame_idx] if raw_depth is not None and len(raw_depth) > frame_idx else None

  if gripper_mask is not None:
    mask_vis = gripper_mask[min(frame_idx, len(gripper_mask)-1)] if gripper_mask.ndim == 3 else gripper_mask
  else:
    mask_vis = None

  fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

  # 1. RGB + SAM Gripper Mask Overlay
  axes[0].imshow(rgb)
  if mask_vis is not None:
    overlay = rgb.copy()
    overlay[mask_vis > 0] = [255, 0, 128]
    blended = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)
    axes[0].imshow(blended)
    axes[0].set_title("RGB + SAM Gripper Mask", fontsize=11)
  else:
    axes[0].set_title("RGB (No Mask)", fontsize=11)
  axes[0].axis('off')

  # 2. Original Sensor Depth
  if d_orig is not None:
    im1 = axes[1].imshow(np.where(d_orig > 0, d_orig, np.nan), cmap='viridis', vmin=0.1, vmax=1.2)
    axes[1].set_title("Original Sensor Depth", fontsize=11)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
  else:
    axes[1].set_title("Original Depth N/A", fontsize=11)
  axes[1].axis('off')

  # 3. Distilled Gripper Surface Depth
  if emp_depth is not None:
    im2 = axes[2].imshow(np.where(emp_depth > 0, emp_depth, np.nan), cmap='viridis', vmin=0.1, vmax=1.2)
    axes[2].set_title("Distilled Gripper Surface Depth", fontsize=11)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
  else:
    axes[2].set_title("Distilled Gripper Depth N/A", fontsize=11)
  axes[2].axis('off')

  # 4. Final Refined Metric Depth
  if d_final is not None:
    im3 = axes[3].imshow(np.where(d_final > 0, d_final, np.nan), cmap='viridis', vmin=0.1, vmax=1.2)
    axes[3].set_title("Final Refined Depth (Injected)", fontsize=11)
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
  else:
    axes[3].set_title("Final Depth N/A", fontsize=11)
  axes[3].axis('off')

  plt.suptitle(f"Wrist Camera [{wrist_serial[:8]}] Gripper Refinement Inspection (Frame {frame_idx})", fontsize=13, y=1.02)
  plt.tight_layout()
  plt.show()



# ===========================================================================
# Animated 4D Point Cloud (Plotly)
# ===========================================================================

def show_animated_plotly_point_cloud(traj_3d, colors_rgb,
                                    title="Animated 3D Tracks",
                                    eye_pos=(0, -0.8, -1.5),
                                    renderer=None):
  """Dynamic interactive 3D point cloud player with timeline slider."""
  import plotly.graph_objects as go  # lazy import
  T, N, _ = traj_3d.shape
  hex_colors = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in colors_rgb]

  x_min, x_max = np.nanmin(traj_3d[:, :, 0]), np.nanmax(traj_3d[:, :, 0])
  y_min, y_max = np.nanmin(traj_3d[:, :, 1]), np.nanmax(traj_3d[:, :, 1])
  z_min, z_max = np.nanmin(traj_3d[:, :, 2]), np.nanmax(traj_3d[:, :, 2])

  fig = go.Figure(
      data=[go.Scatter3d(
          x=traj_3d[0, :, 0], y=traj_3d[0, :, 1], z=traj_3d[0, :, 2],
          mode='markers', marker=dict(size=2.0, color=hex_colors))])

  frames = []
  for t in range(T):
    frames.append(go.Frame(
        data=[go.Scatter3d(
            x=traj_3d[t, :, 0], y=traj_3d[t, :, 1],
            z=traj_3d[t, :, 2])],
        name=str(t)))
  fig.frames = frames

  sliders = [dict(
      steps=[dict(
          method='animate',
          args=[[str(t)], dict(
              mode='immediate',
              frame=dict(duration=80, redraw=True),
              transition=dict(duration=0))],
          label=f"F{t}") for t in range(T)],
      active=0, transition=dict(duration=0), x=0, y=0)]

  fig.update_layout(
      title=title,
      margin=dict(l=0, r=0, b=0, t=40),
      height=600, showlegend=False,
      scene=dict(
          aspectmode='data',
          xaxis=dict(range=[x_min, x_max], autorange=False),
          yaxis=dict(range=[y_min, y_max], autorange=False),
          zaxis=dict(range=[z_min, z_max], autorange=False),
          camera=dict(eye=dict(x=eye_pos[0], y=eye_pos[1],
                               z=eye_pos[2]))),
      updatemenus=[dict(
          type="buttons", showactive=False, x=0.05, y=1.1,
          buttons=[
              dict(label="▶ Play", method="animate",
                   args=[None, dict(
                       frame=dict(duration=80, redraw=True),
                       transition=dict(duration=0),
                       fromcurrent=True)]),
              dict(label="⏸ Pause", method="animate",
                   args=[[None], dict(
                       frame=dict(duration=0, redraw=False),
                       mode="immediate",
                       transition=dict(duration=0))])])],
      sliders=sliders)
  if renderer is not None:
    fig.show(renderer=renderer)
  else:
    fig.show()



# ===========================================================================
# Disparity & Depth Visualization
# ===========================================================================

def visualize_disparity_video(disp_array, vmax=100.0):
  """Colorize a disparity array as a video using COLORMAP_MAGMA."""
  disp_norm = (np.clip(disp_array, 0, vmax) / vmax * 255).astype(np.uint8)
  return np.stack([
      cv2.cvtColor(cv2.applyColorMap(frame, cv2.COLORMAP_MAGMA),
                   cv2.COLOR_BGR2RGB)
      for frame in disp_norm])


def render_multicam_disparity_video(scene_constants, tgt_size=(128, 228),
                                    disp_vmax=100.0, max_frames=None):
  """Generate a [left | right | disparity] multi-camera stitched video."""
  import mediapy as media  # lazy import; only needed in notebook contexts
  camera_rows = []
  for cam_data in scene_constants['camera'].values():
    video_rgb = cam_data['video_rgb']
    video_right = cam_data['video_right']
    raw_depth = cam_data['raw_depth'].astype(np.float32)
    if max_frames is not None:
      video_rgb = video_rgb[:max_frames]
      video_right = video_right[:max_frames]
      raw_depth = raw_depth[:max_frames]
    left_video = media.resize_video(video_rgb, tgt_size)
    right_video = media.resize_video(video_right, tgt_size)
    fx = cam_data['K_mat'][0, 0]
    baseline = cam_data['baseline']
    raw_disp = np.zeros_like(raw_depth)
    valid_mask = raw_depth > 0
    raw_disp[valid_mask] = (fx * baseline) / raw_depth[valid_mask]
    disp_video = visualize_disparity_video(
        media.resize_video(raw_disp, tgt_size), vmax=disp_vmax)
    camera_rows.append(np.concatenate(
        [left_video, right_video, disp_video], axis=2))
  return np.concatenate(camera_rows, axis=1)


def render_distortion_comparison_video(scene_constants, tgt_width=1280):
  """Render a 2x2 grid comparing raw vs rectified stereo images."""
  print("Rendering stereo distortion comparison video...")
  wrist_cam = scene_constants['meta']['wrist_serial']
  cam_data = scene_constants['camera'][wrist_cam]
  video_rect_l = cam_data['video_rgb']
  video_raw_l = cam_data['video_raw_rgb']
  video_rect_r = cam_data['video_right']
  video_raw_r = cam_data['video_raw_right']
  n_frames, h, w, _ = video_rect_l.shape
  comparison_frames = []

  fx_raw_l = cam_data['zed_calibration']['raw']['K'][0, 0]
  fx_rect_l = cam_data['zed_calibration']['calibrated']['K'][0, 0]
  fx_raw_r = cam_data['zed_calibration']['raw']['K_right'][0, 0]
  fx_rect_r = cam_data['zed_calibration']['calibrated']['K_right'][0, 0]

  for t in tqdm(range(n_frames), desc="Stitching frames"):
    img_raw_l = video_raw_l[t].copy()
    img_rect_l = video_rect_l[t].copy()
    img_raw_r = video_raw_r[t].copy()
    img_rect_r = video_rect_r[t].copy()

    step_y, step_x = h // 6, w // 8
    for img in [img_raw_l, img_rect_l, img_raw_r, img_rect_r]:
      for y in range(0, h, step_y):
        cv2.line(img, (0, y), (w, y), (255, 0, 0), 1)
      for x in range(0, w, step_x):
        cv2.line(img, (x, 0), (x, h), (255, 0, 0), 1)

    def add_label(img, text, color):
      cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                  1.0, (0, 0, 0), 4)
      cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                  1.0, color, 2)

    add_label(img_raw_l, f"LEFT RAW - Fx: {fx_raw_l:.1f}", (0, 255, 255))
    add_label(img_rect_l, f"LEFT RECTIFIED - Fx: {fx_rect_l:.1f}",
              (0, 255, 0))
    add_label(img_raw_r, f"RIGHT RAW - Fx: {fx_raw_r:.1f}", (0, 255, 255))
    add_label(img_rect_r, f"RIGHT RECTIFIED - Fx: {fx_rect_r:.1f}",
              (0, 255, 0))

    top_row = np.concatenate([img_raw_l, img_rect_l], axis=1)
    bottom_row = np.concatenate([img_raw_r, img_rect_r], axis=1)
    combined = np.concatenate([top_row, bottom_row], axis=0)
    tgt_h = int(combined.shape[0] * tgt_width / combined.shape[1])
    comparison_frames.append(cv2.resize(combined, (tgt_width, tgt_h)))

  return comparison_frames


# ===========================================================================
# 2D Tracking Video Rendering
# ===========================================================================

def render_2d_tracking_video(video_frames, tracks, visibility,
                             global_colors=None, linewidth=3,
                             tracks_leave_trace=20, tgt_size=None,
                             max_frames=None):
  """Render 2D point tracks with comet trails onto video frames.

  Args:
    video_frames: (T, H, W, 3) array of RGB frames.
    tracks: (T, N, 2) array of 2D track coordinates.
    visibility: (T, N) boolean visibility array.
    global_colors: optional (N, 3) color array.
    linewidth: width of track lines.
    tracks_leave_trace: number of frames for comet trail.
    tgt_size: optional (H, W) tuple to resize output frames.
    max_frames: optional maximum frames to render.
  """
  import mediapy as media  # lazy import
  if max_frames is not None:
    video_frames = video_frames[:max_frames]
    tracks = tracks[:max_frames]
    visibility = visibility[:max_frames]

  if tgt_size is not None:
    orig_h, orig_w = video_frames[0].shape[:2]
    new_h, new_w = tgt_size
    video_frames = media.resize_video(np.array(video_frames), tgt_size)
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    tracks = tracks * np.array([scale_x, scale_y])

  n_frames, n_points, _ = tracks.shape
  point_radius = int(linewidth * 2)
  h_img, w_img = video_frames[0].shape[:2]
  track_pts = tracks.copy()

  is_valid = ((track_pts[..., 0] >= 0) & (track_pts[..., 0] < w_img) &
              (track_pts[..., 1] >= 0) & (track_pts[..., 1] < h_img))
  is_drawable = is_valid & visibility

  video_frames = [f.copy() for f in video_frames]
  track_pts = np.round(track_pts).astype(np.int32)

  if global_colors is None:
    y_coords = tracks[0, :, 1]
    norm = plt.Normalize(y_coords.min(), y_coords.max())
    global_colors = plt.cm.gist_rainbow(norm(y_coords))[:, :3] * 255
  point_colors = [tuple(map(int, c)) for c in global_colors]

  for t in range(n_frames):
    current_img = video_frames[t]
    trace_len = min(t, tracks_leave_trace)

    # Comet trail rendering
    for step in range(trace_len):
      past_t = t - trace_len + step
      alpha = (step / (trace_len + 1)) ** 2
      overlay = current_img.copy()
      valid_edges = np.where(
          is_drawable[past_t] & is_drawable[past_t + 1])[0]
      for i in valid_edges:
        cv2.line(overlay, tuple(track_pts[past_t, i]),
                 tuple(track_pts[past_t + 1, i]),
                 point_colors[i], linewidth, cv2.LINE_AA)
      cv2.addWeighted(overlay, alpha, current_img, 1 - alpha,
                      0, current_img)

    # Current-frame points
    occ_overlay = current_img.copy()
    has_occlusion = False
    active_points = np.where(is_valid[t])[0]
    for i in active_points:
      pt_coord = tuple(track_pts[t, i])
      if visibility[t, i]:
        cv2.circle(current_img, pt_coord, point_radius,
                   point_colors[i], -1, cv2.LINE_AA)
      else:
        cv2.circle(occ_overlay, pt_coord, point_radius,
                   point_colors[i], 1, cv2.LINE_AA)
        has_occlusion = True
    if has_occlusion:
      cv2.addWeighted(occ_overlay, 0.35, current_img, 0.65,
                      0, current_img)

  return video_frames


# ===========================================================================
# Robot Mask / Segmentation Inspection
# ===========================================================================

def render_multiview_mask_inspection(scene_constants, scene_state,
                                    pb_renderer, frame_idx=0):
  """Multi-camera robot segmentation mask overlay (single frame)."""
  camera_ids = list(scene_constants['camera'].keys())
  wrist_cam = scene_constants['meta']['wrist_serial']

  joint_angles = scene_constants['robot']['joint_positions'][frame_idx]
  gripper_state = scene_constants['robot']['gripper_positions'][frame_idx]
  pb_renderer.update_robot_pose(joint_angles, gripper_state=gripper_state)

  fig, axes = plt.subplots(1, len(camera_ids), figsize=(12, 3))
  if len(camera_ids) == 1:
    axes = [axes]
  fig.suptitle(
      f"Multi-View Segmentation Mask Inspection (Frame {frame_idx})",
      fontsize=20, fontweight='bold', y=1.05)

  for i, cam_id in enumerate(camera_ids):
    extrinsics = scene_state[cam_id]['extrinsics'][frame_idx]
    intrinsics = scene_constants['camera'][cam_id]['K_mat']
    img_rgb = scene_constants['camera'][cam_id]['video_rgb'][frame_idx].copy()
    h_img, w_img = img_rgb.shape[:2]
    robot_mask = pb_renderer.render_mask(
        extrinsics, intrinsics, w_img, h_img) > 0
    overlay = img_rgb.copy()
    overlay[robot_mask] = [50, 255, 50]
    blended_img = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
    cam_type = "Wrist Camera" if cam_id == wrist_cam else "External Camera"
    axes[i].imshow(blended_img)
    axes[i].set_title(f"[{cam_type}]\nCam ID: {cam_id}", fontsize=15)
    axes[i].axis('off')
  plt.tight_layout()
  plt.show()


def inspect_gripper_extremes(scene_constants, scene_state, pb_renderer,
                             tgt_width=1800):
  """Compare robot segmentation at gripper-closed vs gripper-open extremes."""
  gripper_states = scene_constants['robot']['gripper_positions']
  idx_max = np.argmax(np.abs(gripper_states))
  val_max = gripper_states[idx_max]
  idx_zero = np.argmin(np.abs(gripper_states))
  val_zero = gripper_states[idx_zero]

  print(f"State ~ 0: Frame {idx_zero} | gripper={val_zero:.4f}")
  print(f"State Max: Frame {idx_max} | gripper={val_max:.4f}")

  camera_ids = list(scene_constants['camera'].keys())
  wrist_serial = scene_constants['meta']['wrist_serial']

  def render_single_frame(frame_idx, gripper_val):
    current_joints = scene_constants['robot']['joint_positions'][frame_idx]
    pb_renderer.update_robot_pose(current_joints, gripper_state=gripper_val)
    frame_views = []
    for cam_id in camera_ids:
      cam_data = scene_constants['camera'][cam_id]
      cam_state = scene_state[cam_id]
      img_rgb = cam_data['video_rgb'][frame_idx].copy()
      h_img, w_img = img_rgb.shape[:2]
      robot_mask = pb_renderer.render_mask(
          cam_state['extrinsics'][frame_idx],
          cam_data['K_mat'],
          w_img, h_img) > 0
      overlay = img_rgb.copy()
      overlay[robot_mask] = [50, 150, 255]
      blended_img = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
      is_wrist = (cam_id == wrist_serial)
      cam_type = "Wrist Cam" if is_wrist else "Ext Cam"
      cv2.putText(blended_img, f"{cam_type} [{cam_id}]", (20, 50),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
      cv2.putText(blended_img, f"{cam_type} [{cam_id}]", (20, 50),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
      frame_views.append(blended_img)
    row_concat = np.concatenate(frame_views, axis=1)
    tgt_height = int(row_concat.shape[0] * (tgt_width / row_concat.shape[1]))
    return cv2.resize(row_concat, (tgt_width, tgt_height))

  img_zero = render_single_frame(idx_zero, val_zero)
  img_max = render_single_frame(idx_max, val_max)

  fig, axes = plt.subplots(2, 1, figsize=(18, 12))
  axes[0].imshow(img_zero)
  axes[0].set_title(
      f"Gripper State ~ 0 (Frame {idx_zero} | State {val_zero:.4f})",
      fontsize=16, fontweight='bold')
  axes[0].axis('off')
  axes[1].imshow(img_max)
  axes[1].set_title(
      f"Gripper Maximum Value (Frame {idx_max} | State {val_max:.4f})",
      fontsize=16, fontweight='bold')
  axes[1].axis('off')
  plt.tight_layout()
  plt.show()


def render_segmentation_video(scene_constants, scene_state, pb_renderer,
                              tgt_width=1200, max_frames=None):
  """Render a multi-camera robot segmentation overlay video."""
  camera_ids = list(scene_constants['camera'].keys())
  wrist_serial = scene_constants['meta']['wrist_serial']
  n_frames = len(scene_constants['camera'][camera_ids[0]]['video_rgb'])
  if max_frames is not None:
    n_frames = min(n_frames, max_frames)
  video_frames = []

  for frame_idx in tqdm(range(n_frames), desc="Rendering segmentation"):
    current_joints = scene_constants['robot']['joint_positions'][frame_idx]
    current_gripper = scene_constants['robot']['gripper_positions'][frame_idx]
    pb_renderer.update_robot_pose(current_joints,
                                 gripper_state=current_gripper)
    frame_views = []
    for cam_id in camera_ids:
      cam_data = scene_constants['camera'][cam_id]
      cam_state = scene_state[cam_id]
      img_rgb = cam_data['video_rgb'][frame_idx].copy()
      h_img, w_img = img_rgb.shape[:2]
      robot_mask = pb_renderer.render_mask(
          cam_state['extrinsics'][frame_idx],
          cam_data['K_mat'],
          w_img, h_img) > 0
      overlay = img_rgb.copy()
      overlay[robot_mask] = [50, 150, 255]
      blended_img = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
      is_wrist = (cam_id == wrist_serial)
      label_color = (0, 255, 255) if is_wrist else (0, 255, 0)
      cv2.putText(blended_img, f"Cam [{cam_id}]", (20, 50),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
      cv2.putText(blended_img, f"Cam [{cam_id}]", (20, 50),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
      frame_views.append(blended_img)
    row_concat = np.concatenate(frame_views, axis=1)
    tgt_height = int(row_concat.shape[0] * (tgt_width / row_concat.shape[1]))
    video_frames.append(cv2.resize(row_concat, (tgt_width, tgt_height)))
  return video_frames


# ===========================================================================
# Camera Extrinsics Axes Visualization
# ===========================================================================

def render_cross_camera_axes(scene_constants, scene_state, axis_len=0.15,
                             tgt_w=1200, max_frames=None):
  """Render RGB coordinate axes of each camera as seen from other cameras."""
  cams = list(scene_constants['camera'].keys())
  n_frames = len(scene_state[cams[0]]['extrinsics'])
  if max_frames is not None:
    n_frames = min(n_frames, max_frames)
  axes_3d = np.array([
      [0, 0, 0, 1], [axis_len, 0, 0, 1],
      [0, axis_len, 0, 1], [0, 0, axis_len, 1]]).T
  video_frames = []

  for frame_idx in tqdm(range(n_frames), desc="Rendering camera axes"):
    camera_views = []
    for obs_cam in cams:
      cam_data = scene_constants['camera'][obs_cam]
      img_rgb = cam_data['video_rgb'][frame_idx].copy()
      h_img, w_img = img_rgb.shape[:2]
      K_mat = cam_data['K_mat']
      obs_pose_inv = np.linalg.inv(
          scene_state[obs_cam]['extrinsics'][frame_idx])

      for tgt_cam in cams:
        if obs_cam == tgt_cam:
          continue
        tgt_pose = scene_state[tgt_cam]['extrinsics'][frame_idx]
        pts_cam = (obs_pose_inv @ tgt_pose @ axes_3d)[:3, :]
        if pts_cam[2, 0] < 0:
          continue
        uv = K_mat @ pts_cam
        org, px, py, pz = map(
            tuple, (uv[:2] / uv[2]).astype(int).T)
        if 0 <= org[0] < w_img and 0 <= org[1] < h_img:
          cv2.line(img_rgb, org, px, (255, 0, 0), 3)
          cv2.line(img_rgb, org, py, (0, 255, 0), 3)
          cv2.line(img_rgb, org, pz, (0, 0, 255), 3)
          cv2.circle(img_rgb, org, 5, (0, 0, 0), -1)
          cv2.circle(img_rgb, org, 2, (255, 255, 255), -1)
          cv2.putText(img_rgb, f"Cam {tgt_cam}", (org[0]+8, org[1]-8),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
          cv2.putText(img_rgb, f"Cam {tgt_cam}", (org[0]+8, org[1]-8),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

      cv2.putText(img_rgb, f"View: {obs_cam}", (15, 35),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
      cv2.putText(img_rgb, f"View: {obs_cam}", (15, 35),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
      camera_views.append(img_rgb)

    row_concat = np.concatenate(camera_views, axis=1)
    tgt_h = int(row_concat.shape[0] * tgt_w / row_concat.shape[1])
    video_frames.append(cv2.resize(row_concat, (tgt_w, tgt_h)))
  return video_frames


# ===========================================================================
# 4D Cinematic Orbit Video (pyrender)
# ===========================================================================

def get_look_at_matrix(eye, target, up=(0, 0, 1)):
  """Compute an OpenGL-style look-at camera matrix."""
  z_axis = np.array(eye, dtype=float) - np.array(target, dtype=float)
  z_axis /= np.linalg.norm(z_axis) + 1e-6
  x_axis = np.cross(up, z_axis)
  x_axis /= np.linalg.norm(x_axis) + 1e-6
  y_axis = np.cross(z_axis, x_axis)
  view_matrix = np.eye(4)
  view_matrix[:3, :4] = np.column_stack((x_axis, y_axis, z_axis, eye))
  return view_matrix


def render_cinematic_4d_orbit(scene_constants, scene_state,
                              max_render_points=400000,
                              width=640, height=360,
                              orbit_center=(0.4, 0.0, 0.0),
                              orbit_radius=1.2, camera_height=0.5,
                              angle_start=None, max_frames=None):
  """Render a 4D point cloud orbit video using pyrender."""
  import pyrender  # lazy import; heavy dependency
  if angle_start is None:
    angle_start = np.pi / 2

  camera_ids = sorted(scene_constants['camera'].keys())
  n_frames = len(scene_state[camera_ids[0]]['extrinsics'])
  if max_frames is not None:
    n_frames = min(n_frames, max_frames)

  scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0])
  cam_node = scene.add(
      pyrender.PerspectiveCamera(yfov=np.pi / 3.0,
                                 aspectRatio=width / height),
      pose=np.eye(4))
  light_node = scene.add(
      pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0),
      pose=np.eye(4))
  renderer = pyrender.OffscreenRenderer(width, height)

  video_frames = []
  for frame_idx in tqdm(range(n_frames), desc="Rendering 4D orbit"):
    points, colors = [], []
    for cam_id in camera_ids:
      cam_data = scene_constants['camera'][cam_id]
      cam_state = scene_state[cam_id]
      points_3d, colors_rgb = unproject_to_3d(
          cam_data['raw_depth'][frame_idx],
          cam_data['video_rgb'][frame_idx],
          cam_data['K_mat'],
          T_cam2world=cam_state['extrinsics'][frame_idx])
      points.append(points_3d)
      colors.append(colors_rgb)
    points = np.vstack(points)
    colors = np.vstack(colors)
    sample_idx = np.random.permutation(len(points))[:max_render_points]
    points, colors = points[sample_idx], colors[sample_idx]

    angle = angle_start + (frame_idx * np.pi / n_frames)
    eye_pos = [orbit_center[0] + orbit_radius * np.cos(angle),
               orbit_center[1] + orbit_radius * np.sin(angle),
               camera_height]
    viz_pose = get_look_at_matrix(eye_pos, orbit_center)
    scene.set_pose(cam_node, pose=viz_pose)
    scene.set_pose(light_node, pose=viz_pose)

    mesh_node = scene.add(
        pyrender.Mesh.from_points(points, colors=colors))
    color_img, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    scene.remove_node(mesh_node)

    img_rgb = color_img[:, :, :3].copy()
    cv2.putText(img_rgb, f"Frame: {frame_idx:03d}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    video_frames.append(img_rgb)

  renderer.delete()
  return video_frames


def render_4d_orbit_with_tracks(
    scene_constants, scene_state,
    tracks_3d=None, track_colors=None, track_vis=None,
    track_history=5, track_sphere_radius=0.008,
    frustum_depth=0.15, frustum_fov_y=60.0, frustum_aspect=4.0 / 3.0,
    max_render_points=400000, max_render_tracks=500,
    width=640, height=360,
    orbit_center=(0.4, 0.0, 0.0),
    orbit_radius=1.2, camera_height=0.5,
    angle_start=None, max_frames=None):
  """Render a 4D orbit video with point cloud, 3D tracks, and camera frustums.

  Extends ``render_cinematic_4d_orbit`` to also render:
  - 3D track positions as colored spheres
  - Track history trails as colored line segments
  - Per-camera frustum wireframes

  All rendering is done headless via pyrender, producing video frames that
  can be displayed directly in Colab with ``media.show_video()``.

  Args:
    scene_constants: pipeline scene_constants dict.
    scene_state: pipeline scene_state dict.
    tracks_3d: (T, N, 3) 3D track positions, or None.
    track_colors: (N, 3) uint8 RGB, or None for auto-rainbow by y.
    track_vis: (T, N) bool visibility, or None for all-visible.
    track_history: number of trailing frames to show per track.
    track_sphere_radius: radius of each track sphere.
    frustum_depth: depth of camera frustum wireframe.
    frustum_fov_y: vertical FOV (degrees) for frustum shape.
    frustum_aspect: aspect ratio for frustum shape.
    max_render_points: max background point cloud points per frame.
    max_render_tracks: subsample tracks to this count if more.
    width: output frame width.
    height: output frame height.
    orbit_center: (3,) orbit look-at center.
    orbit_radius: radius of the orbit path.
    camera_height: z-height of the orbiting camera.
    angle_start: starting angle (radians); default pi/2.
    max_frames: max frames to render; None = all.

  Returns:
    List of (H, W, 3) uint8 RGB frames.
  """
  import pyrender  # lazy import
  import trimesh   # lazy import
  if angle_start is None:
    angle_start = np.pi / 2

  camera_ids = sorted(scene_constants['camera'].keys())
  n_frames = len(scene_state[camera_ids[0]]['extrinsics'])
  if max_frames is not None:
    n_frames = min(n_frames, max_frames)

  # --- Prepare tracks ---
  if tracks_3d is not None:
    n_total_tracks = tracks_3d.shape[1]
    if n_total_tracks > max_render_tracks:
      idx = np.random.permutation(n_total_tracks)[:max_render_tracks]
      tracks_3d = tracks_3d[:, idx]
      if track_colors is not None:
        track_colors = track_colors[idx]
      if track_vis is not None:
        track_vis = track_vis[:, idx]
    if track_colors is None:
      y0 = tracks_3d[0, :, 1]
      norm = plt.Normalize(y0.min(), y0.max())
      track_colors = (plt.cm.hsv(norm(y0))[:, :3] * 255).astype(np.uint8)
    n_tracks = tracks_3d.shape[1]

  # --- Precompute frustum corners in camera space ---
  half_h = frustum_depth * np.tan(np.radians(frustum_fov_y / 2))
  half_w = half_h * frustum_aspect
  corners_cam = np.array([
      [0, 0, 0],
      [-half_w, -half_h, frustum_depth],
      [ half_w, -half_h, frustum_depth],
      [ half_w,  half_h, frustum_depth],
      [-half_w,  half_h, frustum_depth]])
  frustum_edges = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
  cam_colors_rgb = np.array([
      [1.0, 0.4, 0.2, 1.0],   # orange
      [0.2, 0.8, 0.2, 1.0],   # green
      [0.2, 0.4, 1.0, 1.0],   # blue
      [1.0, 1.0, 0.2, 1.0],   # yellow
  ], dtype=np.float32)

  # --- Precompute track sphere template ---
  if tracks_3d is not None:
    sphere_template = trimesh.creation.icosphere(
        subdivisions=1, radius=track_sphere_radius)
    tpl_v = sphere_template.vertices  # (42, 3)
    tpl_f = sphere_template.faces     # (80, 3)
    n_v, n_f = len(tpl_v), len(tpl_f)

  # --- Setup pyrender scene ---
  scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0])
  cam_node = scene.add(
      pyrender.PerspectiveCamera(
          yfov=np.pi / 3.0, aspectRatio=width / height),
      pose=np.eye(4))
  light_node = scene.add(
      pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0),
      pose=np.eye(4))
  renderer = pyrender.OffscreenRenderer(width, height)

  video_frames = []
  for frame_idx in tqdm(range(n_frames),
                        desc="Rendering 4D orbit + tracks"):
    nodes_to_remove = []

    # --- 1. Background point cloud ---
    points, colors = [], []
    for cam_id in camera_ids:
      cam_data = scene_constants['camera'][cam_id]
      cam_state = scene_state[cam_id]
      pts_3d, cols_rgb = unproject_to_3d(
          cam_data['raw_depth'][frame_idx],
          cam_data['video_rgb'][frame_idx],
          cam_data['K_mat'],
          T_cam2world=cam_state['extrinsics'][frame_idx])
      points.append(pts_3d)
      colors.append(cols_rgb)
    points = np.vstack(points)
    colors = np.vstack(colors)
    sample_idx = np.random.permutation(len(points))[:max_render_points]
    points, colors = points[sample_idx], colors[sample_idx]

    # Orbiting camera
    angle = angle_start + (frame_idx * np.pi / n_frames)
    eye_pos = [orbit_center[0] + orbit_radius * np.cos(angle),
               orbit_center[1] + orbit_radius * np.sin(angle),
               camera_height]
    viz_pose = get_look_at_matrix(eye_pos, orbit_center)
    scene.set_pose(cam_node, pose=viz_pose)
    scene.set_pose(light_node, pose=viz_pose)

    pcl_node = scene.add(
        pyrender.Mesh.from_points(points, colors=colors))
    nodes_to_remove.append(pcl_node)

    # --- 2. Track spheres (batch icospheres) ---
    if tracks_3d is not None:
      vis = track_vis[frame_idx] if track_vis is not None \
          else np.ones(n_tracks, dtype=bool)
      vis_pts = tracks_3d[frame_idx][vis]
      vis_cols = track_colors[vis]

      if len(vis_pts) > 0:
        N = len(vis_pts)
        all_verts = np.tile(tpl_v, (N, 1, 1))       # (N, 42, 3)
        all_verts += vis_pts[:, np.newaxis, :]
        all_verts = all_verts.reshape(-1, 3)

        offsets = np.arange(N) * n_v
        all_faces = np.tile(tpl_f, (N, 1))            # (N*80, 3)
        all_faces += np.repeat(offsets, n_f)[:, np.newaxis]

        face_rgba = np.column_stack([
            np.repeat(vis_cols, n_f, axis=0),
            np.full(N * n_f, 255)]).astype(np.uint8)
        mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
        mesh.visual.face_colors = face_rgba
        tk_node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        nodes_to_remove.append(tk_node)

      # --- 3. Track trails (GL_LINES) ---
      trail_pos, trail_col = [], []
      for j in range(max(0, frame_idx - track_history), frame_idx):
        starts = tracks_3d[j]
        ends = tracks_3d[j + 1]
        mask = np.ones(n_tracks, dtype=bool)
        if track_vis is not None:
          mask &= track_vis[j] & track_vis[j + 1]
        lengths = np.linalg.norm(ends - starts, axis=1)
        mask &= lengths > 1e-6
        if mask.any():
          # Interleave start/end: (n_valid, 2, 3) → (2*n_valid, 3)
          pairs = np.stack([starts[mask], ends[mask]], axis=1)
          trail_pos.append(pairs.reshape(-1, 3))
          tc = track_colors[mask].astype(np.float32) / 255.0
          trail_col.append(np.repeat(tc, 2, axis=0))

      if trail_pos:
        line_pos = np.concatenate(trail_pos).astype(np.float32)
        line_col = np.concatenate(trail_col).astype(np.float32)
        line_rgba = np.column_stack(
            [line_col, np.ones(len(line_col))]).astype(np.float32)
        trail_prim = pyrender.Primitive(
            positions=line_pos, color_0=line_rgba, mode=1)  # GL_LINES
        trail_node = scene.add(
            pyrender.Mesh(primitives=[trail_prim]))
        nodes_to_remove.append(trail_node)

    # --- 4. Camera frustum wireframes (GL_LINES) ---
    frust_pos, frust_col = [], []
    for ci, cam_id in enumerate(camera_ids):
      ext_c2w = scene_state[cam_id]['extrinsics'][frame_idx]
      R, t = ext_c2w[:3, :3], ext_c2w[:3, 3]
      corners_w = (R @ corners_cam.T).T + t
      color = cam_colors_rgb[ci % len(cam_colors_rgb)]
      for ei, ej in frustum_edges:
        frust_pos.extend([corners_w[ei], corners_w[ej]])
        frust_col.extend([color, color])

    if frust_pos:
      frust_pos = np.array(frust_pos, dtype=np.float32)
      frust_col = np.array(frust_col, dtype=np.float32)
      frust_prim = pyrender.Primitive(
          positions=frust_pos, color_0=frust_col, mode=1)  # GL_LINES
      frust_node = scene.add(
          pyrender.Mesh(primitives=[frust_prim]))
      nodes_to_remove.append(frust_node)

    # --- Render & cleanup ---
    color_img, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    for node in nodes_to_remove:
      scene.remove_node(node)

    img_rgb = color_img[:, :, :3].copy()
    cv2.putText(img_rgb, f"Frame: {frame_idx:03d}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    video_frames.append(img_rgb)

  renderer.delete()
  return video_frames


