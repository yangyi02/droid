"""Visualization utilities for the DROID processing pipeline.

Collected from pipeline.ipynb and 2026_06_11_DROID_dataset_v60.ipynb.
Provides point cloud rendering, 2D tracking overlays, mask inspection,
robot segmentation video, camera axes visualization, and 4D orbit video.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

from core.geometry import project_points_np
from core.geometry import unproject_points_np
from core.geometry import unproject_to_3d


# ===========================================================================
# Data Inspection
# ===========================================================================

def inspect_dict_structure(data, name="scene_constants", indent=0):
  """Recursively print shape / length / type of every value in a dict."""
  import torch
  spacing = "  " * indent
  if isinstance(data, dict):
    print(f"{spacing}📂 {name} (dict, {len(data)} keys)")
    for k, v in data.items():
      inspect_dict_structure(v, name=str(k), indent=indent + 1)
  elif isinstance(data, np.ndarray):
    print(f"{spacing}📊 {name}: ndarray, shape={data.shape}, dtype={data.dtype}")
  elif torch.is_tensor(data):
    print(f"{spacing}🔥 {name}: Tensor, shape={tuple(data.shape)}, dtype={data.dtype}")
  elif isinstance(data, (list, tuple)):
    print(f"{spacing}📜 {name}: {type(data).__name__}, len={len(data)}")
  else:
    val_str = str(data)
    if len(val_str) > 50:
      val_str = val_str[:47] + "..."
    print(f"{spacing}🏷️ {name}: {type(data).__name__} = {val_str}")


# ===========================================================================
# Static Point Cloud Visualization (Plotly)
# ===========================================================================

def show_plotly_point_cloud(pts, cols, title="3D Point Cloud",
                            max_points=150000, eye_pos=(-1.5, -1.5, 1.0)):
  """Interactive 3D point cloud rendering via Plotly."""
  idx = np.random.permutation(len(pts))[:max_points]
  p, c = pts[idx], cols[idx]
  go.Figure(
      data=[go.Scatter3d(
          x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',
          marker=dict(size=1.5,
                      color=[f'rgb({r},{g},{b})' for r, g, b in c]))],
      layout=go.Layout(
          title=title, margin=dict(l=0, r=0, b=0, t=40),
          height=500, showlegend=False,
          scene=dict(aspectmode='data',
                     camera=dict(eye=dict(x=eye_pos[0], y=eye_pos[1],
                                          z=eye_pos[2]))))
  ).show(renderer="colab")


def render_fused_point_cloud(scene_constants, scene_state, frame_idx=0,
                             max_render_points=150000,
                             eye_pos=(-1.2, -1.2, 0.8), use_tint=False):
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
    title += " 🎨 [Tinted Debug Mode]"
  show_plotly_point_cloud(
      pts=np.vstack(fused_points), cols=np.vstack(fused_colors),
      title=title, max_points=max_render_points, eye_pos=eye_pos)


def render_distilled_gripper_3d(median_depth, K_mat, rgb_img):
  """Render the distilled gripper surface as an interactive 3D point cloud."""
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
      title="Distilled Gripper Surface 🦾",
      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Depth (Z)',
                 aspectmode='data',
                 camera=dict(eye=dict(x=0, y=-0.5, z=-1.5),
                             up=dict(x=0, y=-1, z=0))),
      margin=dict(l=0, r=0, b=0, t=40))
  fig.show()


# ===========================================================================
# Animated 4D Point Cloud (Plotly)
# ===========================================================================

def show_animated_plotly_point_cloud(traj_3d, colors_rgb,
                                    title="Animated 3D Tracks",
                                    eye_pos=(0, -0.8, -1.5)):
  """Dynamic interactive 3D point cloud player with timeline slider."""
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
  fig.show(renderer="colab")


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
                                    disp_vmax=100.0):
  """Generate a [left | right | disparity] multi-camera stitched video."""
  import mediapy as media  # lazy import; only needed in notebook contexts
  camera_rows = []
  for cam_data in scene_constants['camera'].values():
    left_video = media.resize_video(cam_data['video_rgb'], tgt_size)
    right_video = media.resize_video(cam_data['video_right'], tgt_size)
    raw_depth = cam_data['raw_depth'].astype(np.float32)
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
  print("🎨 Rendering stereo distortion comparison video...")
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
                             tracks_leave_trace=20):
  """Render 2D point tracks with comet trails onto video frames."""
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
# 2D Reprojection Visualization
# ===========================================================================

def lift_tracks_to_3d(tracks_2d, vis_2d, depth, K_mat, extrinsics):
  """Lift 2D tracks to 3D using depth + extrinsics."""
  n_frames, n_points = tracks_2d.shape[:2]
  h_img, w_img = depth.shape[1:3]
  traj_3d = np.zeros((n_frames, n_points, 3))
  zs_src = np.zeros((n_frames, n_points))

  for t in range(n_frames):
    pts = tracks_2d[t]
    in_bounds = ((pts[:, 0] >= 0) & (pts[:, 0] < w_img) &
                 (pts[:, 1] >= 0) & (pts[:, 1] < h_img))
    us = np.clip(np.round(pts[:, 0]).astype(int), 0, w_img - 1)
    vs = np.clip(np.round(pts[:, 1]).astype(int), 0, h_img - 1)
    z_raw = depth[t, vs, us].copy()
    invalid_mask = (~vis_2d[t]) | (~in_bounds)
    z_raw[invalid_mask] = 0.0
    zs_src[t] = z_raw
    traj_3d[t] = unproject_points_np(
        pts[:, 0], pts[:, 1], zs_src[t], K_mat, extrinsics[t])
  return traj_3d, zs_src


def project_to_camera(traj_3d, zs_src, depth, K_mat, extrinsics,
                      depth_margin=0.05):
  """Project 3D trajectories to a target camera with occlusion detection."""
  n_frames, n_points = traj_3d.shape[:2]
  h_img, w_img = depth.shape[1:3]
  proj_trk = np.full((n_frames, n_points, 2), -1000.0)
  proj_vis = np.zeros((n_frames, n_points), dtype=bool)

  for t in range(n_frames):
    u, v, z_pred = project_points_np(traj_3d[t], K_mat, extrinsics[t])
    valid_z = (zs_src[t] > 0.05) & (z_pred > 0.05)
    in_bounds = (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
    valid_mask = valid_z & in_bounds
    proj_trk[t, valid_mask, 0] = u[valid_mask]
    proj_trk[t, valid_mask, 1] = v[valid_mask]
    if valid_mask.any():
      ui = np.clip(np.round(u[valid_mask]).astype(int), 0, w_img - 1)
      vi = np.clip(np.round(v[valid_mask]).astype(int), 0, h_img - 1)
      depth_sensor = depth[t, vi, ui]
      is_occ = ((depth_sensor > 0) &
                (depth_sensor < z_pred[valid_mask] - depth_margin))
      proj_vis[t, valid_mask] = ~is_occ
  return proj_trk, proj_vis


def compute_reprojection_data(src_cam, scene_constants, scene_state):
  """Compute 3D trajectories and cross-view reprojections."""
  camera_ids = list(scene_constants['camera'].keys())
  src_data = scene_constants['camera'][src_cam]
  src_state = scene_state[src_cam]
  tracks_2d = src_data['tracks_2d']
  vis_src = src_data['vis_2d']

  traj_3d, zs_src = lift_tracks_to_3d(
      tracks_2d=tracks_2d, vis_2d=vis_src,
      depth=src_data['raw_depth'], K_mat=src_data['K_mat'],
      extrinsics=src_state['extrinsics'])

  trk_dict, vis_dict = {}, {}
  for tgt_cam in camera_ids:
    if tgt_cam == src_cam:
      trk_dict[tgt_cam] = tracks_2d
      vis_dict[tgt_cam] = vis_src
    else:
      tgt_data = scene_constants['camera'][tgt_cam]
      tgt_state = scene_state[tgt_cam]
      trk_dict[tgt_cam], vis_dict[tgt_cam] = project_to_camera(
          traj_3d=traj_3d, zs_src=zs_src,
          depth=tgt_data['raw_depth'], K_mat=tgt_data['K_mat'],
          extrinsics=tgt_state['extrinsics'])
  return traj_3d, zs_src, vis_src, trk_dict, vis_dict, tracks_2d


def render_all_tracks(src_cam, tracks_2d, vis_src, trk_dict, vis_dict,
                      scene_constants):
  """Render cross-view reprojected tracks as a multi-camera grid video."""
  camera_ids = list(scene_constants['camera'].keys())
  h_img, w_img = scene_constants['camera'][src_cam]['video_rgb'].shape[1:3]
  y_vals = tracks_2d[0, :, 1]
  colors = (plt.cm.gist_rainbow(
      plt.Normalize(y_vals.min(), y_vals.max())(y_vals))[:, :3] * 255)
  res_vids = []
  tgt_size = (320, int(320 * h_img / w_img))
  text_org = (20, 50)

  for tgt_cam in camera_ids:
    combined_vis = vis_dict[tgt_cam] & vis_src
    frames = render_2d_tracking_video(
        video_frames=scene_constants['camera'][tgt_cam]['video_rgb'],
        tracks=trk_dict[tgt_cam], visibility=combined_vis,
        global_colors=colors, linewidth=4)
    label = f"Src:{src_cam} -> Tgt:{tgt_cam}"
    cam_vid = []
    for img in frames:
      cv2.putText(img, label, text_org, cv2.FONT_HERSHEY_SIMPLEX,
                  1.2, (0, 0, 0), 4)
      cv2.putText(img, label, text_org, cv2.FONT_HERSHEY_SIMPLEX,
                  1.2, (255, 255, 255), 2)
      cam_vid.append(cv2.resize(img, tgt_size))
    res_vids.append(np.array(cam_vid))
  return np.concatenate(res_vids, axis=2)


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

  print(f"🎯 State ~ 0: Frame {idx_zero} | gripper={val_zero:.4f}")
  print(f"🎯 State Max: Frame {idx_max} | gripper={val_max:.4f}")

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
          extrinsics=cam_state['extrinsics'][frame_idx],
          intrinsics=cam_data['K_mat'],
          width=w_img, height=h_img) > 0
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
                              tgt_width=1200):
  """Render a multi-camera robot segmentation overlay video."""
  camera_ids = list(scene_constants['camera'].keys())
  wrist_serial = scene_constants['meta']['wrist_serial']
  n_frames = len(scene_constants['camera'][camera_ids[0]]['video_rgb'])
  video_frames = []

  for frame_idx in tqdm(range(n_frames), desc="🎥 Rendering segmentation"):
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
          extrinsics=cam_state['extrinsics'][frame_idx],
          intrinsics=cam_data['K_mat'],
          width=w_img, height=h_img) > 0
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
                             tgt_w=1200):
  """Render RGB coordinate axes of each camera as seen from other cameras."""
  cams = list(scene_constants['camera'].keys())
  n_frames = len(scene_state[cams[0]]['extrinsics'])
  axes_3d = np.array([
      [0, 0, 0, 1], [axis_len, 0, 0, 1],
      [0, axis_len, 0, 1], [0, 0, axis_len, 1]]).T
  video_frames = []

  for frame_idx in tqdm(range(n_frames), desc="🎥 Rendering camera axes"):
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
                              angle_start=None):
  """Render a 4D point cloud orbit video using pyrender."""
  import pyrender  # lazy import; heavy dependency
  if angle_start is None:
    angle_start = np.pi / 2

  camera_ids = sorted(scene_constants['camera'].keys())
  n_frames = len(scene_state[camera_ids[0]]['extrinsics'])

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
  for frame_idx in tqdm(range(n_frames), desc="🎥 Rendering 4D orbit"):
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
