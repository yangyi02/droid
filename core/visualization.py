"""Visualization utilities for the DROID processing pipeline.

Collected from pipeline.ipynb and 2026_06_11_DROID_dataset_v60.ipynb.
Provides point cloud rendering, 2D tracking overlays, mask inspection,
robot segmentation video, camera axes visualization, and 4D orbit video.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
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
                            max_points=150000, eye_pos=(-1.5, -1.5, 1.0),
                            height=600, width=1000):
  """Interactive 3D point cloud rendering via Plotly."""
  import plotly.graph_objects as go  # lazy import
  idx = np.random.permutation(len(pts))[:max_points]
  p, c = pts[idx], cols[idx]
  go.Figure(
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
  ).show(renderer="colab")


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
    title += " 🎨 [Tinted Debug Mode]"
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
                        desc="🎥 Rendering 4D orbit + tracks"):
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
        tk_node = scene.add(pyrender.Mesh.from_trimesh(mesh))
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


# ===========================================================================
# ScenePic Interactive 4D Visualization
# ===========================================================================
# Requires: pip install scenepic scipy scikit-learn

def _sp_normalize(v):
  """Normalize a vector, returning zero-vec if norm is tiny."""
  norm = np.linalg.norm(v)
  return np.where(norm < 1e-8, v, v / norm)


def _sp_form_camera_to_world(origin, target, up):
  """Build a camera-to-world 4x4 matrix from origin, target, up."""
  zc = _sp_normalize(target - origin)
  xc = _sp_normalize(np.cross(up, zc))
  yc = _sp_normalize(np.cross(zc, xc))
  xc = _sp_normalize(np.cross(yc, zc))
  r = np.stack([xc, yc, zc], axis=1)
  t_c2w = np.eye(4)
  t_c2w[:3, :3] = r
  t_c2w[:3, 3] = origin
  return t_c2w


def _sp_inverse_pose(t_c2w):
  """Invert a 4x4 camera-to-world pose."""
  r = t_c2w[:3, :3]
  t = t_c2w[:3, 3]
  r_inv = r.T
  t_inv = -r_inv @ t
  t_w2c = np.eye(4)
  t_w2c[:3, :3] = r_inv
  t_w2c[:3, 3] = t_inv
  return t_w2c


def _sp_transform_points(points, mat):
  """Apply a 4x4 homogeneous transform to (N, 3) points."""
  points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
  transformed_h = np.dot(points_h, mat.T)
  return transformed_h[:, :3] / transformed_h[:, 3:]


def _sp_interpoint_distance(points, *, num=None, quantile=0.03):
  """Estimate a characteristic inter-point distance at a given quantile."""
  from scipy.spatial import distance as sci_dist  # lazy import
  if num is not None:
    max_points = points.shape[0]
    num = min(num, max_points)
    idxs = np.random.choice(max_points, size=num, replace=False)
    points = points[idxs]
  dists = sci_dist.cdist(points, points, metric='euclidean')
  mask = np.triu(np.ones(dists.shape, dtype=bool), k=1)
  dists = dists[mask]
  return np.quantile(dists, quantile)


def _sp_filter_floating_points(pcl_xyzs, pcl_rgbs,
                                density_percentile_removed,
                                k_neighbors=4):
  """Filter points from low-density regions (floating outliers)."""
  from sklearn import neighbors  # lazy import
  n_frames = len(pcl_xyzs)
  dist = []
  for t in range(n_frames):
    points_frame = np.array(pcl_xyzs[t])
    nn_search = neighbors.NearestNeighbors(
        n_neighbors=k_neighbors + 1, algorithm='auto')
    nn_search.fit(points_frame)
    distances, _ = nn_search.kneighbors(points_frame)
    dist.append(distances[:, -1])
  dist_all = np.concatenate(dist, axis=0)
  cutoff_dist = np.percentile(dist_all, 100 - density_percentile_removed)
  for t in range(n_frames):
    pcl_xyzs[t] = pcl_xyzs[t][dist[t] <= cutoff_dist]
    pcl_rgbs[t] = pcl_rgbs[t][dist[t] <= cutoff_dist]
  return pcl_xyzs, pcl_rgbs


def _sp_simulate_orbit(pcl_xyzs, tracks_3d, pred_cam_poses_w2c,
                        gt_cam_poses_w2c, cam_center, look_at, up_dir,
                        orbit_angle_degrees, mode):
  """Transform scene data to simulate an orbiting camera effect."""
  num_frames = len(pcl_xyzs)
  if isinstance(pcl_xyzs, np.ndarray):
    mean_point = np.mean(pcl_xyzs.reshape(-1, 3), axis=0)
  else:
    mean_point = np.concatenate(pcl_xyzs, axis=0).mean(axis=0)
  view_dir = _sp_normalize(look_at - cam_center)
  focus_distance = max(np.dot(mean_point - cam_center, view_dir), 0.1)
  p_focus = cam_center + focus_distance * view_dir
  t_crender_w = _sp_form_camera_to_world(cam_center, look_at, up_dir)
  orbit_angle_radians = np.radians(orbit_angle_degrees)
  r_orbit = focus_distance * np.sin(orbit_angle_radians)
  d_plane = focus_distance * np.cos(orbit_angle_radians)
  c_orbit_center = p_focus - d_plane * view_dir
  x_axis = _sp_normalize(np.cross(up_dir, view_dir))
  y_axis = _sp_normalize(np.cross(view_dir, x_axis))
  frame_angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)

  out_pcl, out_tracks, out_pred, out_gt = [], [], [], []
  viewing_transforms = []
  for t in range(num_frames):
    angle = frame_angles[t]
    c_new = c_orbit_center + r_orbit * (
        np.cos(angle) * x_axis + np.sin(angle) * y_axis)
    if mode == 'orbit_fixing_point':
      z_axis_t = _sp_normalize(p_focus - c_new)
      up_t = _sp_normalize(up_dir - np.dot(up_dir, z_axis_t) * z_axis_t)
      t_ct_w = _sp_form_camera_to_world(c_new, p_focus, up_t)
    elif mode == 'orbit_looking_front':
      t_ct_w = _sp_form_camera_to_world(c_new, c_new + view_dir, up_dir)
    else:
      raise ValueError(f'Unknown mode: {mode}')
    t_w_ct = _sp_inverse_pose(t_ct_w)
    mat = t_crender_w @ t_w_ct
    mat_inv = np.linalg.inv(mat)
    if pcl_xyzs is not None:
      out_pcl.append(_sp_transform_points(pcl_xyzs[t], mat))
    if tracks_3d is not None:
      out_tracks.append(_sp_transform_points(tracks_3d[t], mat))
    if pred_cam_poses_w2c is not None:
      out_pred.append(pred_cam_poses_w2c[t] @ mat_inv)
    if gt_cam_poses_w2c is not None:
      out_gt.append(gt_cam_poses_w2c[t] @ mat_inv)
    viewing_transforms.append([mat, mat_inv])

  stack = lambda x: np.stack(x) if x else None
  return stack(out_pcl), stack(out_tracks), stack(out_pred), stack(out_gt), \
      viewing_transforms


def _sp_init_camera_mesh(poses, scene, thickness, depth, color):
  """Create camera frustum meshes for a sequence of w2c poses.

  Args:
    poses: (T, 4, 4) single camera per frame, or (T, C, 4, 4) multiple
        cameras per frame.

  Returns:
    List of lists of meshes: outer list is per-frame, inner list is
    per-camera within that frame.
  """
  import scenepic as sp  # lazy import
  if poses.ndim == 3:
    # (T, 4, 4) → treat as 1 camera per frame
    assert poses.shape[1:] == (4, 4)
    poses = poses[:, np.newaxis, :, :]  # → (T, 1, 4, 4)
  assert poses.ndim == 4 and poses.shape[2:] == (4, 4)
  frame_meshes = []
  for t in range(poses.shape[0]):
    cam_meshes = []
    for c in range(poses.shape[1]):
      frustum = scene.create_mesh()
      cam = sp.Camera(
          world_to_camera=np.diag([-1, -1, -1, 1]) @ poses[t, c],
          fov_y_degrees=60)
      frustum.add_camera_frustum(
          camera=cam, thickness=thickness, depth=depth, color=color)
      cam_meshes.append(frustum)
    frame_meshes.append(cam_meshes)
  return frame_meshes


def _sp_init_point_cloud_mesh(xyzs, rgbs, scene, radius=0.01):
  """Create instanced sphere meshes for a point cloud sequence."""
  import scenepic as sp  # lazy import
  if isinstance(xyzs, np.ndarray) and not xyzs.size:
    return []
  meshes = []
  for i in range(len(xyzs)):
    spheres = scene.create_mesh()
    spheres.add_sphere(sp.Colors.White, transform=sp.Transforms.Scale(radius))
    spheres.enable_instancing(positions=xyzs[i], colors=rgbs[i])
    meshes.append(spheres)
  return meshes


def _sp_init_tracks_mesh(tracks_3d, scene, radius=0.01, thickness=0.01,
                          viewing_transforms=None, history=3):
  """Create track sphere + trail-line meshes for 3D trajectories."""
  import scenepic as sp  # lazy import
  if tracks_3d is None:
    return [], []
  assert tracks_3d.ndim == 3 and tracks_3d.shape[2] == 3
  n_frames, n_points, _ = tracks_3d.shape
  idx = np.argsort(-1 * tracks_3d[0, :, 1])
  tracks_3d = tracks_3d[:, idx, :]
  cm = plt.get_cmap('hsv')
  colors = cm(np.linspace(0, 1, num=n_points))[:, :3]
  spheres = scene.create_mesh()
  spheres.add_sphere(sp.Colors.White, transform=sp.Transforms.Scale(radius))
  spheres.enable_instancing(positions=tracks_3d[0], colors=colors)
  tracks = []
  for i in range(n_frames):
    tracks.append(
        scene.update_instanced_mesh(spheres.mesh_id, positions=tracks_3d[i]))
  lines = []
  for i in range(n_frames):
    mesh = scene.create_mesh()
    for j in range(max(0, i - history), i):
      k = j + 1
      start_points = tracks_3d[j]
      end_points = tracks_3d[k]
      if viewing_transforms is not None:
        j_to_world = viewing_transforms[j][-1]
        k_to_world = viewing_transforms[k][-1]
        world_to_i = viewing_transforms[i][0]
        start_points = _sp_transform_points(start_points, j_to_world)
        start_points = _sp_transform_points(start_points, world_to_i)
        end_points = _sp_transform_points(end_points, k_to_world)
        end_points = _sp_transform_points(end_points, world_to_i)
      for p in range(n_points):
        mesh.add_thickline(
            color=colors[p],
            start_point=start_points[p],
            end_point=end_points[p],
            start_thickness=thickness,
            end_thickness=thickness)
    lines.append(mesh)
  return tracks, lines


def scenepic_to_html(scene):
  """Convert a scenepic Scene to a self-contained HTML string.

  Applies a bug-fix for the recording frame offset in scenepic's JS lib.
  """
  import scenepic as sp  # lazy import
  scene.quantize_updates()
  sp_lib = sp.js_lib_src()
  # Bug-fix: recording frame offset
  buggy_fn = ('(this.maxFrames=e.FrameCount),'
              'this.numFramesPerCanvas[t]=e.FrameCount}'
              'this.currentRecordingFrame=0')
  fixed_fn = ('(this.maxFrames=e.FrameCount+1),'
              'this.numFramesPerCanvas[t]=e.FrameCount}'
              'this.currentRecordingFrame=1')
  sp_lib = sp_lib.replace(buggy_fn, fixed_fn)
  sp_script = scene.get_script().replace(
      'window.onload = function()', 'function scenepic_main_function()')
  return f"""
  <!DOCTYPE html>
  <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>ScenePic</title>
      <script>{sp_lib}</script>
      <script>{sp_script} scenepic_main_function();</script>
    </head>
    <body onload="scenepic_main_function()"></body>
  </html>
  """


def scenepic_add_point_cloud(
    xyzs, rgbs, tracks_3d=None, *, scene,
    cam_center=None, look_at=None, up_dir=None,
    gt_cam_poses=None, pred_cam_poses=None,
    track_thickness=None, track_sphere_radius=None,
    bg_sphere_radius=None, frustum_thickness=None, frustum_depth=None,
    canvas_height=960, canvas_width=1280,
    auto_scale_factor=1.0, accumulate_point_cloud=False,
    viewing_camera_mode='static', orbit_angle_degrees=0.0,
    density_percentile_removed=0.0, **kwargs):
  """Add an animated point cloud + tracks + camera frustums to a ScenePic scene.

  Args:
    xyzs: list of (N_t, 3) arrays — 3D points per frame, or (T, N, 3) array.
    rgbs: list of (N_t, 3) arrays — RGB colors in [0, 1], or (T, N, 3) array.
    tracks_3d: (T, K, 3) array of 3D track positions, or None.
    scene: scenepic.Scene instance.
    cam_center: (3,) initial camera center, or (T, 3) trajectory.
    look_at: (3,) look-at point.
    up_dir: (3,) up direction.
    gt_cam_poses: (T, C, 4, 4) or (T, 4, 4) ground-truth w2c matrices.
    pred_cam_poses: (T, C, 4, 4) or (T, 4, 4) predicted w2c matrices.
    track_thickness: line thickness for track trails.
    track_sphere_radius: sphere radius for track points.
    bg_sphere_radius: sphere radius for background point cloud.
    frustum_thickness: camera frustum line thickness.
    frustum_depth: camera frustum depth.
    canvas_height: canvas height in pixels.
    canvas_width: canvas width in pixels.
    auto_scale_factor: scale factor for auto-computed sizes.
    accumulate_point_cloud: if True, accumulate points across frames.
    viewing_camera_mode: 'static', 'orbit_fixing_point', or
        'orbit_looking_front'.
    orbit_angle_degrees: orbit angle for non-static modes.
    density_percentile_removed: percentile of low-density points to remove.

  Returns:
    scenepic Canvas3D instance.
  """
  import scenepic as sp  # lazy import
  del kwargs

  if viewing_camera_mode != 'static':
    xyzs, tracks_3d, pred_cam_poses, gt_cam_poses, viewing_transforms = (
        _sp_simulate_orbit(
            pcl_xyzs=xyzs, tracks_3d=tracks_3d,
            pred_cam_poses_w2c=pred_cam_poses,
            gt_cam_poses_w2c=gt_cam_poses,
            cam_center=cam_center, look_at=look_at, up_dir=up_dir,
            orbit_angle_degrees=orbit_angle_degrees,
            mode=viewing_camera_mode))
  else:
    viewing_transforms = None

  if xyzs is not None:
    if density_percentile_removed > 0:
      xyzs, rgbs = _sp_filter_floating_points(
          xyzs, rgbs, density_percentile_removed)
    pt_dist = _sp_interpoint_distance(xyzs[0], num=1000) * auto_scale_factor
  else:
    pt_dist = 1.0
  if track_sphere_radius is None:
    track_sphere_radius = 0.15 * pt_dist
  if track_thickness is None:
    track_thickness = 0.05 * track_sphere_radius
  if bg_sphere_radius is None:
    bg_sphere_radius = 0.10 * pt_dist
  if frustum_thickness is None:
    frustum_thickness = track_thickness
  if frustum_depth is None:
    frustum_depth = 5 * bg_sphere_radius

  n_frames = len(xyzs)
  if tracks_3d is not None:
    n_frames = max(n_frames, tracks_3d.shape[0])

  # Camera setup
  if cam_center is not None and len(cam_center.shape) == 2:
    cam_center_list = cam_center
    cam_center = cam_center_list[0]
  else:
    cam_center_list = None

  if cam_center is None:
    cam_center = np.zeros(3)
  if look_at is None:
    look_at = np.array([0.0, 0.0, 1.0])
  if up_dir is None:
    up_dir = np.array([0.0, -1.0, 0.0])

  camera = sp.Camera(
      center=cam_center, aspect_ratio=(4 / 3), fov_y_degrees=70,
      look_at=look_at, up_dir=up_dir, far_crop_distance=100.0)
  canvas = scene.create_canvas_3d(
      width=canvas_width, height=canvas_height,
      shading=sp.Shading(
          bg_color=sp.Colors.White,
          ambient_light_color=sp.Colors.White,
          directional_light_color=sp.Colors.Black),
      camera=camera)

  # Init meshes
  m_tracks = m_tracks_lines = None
  if tracks_3d is not None:
    m_tracks, m_tracks_lines = _sp_init_tracks_mesh(
        tracks_3d, scene=scene, radius=track_sphere_radius,
        thickness=track_thickness, viewing_transforms=viewing_transforms)
  m_points = None
  if xyzs is not None:
    m_points = _sp_init_point_cloud_mesh(
        xyzs, rgbs, scene=scene, radius=bg_sphere_radius)
  m_gt_camera = None
  if gt_cam_poses is not None:
    m_gt_camera = _sp_init_camera_mesh(
        gt_cam_poses, scene=scene, thickness=frustum_thickness,
        depth=frustum_depth, color=sp.Colors.Green)
  m_pred_camera = None
  if pred_cam_poses is not None:
    m_pred_camera = _sp_init_camera_mesh(
        pred_cam_poses, scene=scene, thickness=frustum_thickness,
        depth=frustum_depth, color=sp.Colors.Orange)

  # Assemble frames
  for i in range(n_frames):
    frame = canvas.create_frame()
    if cam_center_list is not None:
      frame.camera = sp.Camera(
          center=cam_center_list[i], aspect_ratio=(4 / 3), fov_y_degrees=70,
          look_at=look_at, up_dir=up_dir, far_crop_distance=100.0)
    if accumulate_point_cloud and m_points is not None:
      for j in range(i):
        frame.add_mesh(m_points[j])
    elif m_points is not None:
      frame.add_mesh(m_points[i])
    if tracks_3d is not None:
      frame.add_mesh(m_tracks[i])
      frame.add_mesh(m_tracks_lines[i])
    if m_gt_camera is not None:
      for mesh in m_gt_camera[i]:
        frame.add_mesh(mesh)
    if m_pred_camera is not None:
      for mesh in m_pred_camera[i]:
        frame.add_mesh(mesh)
  return canvas


def render_scenepic_html(scene_constants, scene_state,
                         tracks_3d=None, t_start=0, t_end=None,
                         stride=1, framerate=10,
                         frustum_depth=0.1, frustum_thickness=0.005,
                         bg_sphere_radius=0.002, track_sphere_radius=0.005,
                         track_thickness=0.001,
                         output_path=None):
  """Render an interactive ScenePic 4D visualization from the pipeline.

  Produces a self-contained HTML string with an animated, interactive 3D
  point cloud, optional 3D tracks, and camera frustums for all views.

  Requires: ``pip install scenepic scipy scikit-learn``

  Args:
    scene_constants: pipeline scene_constants dict.
    scene_state: pipeline scene_state dict.
    tracks_3d: optional (T, N, 3) array of 3D track positions.
    t_start: first frame index (inclusive).
    t_end: last frame index (exclusive); defaults to all frames.
    stride: spatial stride for subsampling point clouds.
    framerate: animation framerate.
    frustum_depth: camera frustum depth.
    frustum_thickness: camera frustum line thickness.
    bg_sphere_radius: point cloud sphere radius.
    track_sphere_radius: track point sphere radius.
    track_thickness: track trail line thickness.
    output_path: if set, write the HTML to this file path.

  Returns:
    HTML string of the interactive visualization.
  """
  import scenepic as sp  # lazy import

  camera_ids = sorted(scene_constants['camera'].keys())
  first_cam = camera_ids[0]
  n_total_frames = len(scene_constants['camera'][first_cam]['video_rgb'])
  if t_end is None:
    t_end = n_total_frames

  # Build per-frame point clouds from all cameras
  xyzs_per_frame = []
  rgbs_per_frame = []
  extrinsics_all = []

  for t in range(t_start, t_end):
    pts_frame, cols_frame = [], []
    ext_frame = []
    for cam_id in camera_ids:
      cam_data = scene_constants['camera'][cam_id]
      cam_state = scene_state[cam_id]
      raw_depth = cam_data['raw_depth'][t].astype(np.float32)
      rgb_img = cam_data['video_rgb'][t]
      K_mat = cam_data['K_mat']
      extrinsics = cam_state['extrinsics'][t]

      points_3d, colors_rgb = unproject_to_3d(
          raw_depth, rgb_img, K_mat, T_cam2world=extrinsics)

      # Spatial subsampling
      if stride > 1:
        h, w = raw_depth.shape[:2]
        n_pts_full = h * w
        if len(points_3d) == n_pts_full:
          pts_grid = points_3d.reshape(h, w, 3)[::stride, ::stride]
          cols_grid = colors_rgb.reshape(h, w, 3)[::stride, ::stride]
          points_3d = pts_grid.reshape(-1, 3)
          colors_rgb = cols_grid.reshape(-1, 3)

      pts_frame.append(points_3d)
      cols_frame.append(colors_rgb.astype(np.float32) / 255.0)
      ext_frame.append(extrinsics)

    xyzs_per_frame.append(np.concatenate(pts_frame, axis=0))
    rgbs_per_frame.append(np.concatenate(cols_frame, axis=0))
    extrinsics_all.append(np.stack(ext_frame, axis=0))

  # Build w2c camera poses: (T, n_cams, 4, 4)
  extrinsics_stack = np.stack(extrinsics_all, axis=0)  # (T, C, 4, 4)
  # Invert to get w2c from c2w
  w2c_poses = np.linalg.inv(extrinsics_stack)

  # Slice tracks if provided
  sliced_tracks = None
  if tracks_3d is not None:
    sliced_tracks = tracks_3d[t_start:t_end]

  # Create scene
  scene = sp.Scene()
  scene.framerate = framerate

  scenepic_add_point_cloud(
      xyzs=xyzs_per_frame,
      rgbs=rgbs_per_frame,
      tracks_3d=sliced_tracks,
      scene=scene,
      pred_cam_poses=w2c_poses,
      frustum_depth=frustum_depth,
      frustum_thickness=frustum_thickness,
      bg_sphere_radius=bg_sphere_radius,
      track_sphere_radius=track_sphere_radius,
      track_thickness=track_thickness)

  html = scenepic_to_html(scene)

  if output_path is not None:
    with open(output_path, 'w') as f:
      f.write(html)
    print(f"ScenePic visualization saved to: {output_path}")

  return html
