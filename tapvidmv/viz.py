import cv2
import matplotlib
import numpy as np

import core.geometry
import release
from core.visualization import occlusion_colors


def track_colors(track_ids, colormap="turbo"):
  fractions = np.linspace(0.05, 0.95, max(len(np.asarray(track_ids)), 1))
  cmap = matplotlib.colormaps[colormap]
  return (np.array([cmap(f)[:3] for f in fractions]) * 255).astype(np.uint8)


def ascii_only(text):
  return text.encode("ascii", "replace").decode("ascii")


def view_color(view):
  palette = np.array([[228, 92, 74], [74, 160, 228], [96, 200, 110], [220, 170, 60]])
  return palette[view % len(palette)]


def header_panel(image, text, *, bar=26):
  canvas = np.ascontiguousarray(image)
  scale = max(0.42, canvas.shape[1] / 1100.0)
  bar = max(bar, int(30 * scale))
  strip = np.full((bar, canvas.shape[1], 3), 22, dtype=np.uint8)
  cv2.putText(
    strip,
    ascii_only(text),
    (int(8 * scale), int(bar * 0.72)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55 * scale,
    (235, 235, 235),
    max(1, int(1.4 * scale)),
    cv2.LINE_AA,
  )
  return np.vstack([strip, canvas])


def draw_points(image, xy, *, visible=None, colors=None, radius=4):
  canvas = np.ascontiguousarray(image.copy())
  height, width = canvas.shape[:2]
  colors = track_colors(np.arange(len(xy))) if colors is None else colors
  for index, point in enumerate(xy):
    if not np.isfinite(point).all():
      continue
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    if not (-radius <= x < width + radius and -radius <= y < height + radius):
      continue
    color = tuple(int(c) for c in colors[index % len(colors)])
    is_visible = True if visible is None else bool(visible[index])
    cv2.circle(canvas, (x, y), radius, color, -1 if is_visible else 1, cv2.LINE_AA)
  return canvas


def draw_trails(canvas, trail_xy, colors, *, valid=None):
  length = trail_xy.shape[0]
  for index in range(trail_xy.shape[1]):
    color = tuple(int(c) for c in colors[index % len(colors)])
    for step in range(1, length):
      start, end = trail_xy[step - 1, index], trail_xy[step, index]
      if not (np.isfinite(start).all() and np.isfinite(end).all()):
        continue
      if valid is not None and not (valid[step - 1, index] and valid[step, index]):
        continue
      cv2.line(
        canvas,
        (int(round(float(start[0]))), int(round(float(start[1])))),
        (int(round(float(end[0]))), int(round(float(end[1])))),
        color,
        max(1, int(round(2.0 * step / length))),
        cv2.LINE_AA,
      )
  return canvas


def letterbox(image, cell_hw, background=(0, 0, 0)):
  cell_h, cell_w = cell_hw
  height, width = image.shape[:2]
  scale = min(cell_w / width, cell_h / height)
  resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
  canvas = np.full((cell_h, cell_w, 3), np.array(background, dtype=np.uint8), dtype=np.uint8)
  top, left = (cell_h - resized.shape[0]) // 2, (cell_w - resized.shape[1]) // 2
  canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
  return canvas


def label_panel(image, text):
  canvas = np.ascontiguousarray(image)
  scale = max(0.5, canvas.shape[1] / 900.0)
  origin = (int(8 * scale), int(28 * scale))
  for color, thickness in (((0, 0, 0), int(4 * scale)), ((255, 255, 255), max(1, int(1.5 * scale)))):
    cv2.putText(canvas, ascii_only(text), origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, color, thickness, cv2.LINE_AA)
  return canvas


def montage(panels, *, columns=None, cell_width=480, background=(0, 0, 0)):
  assert panels
  columns = columns or min(len(panels), int(np.ceil(np.sqrt(len(panels)))))
  rows = int(np.ceil(len(panels) / columns))
  aspect = max(panel.shape[0] / panel.shape[1] for panel in panels)
  cell_hw = (int(cell_width * aspect), cell_width)
  cells = [letterbox(panel, cell_hw, background) for panel in panels]
  blank = np.full((*cell_hw, 3), np.array(background, dtype=np.uint8), dtype=np.uint8)
  cells += [blank] * (rows * columns - len(cells))
  return np.vstack([np.hstack(cells[r * columns : (r + 1) * columns]) for r in range(rows)])


def read_video(path, width):
  capture = cv2.VideoCapture(path)
  frames, scale = [], 1.0
  while True:
    ok, frame = capture.read()
    if not ok:
      break
    scale = width / frame.shape[1]
    frames.append(
      cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        (width, max(1, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
      )
    )
  capture.release()
  return frames, scale


def view_panel(episode, view, frame, track_ids, *, colors=None, trail=0, label=True, width=None):
  data = episode.views[view]
  canvas = data.image(frame)
  scale = 1.0
  if width is not None and width != canvas.shape[1]:
    scale = width / canvas.shape[1]
    canvas = cv2.resize(canvas, (width, max(1, int(round(canvas.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
  colors = track_colors(track_ids) if colors is None else np.asarray(colors)

  if trail > 1:
    start = max(0, frame - trail + 1)
    trail_xyz = episode.tracks_xyz[start : frame + 1, track_ids]
    extrinsics = np.repeat(data.T_cam2world[frame][None], len(trail_xyz), axis=0)
    trail_xy, trail_z = core.geometry.project_tracks(trail_xyz, data.K, extrinsics)
    canvas = draw_trails(canvas, trail_xy * scale, colors, valid=trail_z > 1e-3)

  xy, z = episode.project(view)
  front = z[frame, track_ids] > 0
  visible = data.visibility[frame, track_ids] & front
  canvas = draw_points(
    canvas,
    xy[frame, track_ids][front] * scale,
    visible=visible[front],
    colors=colors[front],
    radius=max(3, int(round(canvas.shape[1] / 150))),
  )
  if label:
    canvas = header_panel(
      canvas,
      f"view {view} ({data.kind})  frame {frame}/{episode.num_frames - 1}  vis {int(visible.sum())}/{len(track_ids)}",
    )
  return canvas


def all_views_panel(episode, frame, track_ids, *, colors=None, cell_width=560, trail=0, label=True):
  colors = track_colors(track_ids) if colors is None else colors
  panels = [
    view_panel(episode, view, frame, track_ids, colors=colors, trail=trail, label=label, width=cell_width)
    for view in range(episode.num_views)
  ]
  return montage(panels, columns=episode.num_views, cell_width=cell_width)


def episode_video(episode, track_ids, *, colors=None, num_frames=120, cell_width=380, trail=12, label=True):
  colors = track_colors(track_ids) if colors is None else colors
  frames = np.unique(np.linspace(0, episode.num_frames - 1, min(num_frames, episode.num_frames)).astype(int))
  return np.stack([
    all_views_panel(episode, int(frame), track_ids, colors=colors, cell_width=cell_width, trail=trail, label=label)
    for frame in frames
  ])


def cap_depth(depth, max_depth_m=release.MAX_DEPTH_M):
  capped = np.asarray(depth, dtype=np.float32).copy()
  capped[~(np.isfinite(capped) & (capped > 0.0) & (capped <= max_depth_m))] = 0.0
  return capped


def colorize_depth(depth, *, percentile=99.0):
  valid = np.isfinite(depth) & (depth > 0.0)
  canvas = np.zeros((*depth.shape, 3), dtype=np.uint8)
  if not valid.any():
    return canvas
  low = float(np.percentile(depth[valid], 100.0 - percentile))
  high = float(np.percentile(depth[valid], percentile))
  high = high if high > low else low + 1e-3
  with np.errstate(invalid="ignore"):
    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
  scaled = np.nan_to_num(normalized * 255.0, nan=0.0, posinf=255.0, neginf=0.0)
  colored = cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_TURBO)
  canvas[valid] = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)[valid]
  return canvas


def overlay_mask(image, mask, *, alpha=0.45, color=(255, 64, 64)):
  canvas = image.astype(np.float32).copy()
  canvas[mask] = (1.0 - alpha) * canvas[mask] + alpha * np.array(color, dtype=np.float32)
  return canvas.astype(np.uint8)


def covisibility_matrix(visibility):
  flat = visibility.reshape(-1, visibility.shape[-1]).astype(np.float32)
  return (flat.T @ flat) / flat.shape[0]


def crossview_reprojection(episode, source, target, frame):
  source_data, target_data = episode.views[source], episode.views[target]
  xy_source, z_source = episode.project(source)
  xy_target, z_target = episode.project(target)
  depth = source_data.depth(frame)
  both = source_data.visibility[frame] & target_data.visibility[frame] & (z_source[frame] > 0) & (z_target[frame] > 0)
  tracks = np.flatnonzero(both)
  if depth is None or not len(tracks):
    return tracks[:0], np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
  uv = xy_source[frame][tracks]
  seen_z = core.geometry.sample_depth(depth, uv[:, 0], uv[:, 1], z_source[frame][tracks])
  world = core.geometry.unproject_pixels(uv[:, 0], uv[:, 1], seen_z, source_data.K, source_data.T_cam2world[frame])
  re_u, re_v, z_re = core.geometry.project_points(world, target_data.K, target_data.T_cam2world[frame])
  reprojected = np.stack([re_u, re_v], axis=1)
  keep = np.isfinite(seen_z) & (seen_z > 0) & (z_re > 0)
  behind = seen_z[keep] - z_source[frame][tracks[keep]]
  return tracks[keep], xy_target[frame][tracks[keep]], reprojected[keep], behind


def crossview_gap_px(episode, source, target, frames):
  gaps = []
  for frame in frames:
    _, truth, reprojected, _ = crossview_reprojection(episode, source, target, frame)
    gaps.append(np.linalg.norm(reprojected - truth, axis=-1))
  return np.concatenate(gaps) if gaps else np.zeros(0)
