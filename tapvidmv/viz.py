import cv2
import matplotlib
import numpy as np

import release


def track_colors(track_ids, colormap="turbo"):
  fractions = np.linspace(0.05, 0.95, max(len(np.asarray(track_ids)), 1))
  cmap = matplotlib.colormaps[colormap]
  return (np.array([cmap(f)[:3] for f in fractions]) * 255).astype(np.uint8)


def hex_colors(colors):
  colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
  packed = colors[:, 0].astype(np.uint32) << 16 | colors[:, 1].astype(np.uint32) << 8 | colors[:, 2].astype(np.uint32)
  return ["#%06x" % value for value in packed]


def ascii_only(text):
  return text.encode("ascii", "replace").decode("ascii")


def pick_tracks(vis, count=24, *, frame=None, require_views=2, seed=7):
  seen_anywhere = vis.any(axis=1)
  eligible = seen_anywhere.sum(axis=0) >= require_views
  if frame is not None:
    eligible &= vis[:, frame].any(axis=0)

  candidates = np.flatnonzero(eligible)
  if len(candidates) <= count:
    return candidates
  return np.sort(np.random.default_rng(seed).choice(candidates, count, replace=False))


def occlusion_colors(is_robot, visible):
  palette = np.array([[34, 220, 100], [255, 65, 65], [56, 189, 248], [250, 204, 21]], dtype=np.uint8)
  return palette[2 * np.asarray(is_robot, dtype=int) + ~np.asarray(visible, dtype=bool)]


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
    if is_visible:
      cv2.circle(canvas, (x, y), radius, (255, 255, 255), 1, cv2.LINE_AA)
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
  colors = track_colors(track_ids) if colors is None else colors

  if trail > 1:
    start = max(0, frame - trail + 1)
    trail_xyz = episode.tracks_xyz[start : frame + 1, track_ids]
    extrinsics = np.repeat(data.extrinsics_w2c[frame][None], len(trail_xyz), axis=0)
    trail_xy, trail_z = release.project_tracks(trail_xyz, data.intrinsics, extrinsics)
    canvas = draw_trails(canvas, trail_xy * scale, colors, valid=trail_z > 1e-3)

  xy, z = episode.project(view)
  visible = data.visibility[frame, track_ids] & (z[frame, track_ids] > 0)
  canvas = draw_points(
    canvas,
    xy[frame, track_ids] * scale,
    visible=visible,
    colors=colors,
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


def crossview_patches(episode, track, frame, *, patch=112, out_size=180):
  panels = []
  for view in range(episode.num_views):
    data = episode.views[view]
    xy, z = episode.project(view)
    point = xy[frame, track]
    image = data.image(frame)
    height, width = image.shape[:2]
    half = patch // 2
    x = int(round(float(np.clip(point[0], half, width - half - 1))))
    y = int(round(float(np.clip(point[1], half, height - half - 1))))
    crop = cv2.resize(
      image[y - half : y + half, x - half : x + half], (out_size, out_size), interpolation=cv2.INTER_NEAREST
    )
    scale = out_size / patch
    center = (
      int(round((float(point[0]) - (x - half)) * scale)),
      int(round((float(point[1]) - (y - half)) * scale)),
    )
    visible = bool(data.visibility[frame, track]) and float(z[frame, track]) > 0
    color = (60, 220, 60) if visible else (220, 60, 60)
    cv2.drawMarker(crop, center, color, cv2.MARKER_CROSS, int(out_size * 0.25), 2, cv2.LINE_AA)
    cv2.rectangle(crop, (0, 0), (out_size - 1, out_size - 1), color, 3)
    panels.append(label_panel(crop, f"v{view} {'vis' if visible else 'occl'} z={float(z[frame, track]):.2f}"))
  return panels


def plot_tracks_3d(
  episode,
  track_ids,
  *,
  frame=None,
  trail=None,
  colors=None,
  cloud_views=(1, 2),
  cloud_stride=6,
  max_cloud_points=40_000,
  show_cloud=True,
  title=None,
  height=760,
):
  import plotly.graph_objects as go

  frame = episode.num_frames // 2 if frame is None else frame
  track_ids = np.asarray(track_ids)
  colors = track_colors(track_ids) if colors is None else np.asarray(colors)
  figure = go.Figure()

  if show_cloud:
    for view in cloud_views:
      depth = episode.views[view].depth(frame)
      if depth is None:
        continue
      points, point_colors = release.unproject_depth(
        depth,
        episode.views[view].image(frame),
        episode.views[view].intrinsics,
        episode.views[view].extrinsics_w2c[frame],
        stride=cloud_stride,
      )
      if not len(points):
        continue
      if len(points) > max_cloud_points:
        keep = np.random.default_rng(7).choice(len(points), max_cloud_points, replace=False)
        points, point_colors = points[keep], point_colors[keep]
      figure.add_trace(
        go.Scatter3d(
          x=points[:, 0],
          y=points[:, 1],
          z=points[:, 2],
          mode="markers",
          marker=dict(size=1.5, color=hex_colors(point_colors)),
          name=f"depth cloud, view {view}",
          hoverinfo="skip",
        )
      )

  start = 0 if trail is None else max(0, frame - trail + 1)
  stop = episode.num_frames if trail is None else frame + 1
  paths = episode.tracks_xyz[start:stop, track_ids]
  separator = np.full((1, len(track_ids), 3), np.nan, dtype=np.float32)
  joined = np.concatenate([paths, separator]).transpose(1, 0, 2).reshape(-1, 3)
  figure.add_trace(
    go.Scatter3d(
      x=joined[:, 0],
      y=joined[:, 1],
      z=joined[:, 2],
      mode="lines",
      line=dict(color=np.repeat(hex_colors(colors), paths.shape[0] + 1).tolist(), width=4),
      name="3D tracks" if trail is None else f"3D tracks, last {trail} frames",
      hoverinfo="skip",
    )
  )

  heads = episode.tracks_xyz[frame, track_ids]
  figure.add_trace(
    go.Scatter3d(
      x=heads[:, 0],
      y=heads[:, 1],
      z=heads[:, 2],
      mode="markers",
      marker=dict(color=hex_colors(colors), size=5),
      text=[f"track {int(track)}" for track in track_ids],
      hoverinfo="text",
      name=f"positions at frame {frame}",
    )
  )

  for view, data in enumerate(episode.views):
    centers = data.centers
    color = "rgb({},{},{})".format(*view_color(view))
    figure.add_trace(
      go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        mode="lines",
        line=dict(width=4, color=color),
        name=f"camera {view} path",
        hoverinfo="skip",
      )
    )
    figure.add_trace(
      go.Scatter3d(
        x=[centers[frame, 0]],
        y=[centers[frame, 1]],
        z=[centers[frame, 2]],
        mode="markers+text",
        marker=dict(size=5, symbol="diamond", color=color),
        text=[f"v{view}"],
        textposition="top center",
        name=f"camera {view}",
      )
    )

  figure.update_layout(
    height=height,
    margin=dict(l=0, r=0, t=34, b=0),
    title=title if title is not None else f"{episode.name} - 3D ground truth",
    scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
  )
  return figure


def tracks_3d_axes(ax, episode, track_ids, *, colors=None, elev=22, azim=-60, linewidth=0.9, show_cameras=True):
  track_ids = np.asarray(track_ids)
  colors = (track_colors(track_ids) if colors is None else np.asarray(colors)) / 255.0
  paths = episode.tracks_xyz[:, track_ids]
  for index in range(paths.shape[1]):
    ax.plot(*paths[:, index].T, color=colors[index % len(colors)], lw=linewidth)
  ax.scatter(*paths[-1].T, color=colors[: paths.shape[1]], s=4, depthshade=False)
  if show_cameras:
    for view, data in enumerate(episode.views):
      centers = data.centers
      color = view_color(view) / 255.0
      if data.camera_motion_m < 0.01:
        ax.scatter(*centers[0], color=color, s=26, marker="D", depthshade=False)
      else:
        ax.plot(*centers.T, color=color, lw=1.6)
  flat = paths.reshape(-1, 3)
  low, high = flat.min(axis=0), flat.max(axis=0)
  span = np.maximum(high - low, 1e-3)
  low, high = low - 0.15 * span, high + 0.15 * span
  ax.set_xlim(low[0], high[0])
  ax.set_ylim(low[1], high[1])
  ax.set_zlim(low[2], high[2])
  ax.set_box_aspect(high - low)
  ax.view_init(elev=elev, azim=azim)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])
  return ax


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


def crossview_gap_px(episode, source, target, frames):
  import core.geometry

  source_data, target_data = episode.views[source], episode.views[target]
  xy_source, z_source = episode.project(source)
  xy_target, z_target = episode.project(target)
  gaps = []
  for frame in frames:
    depth = source_data.depth(frame)
    if depth is None:
      continue
    both = source_data.visibility[frame] & target_data.visibility[frame] & (z_source[frame] > 0) & (z_target[frame] > 0)
    if not both.any():
      continue
    uv = xy_source[frame][both]
    seen_z = core.geometry.sample_depth(depth, uv[:, 0], uv[:, 1], z_source[frame][both])
    fx, fy, cx, cy = [float(v) for v in source_data.intrinsics]
    camera_points = np.stack([(uv[:, 0] - cx) / fx * seen_z, (uv[:, 1] - cy) / fy * seen_z, seen_z], axis=-1)
    rotation, translation = source_data.extrinsics_w2c[frame][:3, :3], source_data.extrinsics_w2c[frame][:3, 3]
    world = (camera_points - translation) @ rotation
    reprojected, z_re = release.project_tracks(
      world[None], target_data.intrinsics, target_data.extrinsics_w2c[frame][None]
    )
    keep = np.isfinite(seen_z) & (seen_z > 0) & (z_re[0] > 0)
    gaps.append(np.linalg.norm(reprojected[0][keep] - xy_target[frame][both][keep], axis=-1))
  return np.concatenate(gaps) if gaps else np.zeros(0)
