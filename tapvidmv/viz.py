import cv2
import matplotlib
import numpy as np


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


def read_frames(path, indices):
  wanted = set(int(i) for i in indices)
  capture = cv2.VideoCapture(path)
  frames, index = {}, 0
  while wanted:
    ok, frame = capture.read()
    if not ok:
      break
    if index in wanted:
      frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      wanted.discard(index)
    index += 1
  capture.release()
  return frames


def frame_count(path):
  capture = cv2.VideoCapture(path)
  total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
  capture.release()
  return total
