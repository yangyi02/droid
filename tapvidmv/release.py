import functools
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MAX_DEPTH_M = 2.0

TRACKS = "tracks_xyz.npy"
QUERIES = "queries_xytv.npy"
IMAGES = "images_jpeg_bytes.npy"
INTRINSICS = "intrinsics.npy"
EXTRINSICS = "extrinsics_w2c.npy"
VISIBILITY = "visibility.npy"
DEPTH = "depth.npy"
MASK = "foreground_mask.npy"


def find_dataset():
  return Path(__file__).parent / "data"


def episode_names(root):
  root = Path(root)
  assert root.is_dir(), f"no export at {root}"
  return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / TRACKS).exists())


def decode_jpeg(raw):
  buffer = np.frombuffer(bytes(raw), dtype=np.uint8)
  return cv2.cvtColor(cv2.imdecode(buffer, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def project_tracks(tracks_xyz, intrinsics, extrinsics_w2c):
  points_h = np.concatenate([tracks_xyz, np.ones((*tracks_xyz.shape[:2], 1), dtype=np.float32)], axis=-1)
  points_camera = np.einsum("tij,tnj->tni", extrinsics_w2c, points_h)
  z = points_camera[..., 2]
  with np.errstate(invalid="ignore", divide="ignore"):
    xy = points_camera[..., :2] / z[..., None]
  xy = xy * intrinsics[None, None, :2] + intrinsics[None, None, 2:]
  return xy.astype(np.float32), z.astype(np.float32)


def camera_centers(extrinsics_w2c):
  rotation, translation = extrinsics_w2c[:, :3, :3], extrinsics_w2c[:, :3, 3]
  return -np.einsum("tji,tj->ti", rotation, translation)


def unproject_depth(depth, image, intrinsics, extrinsics_w2c, *, stride=6, max_depth_m=MAX_DEPTH_M):
  height, width = depth.shape
  rows, columns = np.mgrid[0:height:stride, 0:width:stride]
  z = depth[::stride, ::stride]
  valid = np.isfinite(z) & (z > 0.0)
  if max_depth_m is not None:
    valid &= z <= max_depth_m
  fx, fy, cx, cy = [float(value) for value in intrinsics]
  points_camera = np.stack([(columns - cx) / fx * z, (rows - cy) / fy * z, z], axis=-1)[valid]
  rotation, translation = extrinsics_w2c[:3, :3], extrinsics_w2c[:3, 3]
  points_world = (points_camera - translation) @ rotation
  scale_h, scale_w = image.shape[0] / height, image.shape[1] / width
  color_rows = np.clip((rows * scale_h).astype(np.int64), 0, image.shape[0] - 1)
  color_columns = np.clip((columns * scale_w).astype(np.int64), 0, image.shape[1] - 1)
  return points_world.astype(np.float32), image[color_rows, color_columns][valid]


@dataclass(eq=False)
class View:
  index: int
  intrinsics: np.ndarray
  extrinsics_w2c: np.ndarray
  visibility: np.ndarray
  jpegs: np.ndarray
  path: Path

  def image(self, frame):
    return decode_jpeg(self.jpegs[int(frame)])

  def depth(self, frame):
    path = self.path / DEPTH
    if not path.exists():
      return None
    return np.asarray(np.load(path, mmap_mode="r")[int(frame)], dtype=np.float32)

  def foreground_mask(self, frame):
    path = self.path / MASK
    if not path.exists():
      return None
    return np.asarray(np.load(path, mmap_mode="r")[int(frame)])

  @functools.cached_property
  def centers(self):
    return camera_centers(self.extrinsics_w2c)

  @functools.cached_property
  def camera_motion_m(self):
    return float(np.linalg.norm(self.centers - self.centers[0], axis=-1).max())

  @property
  def kind(self):
    return f"wrist, moves {self.camera_motion_m:.2f}m" if self.index == 0 else "fixed exterior"

  @functools.cached_property
  def image_hw(self):
    return self.image(0).shape[:2]


@dataclass(eq=False)
class Episode:
  name: str
  tracks_xyz: np.ndarray
  queries_xytv: np.ndarray
  views: list

  @property
  def num_frames(self):
    return self.tracks_xyz.shape[0]

  @property
  def num_tracks(self):
    return self.tracks_xyz.shape[1]

  @property
  def num_views(self):
    return len(self.views)

  @property
  def query_v(self):
    return self.queries_xytv[:, 3].astype(int)

  @functools.cached_property
  def visibility(self):
    return np.stack([view.visibility for view in self.views], axis=-1)

  @functools.lru_cache(maxsize=8)
  def project(self, view):
    data = self.views[view]
    return project_tracks(self.tracks_xyz, data.intrinsics, data.extrinsics_w2c)


@functools.lru_cache(maxsize=4)
def load_episode(name, root=None):
  root = Path(root) if root is not None else find_dataset()
  episode_dir = root / name
  views = []
  for index in sorted(int(p.name) for p in episode_dir.iterdir() if p.is_dir() and p.name.isdigit()):
    path = episode_dir / str(index)
    views.append(
      View(
        index=index,
        intrinsics=np.load(path / INTRINSICS).astype(np.float32),
        extrinsics_w2c=np.load(path / EXTRINSICS).astype(np.float32),
        visibility=np.load(path / VISIBILITY),
        jpegs=np.load(path / IMAGES, allow_pickle=True),
        path=path,
      )
    )
  return Episode(
    name=name,
    tracks_xyz=np.load(episode_dir / TRACKS),
    queries_xytv=np.load(episode_dir / QUERIES),
    views=views,
  )
