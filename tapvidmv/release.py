import functools
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import core.geometry

MAX_DEPTH_M = 2.0

TRACKS = "tracks_xyz.npy"
QUERIES = "queries_xytv.npy"
IMAGES = "images_jpeg_bytes.npy"
INTRINSICS = "intrinsics.npy"
EXTRINSICS = "extrinsics_w2c.npy"
VISIBILITY = "visibility.npy"
DEPTH = "depth.npz"
MASK = "foreground_mask.npz"


def find_dataset():
  return Path(__file__).parent / "data" / "release"


def episode_names(root):
  root = Path(root)
  assert root.is_dir(), f"no export at {root}"
  return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / TRACKS).exists())


def decode_jpeg(raw):
  buffer = np.frombuffer(bytes(raw), dtype=np.uint8)
  return cv2.cvtColor(cv2.imdecode(buffer, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def unproject_depth(depth, image, K, T_cam2world, *, stride=6, max_depth_m=MAX_DEPTH_M):
  rows, cols = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
  z = depth[::stride, ::stride]
  valid = np.isfinite(z) & (z > 0.0) & (z <= max_depth_m if max_depth_m is not None else True)
  points = core.geometry.unproject_pixels(
    cols[valid].astype(np.float32), rows[valid].astype(np.float32), z[valid], K, T_cam2world
  )
  return points.astype(np.float32), image[rows[valid], cols[valid]]


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

  @functools.cached_property
  def K(self):
    fx, fy, cx, cy = self.intrinsics
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

  @functools.cached_property
  def T_cam2world(self):
    return np.linalg.inv(self.extrinsics_w2c).astype(np.float32)

  @functools.cached_property
  def depth_mm(self):
    path = self.path / DEPTH
    return np.load(path)["depth"] if path.exists() else None

  @functools.cached_property
  def masks(self):
    path = self.path / MASK
    return np.load(path)["mask"] if path.exists() else None

  def depth(self, frame):
    return None if self.depth_mm is None else self.depth_mm[int(frame)].astype(np.float32) / 1000.0

  def foreground_mask(self, frame):
    return None if self.masks is None else self.masks[int(frame)]

  @functools.cached_property
  def centers(self):
    return self.T_cam2world[:, :3, 3]

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
    return core.geometry.project_tracks(self.tracks_xyz, data.K, data.T_cam2world)


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
