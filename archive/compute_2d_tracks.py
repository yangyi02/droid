"""Per-view 2D point tracking with selectable tracker model.

Extracts grid-based query points at frame 0 and tracks them independently
per camera view.  Supports three tracker backends:

  - **CoTracker3** (offline, Meta): dense grid tracking at native resolution.
  - **TAPNext++** (online, DeepMind): frame-by-frame tracking at 512×512.
  - **AllTracker** (offline, Harley et al.): dense flow-based tracking.

Output format (per camera, stored in scene_constants):
  tracks_2d : np.float32 (T, N, 2)   pixel coordinates (x, y)
  vis_2d    : np.bool_   (T, N)       visibility mask

Usage (standalone):
  python compute_2d_tracks.py --method cotracker --grid_size 30
  python compute_2d_tracks.py --method tapnext --grid_size 30

Usage (library, from pipeline.ipynb):
  from compute_2d_tracks import init_tracker, run_2d_tracking
  tracker = init_tracker("cotracker", device)
  scene_constants = run_2d_tracking(tracker, scene_constants, device,
                                    grid_size=30)
"""

import argparse
import os
import sys

import numpy as np
import torch

from core.io import get_accelerator, load_depth_data


# ===========================================================================
# Grid query point generation
# ===========================================================================

def generate_grid_queries(h, w, grid_size=30):
  """Generate a uniform grid of (x, y) query points at native resolution.

  Args:
    h: Image height.
    w: Image width.
    grid_size: Number of points per axis (total = grid_size²).

  Returns:
    queries: np.float32 (N, 2) with (x, y) pixel coordinates.
  """
  xs = np.linspace(0, w - 1, grid_size, dtype=np.float32)
  ys = np.linspace(0, h - 1, grid_size, dtype=np.float32)
  xx, yy = np.meshgrid(xs, ys)
  queries = np.stack([xx.ravel(), yy.ravel()], axis=-1)
  return queries


# ===========================================================================
# Tracking Backend Interface
# ===========================================================================

class TrackingBackend:
  """Base class for all tracking backends."""

  def set_video(self, video_rgb):
    """Set the video to track on. Preprocesses and uploads to device."""
    raise NotImplementedError

  def track_grid(self, grid_size=30):
    """Run dense grid tracking starting at t=0 on the current video."""
    raise NotImplementedError

  def track_queries(self, queries):
    """Track arbitrary query points (t, x, y) on the current video."""
    raise NotImplementedError

  def clear_video(self):
    """Free video-related resources."""
    pass

  def track(self, video_rgb, grid_size=30):
    """Backward-compatible convenience wrapper."""
    self.set_video(video_rgb)
    tracks, vis = self.track_grid(grid_size)
    self.clear_video()
    return tracks, vis


# ===========================================================================
# CoTracker3 backend
# ===========================================================================

class CoTrackerBackend(TrackingBackend):
  """CoTracker3 offline dense tracking at native resolution."""

  def __init__(self, device):
    vendor_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "third_party")
    cotracker_path = os.path.join(vendor_dir, "co-tracker")
    if cotracker_path not in sys.path:
      sys.path.append(cotracker_path)

    from cotracker.predictor import CoTrackerPredictor

    self.model = CoTrackerPredictor(
        checkpoint=os.path.join(
            cotracker_path, "weights/cotracker3_offline.pth"),
    ).to(device)
    self.device = device
    self.name = "CoTracker3"
    print(f"  ✅ {self.name} loaded on {device}")
    self.video_tensor = None

  def set_video(self, video_rgb):
    self.video_tensor = (
        torch.from_numpy(video_rgb)
        .permute(0, 3, 1, 2)[None].float().to(self.device)
    )

  def track_grid(self, grid_size=30):
    if self.video_tensor is None:
      raise ValueError("Must call set_video first")
    with torch.no_grad():
      pred_tracks, pred_vis = self.model(
          self.video_tensor, grid_size=grid_size, grid_query_frame=0,
          backward_tracking=False,
      )
    return pred_tracks[0].cpu().numpy(), pred_vis[0].cpu().numpy() > 0.5

  def track_queries(self, queries):
    if self.video_tensor is None:
      raise ValueError("Must call set_video first")
    queries_t = torch.tensor(
        queries, dtype=torch.float32, device=self.device)[None]
    with torch.no_grad():
      pred_tracks, pred_vis = self.model(
          self.video_tensor, queries=queries_t, backward_tracking=True)
    return pred_tracks[0].cpu().numpy(), pred_vis[0].cpu().numpy() > 0.5

  def clear_video(self):
    if self.video_tensor is not None:
      del self.video_tensor
      self.video_tensor = None
      torch.cuda.empty_cache()


# ===========================================================================
# TAPNext++ backend
# ===========================================================================

class TAPNextBackend(TrackingBackend):
  """TAPNext++ online frame-by-frame tracking at 512×512.

  Uses the high-level ``TAPNextPP.track_frame()`` API which handles all
  preprocessing (BGR→RGB, resize, normalize to [-1,1], channels-last) and
  coordinate conversion (model [0,256] ↔ display pixels) internally.
  """

  def __init__(self, device, ckpt_path=None):
    """Initialize TAPNext++ model.

    Args:
      device: Torch device.
      ckpt_path: Path to checkpoint. If None, defaults to
          third_party/tapnext_weights/tapnextpp_512.ckpt.
    """
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP

    if ckpt_path is None:
      ckpt_path = os.path.join(
          os.path.dirname(os.path.abspath(__file__)),
          "third_party", "tapnext_weights", "tapnextpp_512.ckpt")

    if not os.path.exists(ckpt_path):
      raise FileNotFoundError(
          f"TAPNext++ checkpoint not found at {ckpt_path}. "
          "Please run setup.sh first."
      )

    self.model_resolution = 512
    self.model = TAPNextPP.from_checkpoint(
        ckpt_path, device=str(device),
        input_resolution=self.model_resolution,
    )
    self.device = device
    self.name = "TAPNext++"
    print(f"  ✅ {self.name} loaded on {device} (resolution={self.model_resolution})")
    self.video_rgb = None

  def set_video(self, video_rgb):
    self.video_rgb = video_rgb
    self.T, self.H, self.W, _ = video_rgb.shape

  def track_grid(self, grid_size=30):
    if self.video_rgb is None:
      raise ValueError("Must call set_video first")
    N = grid_size * grid_size
    queries_xy = generate_grid_queries(self.H, self.W, grid_size)  # (N, 2)

    tracks_out = np.zeros((self.T, N, 2), dtype=np.float32)
    vis_out = np.zeros((self.T, N), dtype=bool)
    state = None

    for t in range(self.T):
      frame_bgr = self.video_rgb[t, :, :, ::-1].copy()

      if t == 0:
        positions_xy, visible, state = self.model.track_frame(
            frame_bgr, query_points_xy=queries_xy,
        )
      else:
        positions_xy, visible, state = self.model.track_frame(
            frame_bgr, state=state,
        )

      tracks_out[t] = positions_xy
      vis_out[t] = visible

    return tracks_out, vis_out

  def track_queries(self, queries):
    if self.video_rgb is None:
      raise ValueError("Must call set_video first")
    N = len(queries)
    if N == 0:
      return np.zeros((self.T, 0, 2), dtype=np.float32), np.zeros((self.T, 0), dtype=bool)

    t_k = int(queries[0, 0])
    assert np.all(queries[:, 0] == t_k), "All queries in batch must have same start frame"
    queries_xy = queries[:, 1:3]

    tracks_out = np.full((self.T, N, 2), np.nan, dtype=np.float32)
    vis_out = np.zeros((self.T, N), dtype=bool)

    # 1. Forward tracking (t_k -> T-1)
    if t_k < self.T:
      state_f = None
      for t in range(t_k, self.T):
        frame_bgr = self.video_rgb[t, :, :, ::-1].copy()
        if t == t_k:
          pos, vis, state_f = self.model.track_frame(
              frame_bgr, query_points_xy=queries_xy)
        else:
          pos, vis, state_f = self.model.track_frame(frame_bgr, state=state_f)
        tracks_out[t] = pos
        vis_out[t] = vis

    # 2. Backward tracking (t_k-1 -> 0) - running the recurrent state backward
    if t_k > 0:
      state_b = None
      for t in range(t_k, -1, -1):
        frame_bgr = self.video_rgb[t, :, :, ::-1].copy()
        if t == t_k:
          pos, vis, state_b = self.model.track_frame(
              frame_bgr, query_points_xy=queries_xy)
        else:
          pos, vis, state_b = self.model.track_frame(frame_bgr, state=state_b)
        tracks_out[t] = pos
        vis_out[t] = vis

    return tracks_out, vis_out

  def clear_video(self):
    if self.video_rgb is not None:
      del self.video_rgb
      self.video_rgb = None
      torch.cuda.empty_cache()


# ===========================================================================
# Namespace isolation helper
# ===========================================================================

def _import_from_vendor(vendor_path, import_fn, shadow_packages=("utils",)):
  """Import third-party modules whose package names collide with ours.

  Temporarily evicts ``shadow_packages`` (e.g. ``utils``, ``model``) from
  ``sys.modules`` so that vendor code can load *its own* versions, then
  restores ours afterwards. Supports namespace packages (no __init__.py).

  Args:
    vendor_path: Absolute path to the vendor repo root (added to sys.path).
    import_fn:   Callable that performs the actual imports and returns results.
    shadow_packages: Package names that collide.

  Returns:
    Whatever ``import_fn`` returns.
  """
  import types

  if vendor_path not in sys.path:
    sys.path.insert(0, vendor_path)

  # Save & remove our versions of the conflicting packages
  saved = {}
  for k in list(sys.modules):
    for pkg in shadow_packages:
      if k == pkg or k.startswith(pkg + "."):
        saved[k] = sys.modules.pop(k)

  # Inject placeholders for namespace packages (no __init__.py)
  # This forces python to resolve imports within the vendor path directory
  # instead of falling back to our regular packages (which have __init__.py
  # and would otherwise take precedence during sys.path search).
  for pkg in shadow_packages:
    pkg_dir = os.path.join(vendor_path, pkg)
    if os.path.isdir(pkg_dir) and not os.path.exists(os.path.join(pkg_dir, "__init__.py")):
      mod = types.ModuleType(pkg)
      mod.__path__ = [pkg_dir]
      sys.modules[pkg] = mod

  try:
    result = import_fn()
  finally:
    # Remove the vendor's versions that were just loaded
    vendor_mods = {}
    for k in list(sys.modules):
      for pkg in shadow_packages:
        if k == pkg or k.startswith(pkg + "."):
          vendor_mods[k] = sys.modules.pop(k)
    # Restore ours
    sys.modules.update(saved)
    # Stash vendor's under a private prefix so they stay importable
    tag = os.path.basename(vendor_path)
    for k, v in vendor_mods.items():
      sys.modules[f"_{tag}_{k}"] = v

  return result


# ===========================================================================
# AllTracker backend
# ===========================================================================

class AllTrackerBackend(TrackingBackend):
  """AllTracker dense flow-based tracking.

  Computes all-pixel flow fields from a query frame to every other frame,
  then samples the flow at the requested grid query locations.
  """

  def __init__(self, device, ckpt_path=None):
    vendor_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "third_party")
    alltracker_path = os.path.join(vendor_dir, "alltracker")

    # AllTracker has its own utils.misc / utils.basic that collide with ours
    Net = _import_from_vendor(
        alltracker_path,
        lambda: __import__("nets.alltracker", fromlist=["Net"]).Net,
        shadow_packages=("utils",),
    )

    if ckpt_path is None:
      weights_dir = os.path.join(alltracker_path, "weights")
      os.makedirs(weights_dir, exist_ok=True)
      ckpt_path = os.path.join(weights_dir, "alltracker.pth")
      if not os.path.exists(ckpt_path):
        print("  📥 Downloading AllTracker weights from HuggingFace...")
        hf_url = ("https://huggingface.co/aharley/alltracker/resolve/main/"
                  "alltracker.pth")
        torch.hub.download_url_to_file(hf_url, ckpt_path)

    # Pretrained checkpoint uses seqlen=16 for time embeddings.
    # The model processes longer videos using sliding window / padding of size seqlen.
    model = Net(seqlen=16)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    self.model = model
    self.device = device
    self.name = "AllTracker"
    print(f"  ✅ {self.name} loaded on {device}")
    self.video_resized = None

  def set_video(self, video_rgb):
    import torch.nn.functional as F
    self.T, self.H_orig, self.W_orig, _ = video_rgb.shape
    H_model, W_model = 384, 512

    self.scale_x = W_model / self.W_orig
    self.scale_y = H_model / self.H_orig

    video_t = torch.from_numpy(video_rgb).float().permute(0, 3, 1, 2)
    self.video_resized = F.interpolate(
        video_t, size=(H_model, W_model), mode='bilinear', align_corners=False)
    self.video_resized = self.video_resized[None].to(self.device)

  def track_grid(self, grid_size=30):
    if self.video_resized is None:
      raise ValueError("Must call set_video first")
    
    H_model, W_model = 384, 512
    queries_orig = generate_grid_queries(self.H_orig, self.W_orig, grid_size)
    N = queries_orig.shape[0]

    queries_model = queries_orig.copy()
    queries_model[:, 0] *= self.scale_x
    queries_model[:, 1] *= self.scale_y
    queries_model_px = np.clip(np.round(queries_model).astype(int), 0, [W_model - 1, H_model - 1])

    if self.T > 128:
      full_flows, full_visconfs, _, _ = self.model.forward_sliding(self.video_resized, is_training=False)
    else:
      full_flows, full_visconfs, _, _ = self.model(self.video_resized, is_training=False)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H_model, device=full_flows.device, dtype=torch.float32),
        torch.arange(W_model, device=full_flows.device, dtype=torch.float32),
        indexing='ij'
    )
    grid_xy = torch.stack([grid_x, grid_y], dim=0)[None, None]
    traj_maps_model = full_flows + grid_xy

    qx = queries_model_px[:, 0]
    qy = queries_model_px[:, 1]
    trajs_model_sampled = traj_maps_model[0, :, :, qy, qx].permute(0, 2, 1)

    tracks_out = trajs_model_sampled.cpu().numpy()
    tracks_out[..., 0] /= self.scale_x
    tracks_out[..., 1] /= self.scale_y

    visconfs_sampled = full_visconfs[0, :, :, qy, qx].permute(0, 2, 1)
    vis_prob = visconfs_sampled[..., 0] * visconfs_sampled[..., 1]
    vis_out = (vis_prob.cpu().numpy() > 0.6)

    return tracks_out, vis_out

  def track_queries(self, queries):
    if self.video_resized is None:
      raise ValueError("Must call set_video first")
    
    N = len(queries)
    if N == 0:
      return np.zeros((self.T, 0, 2), dtype=np.float32), np.zeros((self.T, 0), dtype=bool)

    t_k = int(queries[0, 0])
    assert np.all(queries[:, 0] == t_k), "All queries in batch must have same start frame"
    queries_orig = queries[:, 1:3]

    queries_model = queries_orig.copy()
    queries_model[:, 0] *= self.scale_x
    queries_model[:, 1] *= self.scale_y
    queries_model_px = np.clip(np.round(queries_model).astype(int), 0, [512 - 1, 384 - 1])
    qx = queries_model_px[:, 0]
    qy = queries_model_px[:, 1]

    tracks_out = np.full((self.T, N, 2), np.nan, dtype=np.float32)
    vis_out = np.zeros((self.T, N), dtype=bool)

    H_model, W_model = 384, 512

    def _run_slice_and_sample(video_slice, start_t, step):
      T_slice = video_slice.shape[1]
      if T_slice <= 1:
        t_indices = [start_t]
        tracks_slice = np.repeat(queries_orig[None], T_slice, axis=0)
        vis_slice = np.ones((T_slice, N), dtype=bool)
        return t_indices, tracks_slice, vis_slice

      with torch.no_grad():
        if T_slice > 128:
          flows, visconfs, _, _ = self.model.forward_sliding(video_slice, is_training=False)
        else:
          flows, visconfs, _, _ = self.model(video_slice, is_training=False)

      grid_y, grid_x = torch.meshgrid(
          torch.arange(H_model, device=flows.device, dtype=torch.float32),
          torch.arange(W_model, device=flows.device, dtype=torch.float32),
          indexing='ij'
      )
      grid_xy = torch.stack([grid_x, grid_y], dim=0)[None, None]
      traj_maps = flows + grid_xy

      trajs_sampled = traj_maps[0, :, :, qy, qx].permute(0, 2, 1)
      tracks_slice = trajs_sampled.cpu().numpy()
      tracks_slice[..., 0] /= self.scale_x
      tracks_slice[..., 1] /= self.scale_y

      visconfs_sampled = visconfs[0, :, :, qy, qx].permute(0, 2, 1)
      vis_prob = visconfs_sampled[..., 0] * visconfs_sampled[..., 1]
      vis_slice = (vis_prob.cpu().numpy() > 0.6)

      t_indices = list(range(start_t, start_t + step * T_slice, step))
      return t_indices, tracks_slice, vis_slice

    # 1. Forward tracking (t_k -> T-1)
    if t_k < self.T:
      video_forward = self.video_resized[:, t_k:]
      t_inds, tr_f, vis_f = _run_slice_and_sample(video_forward, t_k, 1)
      for idx, t in enumerate(t_inds):
        tracks_out[t] = tr_f[idx]
        vis_out[t] = vis_f[idx]

    # 2. Backward tracking (t_k-1 -> 0)
    if t_k > 0:
      video_backward = self.video_resized[:, t_k::-1]
      t_inds, tr_b, vis_b = _run_slice_and_sample(video_backward, t_k, -1)
      for idx, t in enumerate(t_inds):
        tracks_out[t] = tr_b[idx]
        vis_out[t] = vis_b[idx]

    return tracks_out, vis_out

  def clear_video(self):
    if self.video_resized is not None:
      del self.video_resized
      self.video_resized = None
      torch.cuda.empty_cache()




# ===========================================================================
# Unified interface
# ===========================================================================

def init_tracker(method, device, ckpt_path=None):
  """Initialize a tracker backend.

  Args:
    method: "cotracker", "tapnext", or "alltracker".
    device: Torch device.
    ckpt_path: Optional checkpoint path.

  Returns:
    Tracker backend instance with a .track(video_rgb, grid_size) method.
  """
  print(f"🚀 Initializing 2D tracker: {method}")
  if method == "cotracker":
    return CoTrackerBackend(device)
  elif method == "tapnext":
    return TAPNextBackend(device, ckpt_path=ckpt_path)
  elif method == "alltracker":
    return AllTrackerBackend(device, ckpt_path=ckpt_path)
  else:
    raise ValueError(f"Unknown tracker method: {method}")


def run_2d_tracking(tracker, scene_constants, device, grid_size=30):
  """Run per-view 2D tracking on all cameras.

  Populates scene_constants['camera'][cam_id]['tracks_2d'] and
  scene_constants['camera'][cam_id]['vis_2d'] for each camera.

  Args:
    tracker: Backend instance from init_tracker().
    scene_constants: Scene data dict (requires 'video_rgb' per camera).
    device: Torch device (unused, kept for API consistency).
    grid_size: Grid density per axis.

  Returns:
    scene_constants with tracks_2d / vis_2d populated.
  """
  print(f"\n{'=' * 60}")
  print(f"🎯 Per-View 2D Tracking [{tracker.name}] (grid={grid_size})")
  print(f"{'=' * 60}")

  for cam_id in scene_constants['camera']:
    cam_data = scene_constants['camera'][cam_id]
    if 'video_rgb' not in cam_data:
      print(f"  ⚠️ [{cam_id}] No video_rgb, skipping.")
      continue

    video_rgb = cam_data['video_rgb']
    print(f"  📷 [{cam_id}] {video_rgb.shape[0]} frames × "
          f"{video_rgb.shape[1]}×{video_rgb.shape[2]}")

    tracks, vis = tracker.track(video_rgb, grid_size=grid_size)

    cam_data['tracks_2d'] = tracks
    cam_data['vis_2d'] = vis

    T, N, _ = tracks.shape
    vis_rate = vis.mean() * 100
    print(f"     ✅ {N} tracks × {T} frames "
          f"(avg visibility: {vis_rate:.1f}%)")

  return scene_constants


def export_2d_tracks(scene_constants, export_root=None):
  """Export per-camera 2D tracks and visibility to disk.

  Args:
    scene_constants: Scene data dict with tracks_2d / vis_2d.
    export_root: Output directory root. Defaults to
        ~/droid_data/output/mv-tap/droid/tracks_2d.
  """
  if export_root is None:
    export_root = os.path.expanduser(
        "~/droid_data/output/mv-tap/droid/tracks_2d")

  ep_id = scene_constants['meta']['episode_id']
  ep_dir = os.path.join(os.path.expanduser(export_root), ep_id)

  for cam_id in scene_constants['camera']:
    cam_data = scene_constants['camera'][cam_id]
    if 'tracks_2d' not in cam_data:
      continue

    cam_dir = os.path.join(ep_dir, cam_id)
    os.makedirs(cam_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(cam_dir, "tracks_2d.npz"),
        tracks_2d=cam_data['tracks_2d'],
        vis_2d=cam_data['vis_2d'],
    )

  print(f"  💾 2D tracks exported to {ep_dir}")


# ===========================================================================
# Standalone CLI
# ===========================================================================

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="DROID: Per-view 2D Point Tracking")
  parser.add_argument("--method", type=str,
                       choices=["cotracker", "tapnext", "alltracker"],
                       default="cotracker",
                       help="Tracker model to use")
  parser.add_argument("--grid_size", type=int, default=30,
                       help="Grid density per axis (total = grid_size²)")
  parser.add_argument("--depth_root", type=str,
                       default="~/droid_data/output/mv-tap/droid/depth",
                       help="Root directory of depth outputs")
  parser.add_argument("--export_root", type=str,
                       default="~/droid_data/output/mv-tap/droid/tracks_2d",
                       help="Output directory for 2D tracks")
  parser.add_argument("--limit", type=int, default=-1,
                       help="Limit total number of episodes to process")
  parser.add_argument("--ckpt", type=str, default=None,
                       help="Override checkpoint path (for tapnext)")
  args = parser.parse_args()

  device = get_accelerator()
  tracker = init_tracker(args.method, device, ckpt_path=args.ckpt)

  # Discover episodes
  depth_abs = os.path.abspath(os.path.expanduser(args.depth_root))
  available_eps = sorted([
      d for d in os.listdir(depth_abs)
      if os.path.isdir(os.path.join(depth_abs, d))
  ])
  if args.limit > 0:
    available_eps = available_eps[:args.limit]

  print(f"📋 Processing {len(available_eps)} episodes with {args.method}")

  for idx, ep_id in enumerate(available_eps):
    print(f"\n🎬 [{idx + 1}/{len(available_eps)}] Episode: {ep_id}")
    try:
      scene_constants = load_depth_data(ep_id, args.depth_root)
      scene_constants = run_2d_tracking(
          tracker, scene_constants, device, grid_size=args.grid_size)
      export_2d_tracks(scene_constants, export_root=args.export_root)
      print(f"  ✅ Episode {ep_id} done.")
    except Exception as e:
      print(f"  ❌ Episode {ep_id} failed: {e}")
      import traceback
      traceback.print_exc()
    finally:
      scene_constants = None
      torch.cuda.empty_cache()

  print(f"\n🎉 2D tracking complete!")
