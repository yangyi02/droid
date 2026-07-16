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
# CoTracker3 backend
# ===========================================================================

class CoTrackerBackend:
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

  @torch.no_grad()
  def track(self, video_rgb, grid_size=30):
    """Run dense grid tracking on a full video.

    Args:
      video_rgb: np.uint8 (T, H, W, 3) RGB video.
      grid_size: Grid density per axis.

    Returns:
      tracks: np.float32 (T, N, 2) pixel coordinates (x, y).
      vis:    np.bool_   (T, N)    visibility mask.
    """
    video_tensor = (
        torch.from_numpy(video_rgb)
        .permute(0, 3, 1, 2)[None].float().to(self.device)
    )
    pred_tracks, pred_vis = self.model(
        video_tensor, grid_size=grid_size, grid_query_frame=0,
        backward_tracking=False,
    )

    tracks = pred_tracks[0].cpu().numpy()        # (T, N, 2)
    vis = pred_vis[0].cpu().numpy() > 0.5         # (T, N)

    del video_tensor, pred_tracks, pred_vis
    torch.cuda.empty_cache()

    return tracks, vis


# ===========================================================================
# TAPNext++ backend
# ===========================================================================

class TAPNextBackend:
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

  @torch.no_grad()
  def track(self, video_rgb, grid_size=30):
    """Run online frame-by-frame tracking on a full video.

    Uses ``TAPNextPP.track_frame()`` which expects BGR uint8 frames and
    returns (x, y) positions in display pixel coordinates.

    Args:
      video_rgb: np.uint8 (T, H, W, 3) RGB video.
      grid_size: Grid density per axis.

    Returns:
      tracks: np.float32 (T, N, 2) pixel coordinates (x, y) at
          **original** resolution.
      vis:    np.bool_   (T, N)    visibility mask.
    """
    T, H, W, _ = video_rgb.shape
    N = grid_size * grid_size

    # Generate query points in original pixel coordinates (x, y)
    queries_xy = generate_grid_queries(H, W, grid_size)  # (N, 2)

    tracks_out = np.zeros((T, N, 2), dtype=np.float32)
    vis_out = np.zeros((T, N), dtype=bool)
    state = None

    for t in range(T):
      # track_frame() expects BGR uint8 frames — convert from RGB
      frame_bgr = video_rgb[t, :, :, ::-1].copy()

      if t == 0:
        # First frame: initialize tracking with query points
        positions_xy, visible, state = self.model.track_frame(
            frame_bgr, query_points_xy=queries_xy,
        )
      else:
        # Subsequent frames: pass recurrent state
        positions_xy, visible, state = self.model.track_frame(
            frame_bgr, state=state,
        )

      tracks_out[t] = positions_xy  # (N, 2) in display (x, y) pixels
      vis_out[t] = visible           # (N,) bool

    torch.cuda.empty_cache()
    return tracks_out, vis_out


# ===========================================================================
# Namespace isolation helper
# ===========================================================================

def _import_from_vendor(vendor_path, import_fn, shadow_packages=("utils",)):
  """Import third-party modules whose package names collide with ours.

  Temporarily evicts ``shadow_packages`` (e.g. ``utils``, ``model``) from
  ``sys.modules`` so that vendor code can load *its own* versions, then
  restores ours afterwards.

  Args:
    vendor_path: Absolute path to the vendor repo root (added to sys.path).
    import_fn:   Callable that performs the actual imports and returns results.
    shadow_packages: Package names that collide.

  Returns:
    Whatever ``import_fn`` returns.
  """
  if vendor_path not in sys.path:
    sys.path.insert(0, vendor_path)

  # Save & remove our versions of the conflicting packages
  saved = {}
  for k in list(sys.modules):
    for pkg in shadow_packages:
      if k == pkg or k.startswith(pkg + "."):
        saved[k] = sys.modules.pop(k)

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

class AllTrackerBackend:
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

    model = Net(seqlen=128)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    self.model = model
    self.device = device
    self.name = "AllTracker"
    print(f"  ✅ {self.name} loaded on {device}")

  @torch.no_grad()
  def track(self, video_rgb, grid_size=30):
    """Run dense flow tracking, sampling at grid query points.

    Args:
      video_rgb: np.uint8 (T, H, W, 3) RGB video.
      grid_size: Grid density per axis.

    Returns:
      tracks: np.float32 (T, N, 2) pixel coordinates (x, y).
      vis:    np.bool_   (T, N)    visibility mask.
    """
    T, H, W, _ = video_rgb.shape
    queries_xy = generate_grid_queries(H, W, grid_size)  # (N, 2)
    N = queries_xy.shape[0]

    # AllTracker expects (B, T, C, H, W) float32 [0-255]
    video_t = (
        torch.from_numpy(video_rgb).float()
        .permute(0, 3, 1, 2)[None]  # (1, T, 3, H, W)
        .to(self.device)
    )

    # Model forward: outputs trajs (1, T, N_dense, 2) and vis (1, T, N_dense)
    # For grid queries, we pass query_points
    query_xy_t = torch.from_numpy(queries_xy).float().to(self.device)  # (N, 2)

    # AllTracker can take queries or run dense; use forward with query frame 0
    trajs, visibs = self.model(
        video_t, queries=query_xy_t[None], query_frame=0,
    )

    tracks = trajs[0].cpu().numpy()   # (T, N, 2)
    vis = visibs[0].cpu().numpy() > 0.5  # (T, N)

    del video_t, trajs, visibs
    torch.cuda.empty_cache()
    return tracks, vis




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
