"""Per-view 2D point tracking with selectable tracker model.

Extracts grid-based query points at frame 0 and tracks them independently
per camera view.  Supports two tracker backends:

  - **CoTracker3** (offline, Meta): dense grid tracking at native resolution.
  - **TAPNext++** (online, DeepMind): frame-by-frame tracking at 512×512.

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
  """TAPNext++ online frame-by-frame tracking at 512×512."""

  def __init__(self, device, ckpt_path=None):
    """Initialize TAPNext++ model.

    Args:
      device: Torch device.
      ckpt_path: Path to checkpoint. If None, downloads from GCS
          to third_party/tapnext_weights/.
    """
    # Ensure tapnet is importable
    try:
      from tapnet.tapnextpp.votsp2026.model import TAPNextPP
    except ImportError:
      print("  ⚠️ tapnet not installed, installing from GitHub...")
      os.system("pip install -q git+https://github.com/google-deepmind/tapnet.git")
      os.system("pip install -q git+https://github.com/google-deepmind/recurrentgemma.git@main")
      from tapnet.tapnextpp.votsp2026.model import TAPNextPP

    if ckpt_path is None:
      weights_dir = os.path.join(
          os.path.dirname(os.path.abspath(__file__)),
          "third_party", "tapnext_weights")
      os.makedirs(weights_dir, exist_ok=True)
      ckpt_path = os.path.join(weights_dir, "tapnextpp_512.ckpt")
      if not os.path.exists(ckpt_path):
        print(f"  ⬇️  Downloading TAPNext++ 512 checkpoint...")
        os.system(
            f"wget -q -O '{ckpt_path}' "
            "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt"
        )

    self.model_resolution = 512
    self.model = TAPNextPP.from_checkpoint(
        ckpt_path, device=str(device),
        input_resolution=self.model_resolution,
    )
    self.device = device
    self.name = "TAPNext++"
    print(f"  ✅ {self.name} loaded on {device}"
          f" (resolution={self.model_resolution})")

  @torch.no_grad()
  def track(self, video_rgb, grid_size=30):
    """Run online frame-by-frame tracking on a full video.

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
    res = self.model_resolution

    # *** IMPORTANT: TAPNext++ inner model always works in [0, 256]
    # coordinate space regardless of input resolution (256 or 512).
    # Query points must be in [0, 256] and outputs come back in [0, 256].
    INTERNAL_RES = 256.0
    scale_x_to_model = INTERNAL_RES / W   # original → model [0, 256]
    scale_y_to_model = INTERNAL_RES / H
    scale_x_from_model = W / INTERNAL_RES  # model [0, 256] → original
    scale_y_from_model = H / INTERNAL_RES

    # Generate query points in [0, 256] model coordinate space
    queries_orig = generate_grid_queries(H, W, grid_size)  # (N, 2) x,y
    q_model_x = queries_orig[:, 0] * scale_x_to_model  # → [0, 256)
    q_model_y = queries_orig[:, 1] * scale_y_to_model

    # TAPNext++ query format: (B, N, 3) with (t, y, x) — note y,x order!
    q_points = torch.zeros((1, N, 3), dtype=torch.float32, device=self.device)
    q_points[0, :, 0] = 0  # query frame = 0
    q_points[0, :, 1] = torch.from_numpy(q_model_y).float()
    q_points[0, :, 2] = torch.from_numpy(q_model_x).float()

    # Resize video to model input resolution
    video_tensor = (
        torch.from_numpy(video_rgb).float()
        .permute(0, 3, 1, 2)  # (T, 3, H, W)
    )
    video_resized = torch.nn.functional.interpolate(
        video_tensor, size=(res, res), mode='bilinear', align_corners=False,
    )
    video_batch = video_resized[None].to(self.device)  # (1, T, 3, res, res)

    # Extract inner model
    inner_model = (self.model._model
                   if hasattr(self.model, '_model') else self.model)

    # Frame-by-frame online tracking
    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=True):
      pred_tracks_list, pred_vis_list = [], []

      # First frame: initialize with query points
      curr_tracks, _, curr_vis_logits, state = inner_model(
          video=video_batch[:, :1], query_points=q_points,
      )
      pred_tracks_list.append(curr_tracks.cpu())
      pred_vis_list.append((curr_vis_logits > 0).cpu())

      # Subsequent frames: online tracking
      for t in range(1, T):
        curr_tracks, _, curr_vis_logits, state = inner_model(
            video=video_batch[:, t:t+1], state=state,
        )
        pred_tracks_list.append(curr_tracks.cpu())
        pred_vis_list.append((curr_vis_logits > 0).cpu())

    # Concatenate: each element is (1, 1, N, 2) → cat → (1, T, N, 2)
    tracks_model = torch.cat(pred_tracks_list, dim=1)  # (1, T, N, 2)
    vis_model = torch.cat(pred_vis_list, dim=1)         # (1, T, N, 1) or (1,T,N)

    # TAPNext++ tracks are in (y, x) order in [0, 256] model coords
    # Convert to (x, y) in original pixel resolution
    tracks_np = tracks_model[0].float().numpy()  # (T, N, 2) in [0,256] (y,x)
    tracks_out = np.zeros((T, N, 2), dtype=np.float32)
    tracks_out[:, :, 0] = tracks_np[:, :, 1] * scale_x_from_model  # x
    tracks_out[:, :, 1] = tracks_np[:, :, 0] * scale_y_from_model  # y

    vis_out = vis_model[0].squeeze(-1).bool().numpy()  # (T, N)

    del video_tensor, video_resized, video_batch
    del tracks_model, vis_model, state
    torch.cuda.empty_cache()

    return tracks_out, vis_out


# ===========================================================================
# Unified interface
# ===========================================================================

def init_tracker(method, device, ckpt_path=None):
  """Initialize a tracker backend.

  Args:
    method: "cotracker" or "tapnext".
    device: Torch device.
    ckpt_path: Optional checkpoint path (for tapnext).

  Returns:
    Tracker backend instance with a .track(video_rgb, grid_size) method.
  """
  print(f"🚀 Initializing 2D tracker: {method}")
  if method == "cotracker":
    return CoTrackerBackend(device)
  elif method == "tapnext":
    return TAPNextBackend(device, ckpt_path=ckpt_path)
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
  parser.add_argument("--method", type=str, choices=["cotracker", "tapnext"],
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
