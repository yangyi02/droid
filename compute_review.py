import os

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from absl import app
from ml_collections import config_flags

import config
import core.geometry
import core.io
import core.runner

JPEG_QUALITY = 95

ROBOT = [56, 189, 248]
STATIC = [250, 204, 21]
VISIBLE = [34, 220, 100]
NOT_VISIBLE = [255, 65, 65]
OUTSIDE = [90, 120, 255]
INSPECT = [255, 40, 235]
QUERY = [255, 205, 30]
WHITE = [255, 255, 255]

TRACK_RADIUS_M = 0.003
INSPECT_RADIUS_M = 0.006
RAY_RADIUS_M = 0.0015


class Review:
  def __init__(self, episode, poses, tracks):
    self.tracks_3d = tracks["tracks_3d"]
    self.vis = tracks["vis"]
    self.n_robot = tracks["n_robot"]
    self.queries = core.io.build_queries(tracks["uv"], tracks["query_view"], tracks["query_frame"])

    K, T_cam2world, image_wh = [], [], []
    for cam_id, cam_data in episode["camera"].items():
      height, width = cam_data["raw_depth"][0].shape
      K.append(cam_data["K"])
      T_cam2world.append(poses[cam_id]["extrinsics"][: self.n_frames])
      image_wh.append([width, height])

    self.K = np.asarray(K, dtype=np.float32)
    self.T_cam2world = np.asarray(T_cam2world, dtype=np.float32)
    self.image_wh = np.asarray(image_wh, dtype=np.int32)

  @property
  def n_frames(self):
    return len(self.tracks_3d)

  @property
  def n_points(self):
    return self.tracks_3d.shape[1]

  @property
  def n_views(self):
    return len(self.K)

  @property
  def track_colors(self):
    return np.where(np.arange(self.n_points)[:, None] < self.n_robot, ROBOT, STATIC).astype(np.uint8)


def inspect_tracks(review, n_inspect):
  """A plain random sample of the tracks, seeded so the same episode always shows the same ones.

  It used to prefer tracks whose visibility changes, on the grounds that a point nobody ever loses
  sight of has nothing to check. That is true for finding faults and wrong for judging the set: the
  sample it produced flipped three times as often as the background does, and the quarter of the
  background that never changes at all could not appear in it."""
  return sorted(np.random.default_rng(0).choice(review.n_points, min(n_inspect, review.n_points), replace=False).tolist())


def verdict(visible, inside, z, at_query):
  """What one camera says about the track on one frame."""
  status = "visible" if visible else "hidden"
  if z <= 0:
    status = "behind"
  elif not inside:
    status = "off-frame"
  if visible and not inside:
    status += "!"
  return status + "*" if at_query else status


def frame_status(review, view, t, track):
  """Whether the track lands inside this camera's image on this frame, and how far in front it is."""
  u, v, z = core.geometry.project_points(
    review.tracks_3d[t, track][None], review.K[view], review.T_cam2world[view, t]
  )
  return bool((core.geometry.in_frame(u, v, *review.image_wh[view]) & (z > 0))[0]), float(z[0])


def ray_color(review, view, t, track):
  inside, _ = frame_status(review, view, t, track)
  if not inside:
    return OUTSIDE
  return VISIBLE if review.vis[view, t, track] else NOT_VISIBLE


def query_cross(review, track):
  x, y = review.queries[track, :2] + 0.5
  return [[[x - 7, y], [x + 7, y]], [[x, y - 7], [x, y + 7]]]


def depth_cloud(depth, img_rgb, K, T_cam2world, stride, max_depth):
  strided = depth[::stride, ::stride]
  v, u = np.nonzero((strided > 0) & (strided <= max_depth))
  v, u = v * stride, u * stride
  points_world = core.geometry.unproject_pixels(
    u.astype(np.float32), v.astype(np.float32), depth[v, u], K, T_cam2world
  )
  return points_world.astype(np.float32), img_rgb[v, u]


def blueprint(review, inspect, episode_id, fps):
  center = review.tracks_3d[0, inspect[0]]
  extent = float(np.linalg.norm(np.ptp(review.tracks_3d.reshape(-1, 3), axis=0)))

  return rrb.Blueprint(
    rrb.Vertical(
      rrb.Spatial3DView(
        # The episode is in the title because two recordings served together are told apart by nothing
        # else on screen: they share an application id, and what is left is a random recording id.
        name=f"3D tracks — {episode_id}",
        origin="/",
        line_grid=False,
        background=rrb.Background(kind="GradientDark"),
        contents=["+ /scene/**", "+ /cameras/**", "+ /tracks", "+ /inspect/**"],
        eye_controls=rrb.EyeControls3D(
          # No tracking_entity: it would pin the orbit to one track for good, and switching which
          # track is shown does not move it. Double-click a point in the viewer to re-centre.
          kind="Orbital",
          look_target=center,
          position=center + np.array([1.2, -1.4, 0.9]) * max(extent * 0.4, 0.3),
          eye_up=[0, 0, 1],
        ),
        overrides={f"/inspect/{track}": rrb.EntityBehavior(visible=False) for track in inspect[1:]},
      ),
      rrb.Horizontal(
        *[
          rrb.Spatial2DView(
            name=f"Camera {view}",
            origin=f"/views/{view}",
            visual_bounds=rrb.VisualBounds2D(x_range=[0, int(wh[0])], y_range=[0, int(wh[1])]),
            overrides={f"/views/{view}/tracks": rrb.EntityBehavior(visible=False)},
          )
          for view, wh in enumerate(review.image_wh)
        ]
      ),
      row_shares=[0.70, 0.30],
    ),
    rrb.TimePanel(state="collapsed", timeline="frame", fps=fps, play_state="paused", loop_mode="all"),
    rrb.BlueprintPanel(state="collapsed"),
    rrb.SelectionPanel(state="collapsed"),
    collapse_panels=False,
  )


def log_cameras(rec, review, episode, cfg):
  rec.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

  cameras = list(episode["camera"].values())
  for view in range(review.n_views):
    K = review.K[view]
    rec.log(
      f"/cameras/{view}/pose/image",
      rr.Pinhole(
        focal_length=[K[0, 0], K[1, 1]],
        principal_point=[K[0, 2] + 0.5, K[1, 2] + 0.5],
        resolution=review.image_wh[view],
        camera_xyz=rr.ViewCoordinates.RDF,
        image_plane_distance=0.08,
      ),
      static=True,
    )

  n_scene_points, n_off_image = 0, 0
  for t in range(review.n_frames):
    rec.set_time("frame", sequence=t)
    for view, cam_data in enumerate(cameras):
      K = review.K[view]
      img_rgb = cam_data["video_rgb"][t]
      T_cam2world = review.T_cam2world[view, t]

      rec.log(f"/views/{view}/rgb", rr.Image(img_rgb).compress(jpeg_quality=JPEG_QUALITY))
      rec.log(
        f"/cameras/{view}/pose",
        rr.Transform3D(translation=T_cam2world[:3, 3], mat3x3=T_cam2world[:3, :3]),
      )

      u, v, z = core.geometry.project_points(review.tracks_3d[t], K, T_cam2world)
      inside = core.geometry.in_frame(u, v, *review.image_wh[view]) & (z > 0)
      visible = review.vis[view, t]
      n_off_image += int((visible & ~inside).sum())
      rec.log(
        f"/views/{view}/tracks",
        rr.Points2D(
          np.stack([u[inside], v[inside]], axis=1) + 0.5,
          colors=review.track_colors[inside],
          radii=rr.Radius.ui_points(3),
          labels=[str(point) for point in np.flatnonzero(inside)],
          show_labels=False,
          draw_order=20,
        ),
      )

      points_world, colors = depth_cloud(
        cam_data["raw_depth"][t], img_rgb, K, T_cam2world, cfg.depth_stride, cfg.max_depth
      )
      rec.log(f"/scene/{view}", rr.Points3D(points_world, colors=colors, radii=cfg.scene_radius))
      n_scene_points += len(points_world)

    rec.flush(timeout_sec=120)

  return n_scene_points, n_off_image


def log_tracks(rec, review):
  colors = review.track_colors
  labels = [str(point) for point in range(review.n_points)]

  for t, points_world in enumerate(review.tracks_3d):
    rec.set_time("frame", sequence=t)
    rec.log(
      "/tracks",
      rr.Points3D(points_world, colors=colors, radii=TRACK_RADIUS_M, labels=labels, show_labels=False),
    )


def log_inspect(rec, review, inspect):
  """One toggleable subtree per inspected track: the point and what each camera says about it."""
  centers = review.T_cam2world[:, :, :3, 3]
  query_frame = review.queries[:, 2].astype(int)
  query_view = review.queries[:, 3].astype(int)

  for track in inspect:
    tracks_3d = review.tracks_3d[:, track]
    kind = "robot" if track < review.n_robot else "static"

    for t, point_world in enumerate(tracks_3d):
      rec.set_time("frame", sequence=t)
      calls = [
        f"cam{view} {verdict(bool(review.vis[view, t, track]), *frame_status(review, view, t, track), t == query_frame[track] and view == query_view[track])}"
        for view in range(review.n_views)
      ]
      rec.log(
        f"/inspect/{track}/point",
        rr.Points3D(
          point_world[None],
          colors=INSPECT,
          radii=INSPECT_RADIUS_M,
          labels=[f"{track} ({kind}) | " + " | ".join(calls)],
          show_labels=True,
        ),
      )
      rec.log(
        f"/inspect/{track}/rays",
        rr.LineStrips3D(
          [[point_world, centers[view, t]] for view in range(review.n_views)],
          colors=[ray_color(review, view, t, track) for view in range(review.n_views)],
          radii=RAY_RADIUS_M,
        ),
      )


def log_inspect_views(rec, review, inspect):
  """Every inspected track in every camera at once, carrying only its number: which one to look at
  closely is a question for the 3D view, and a verdict per track per view would bury the image."""
  inspect = np.asarray(inspect)
  query_frame = review.queries[inspect, 2].astype(int)
  query_view = review.queries[inspect, 3].astype(int)
  labels = [str(track) for track in inspect]

  for view in range(review.n_views):
    width, height = review.image_wh[view]
    for t in range(review.n_frames):
      rec.set_time("frame", sequence=t)
      u, v, z = core.geometry.project_points(review.tracks_3d[t, inspect], review.K[view], review.T_cam2world[view, t])
      inside = core.geometry.in_frame(u, v, width, height) & (z > 0)
      visible = review.vis[view, t, inspect]

      rec.log(
        f"/views/{view}/inspect",
        rr.Points2D(
          np.stack([u[inside], v[inside]], axis=1) + 0.5,
          colors=np.where(visible[inside, None], VISIBLE, NOT_VISIBLE).astype(np.uint8),
          radii=rr.Radius.ui_points(5),
          labels=[labels[i] for i in np.flatnonzero(inside)],
          show_labels=True,
          draw_order=22,
        ),
      )
      born = (query_frame == t) & (query_view == view)
      rec.log(
        f"/views/{view}/query",
        rr.LineStrips2D(
          [cross for i in np.flatnonzero(born) for cross in query_cross(review, int(inspect[i]))],
          colors=QUERY,
          radii=0.8,
          draw_order=23,
        ),
      )


def build_recording(episode, review, episode_id, review_root, cfg):
  rrd = os.path.abspath(os.path.expanduser(os.path.join(review_root, f"{episode_id}.rrd")))
  os.makedirs(os.path.dirname(rrd), exist_ok=True)
  staging = rrd + ".partial"
  inspect = inspect_tracks(review, cfg.n_inspect)

  # One application id per episode. The viewer keeps a blueprint per application, so sharing one across
  # episodes meant the first recording opened set the layout for the rest: its overrides name the track
  # ids it inspects, and against another episode's ids they match nothing and every overlay comes up on
  # at once. The plus signs would otherwise be migrated to an entry name behind our back.
  rec = rr.RecordingStream(episode_id.replace("+", "-"), recording_id=episode_id)
  try:
    rec.save(staging, default_blueprint=blueprint(review, inspect, episode_id, cfg.fps))
    log_tracks(rec, review)
    log_inspect(rec, review, inspect)
    log_inspect_views(rec, review, inspect)
    rec.flush(timeout_sec=120)
    n_scene_points, n_off_image = log_cameras(rec, review, episode, cfg)
  finally:
    rec.disconnect()
  os.replace(staging, rrd)

  print(
    f"  {review.n_points} tracks ({review.n_robot} robot, {review.n_points - review.n_robot} static)"
    f" | {review.n_frames} frames | {review.n_views} views"
    f" | {n_scene_points / 1e6:.1f}M scene points"
    f" | annotated visible {100 * review.vis.mean():.0f}%"
    f" | visible off-image {n_off_image}"
    f" | inspecting {len(inspect)} tracks, showing {inspect[0]}"
    f" | {os.path.getsize(rrd) / 1024**2:.0f} MiB"
  )
  return rrd


def process_episode(episode_id, config):
  episode = core.io.load_depth_data(episode_id, config.paths.depth, load_video=True)
  poses = core.io.load_extrinsics(episode, config.paths.extrinsics)
  tracks = core.io.load_track_data(episode_id, config.paths.tracks)

  review = Review(episode, poses, tracks)
  build_recording(episode, review, episode_id, config.paths.review, config.review)


def main(_):
  config = config_flag.value

  available = core.runner.list_episode_dirs(config.paths.tracks)
  if config.paths.episode_list:
    available &= core.io.read_episode_list(config.paths.episode_list)

  target = core.runner.shard_episodes(
    available,
    config.runner.rank,
    config.runner.world_size,
    config.runner.limit,
  )
  review_root = os.path.abspath(os.path.expanduser(config.paths.review))
  done = {e for e in target if os.path.exists(os.path.join(review_root, f"{e}.rrd"))}

  def run_one(episode_id):
    process_episode(episode_id, config)

  core.runner.run_episodes(
    target,
    run_one,
    rank=config.runner.rank,
    world_size=config.runner.world_size,
    done=done,
    stage="Review",
  )


if __name__ == "__main__":
  config_flag = config_flags.DEFINE_config_file("config", config.__file__)
  app.run(main)
