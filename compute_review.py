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

APP_ID = "droid_track_review"
JPEG_QUALITY = 95

ROBOT = [56, 189, 248]
STATIC = [250, 204, 21]
VISIBLE = [34, 220, 100]
NOT_VISIBLE = [255, 65, 65]
INSPECT = [255, 40, 235]
QUERY = [255, 205, 30]
WHITE = [255, 255, 255]

SCENE_RADIUS_M = 0.001
TRACK_RADIUS_M = 0.003
INSPECT_RADIUS_M = 0.006
RAY_RADIUS_M = 0.0015
TRAIL_FRAMES = 12


def build_queries(uv, query_view):
  xy = np.round(uv[query_view, 0, np.arange(uv.shape[2])])
  frame = np.zeros((len(xy), 1), dtype=np.float32)
  return np.concatenate([xy, frame, query_view[:, None]], axis=1).astype(np.float32)


class Review:
  def __init__(self, episode, poses, tracks):
    self.tracks_3d = tracks["tracks_3d"]
    self.vis = tracks["vis"]
    self.n_robot = tracks["n_robot"]
    self.queries = build_queries(tracks["uv"], tracks["query_view"])

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


def inspect_tracks(review, n_inspect):
  chosen = []
  for group in (np.arange(review.n_robot), np.arange(review.n_robot, review.n_points)):
    points_world = review.tracks_3d[0, group]
    nearest_sq = np.sum((points_world - points_world.mean(axis=0)) ** 2, axis=1)
    for _ in range(min(n_inspect // 2, len(group))):
      pick = int(np.argmax(nearest_sq))
      chosen.append(int(group[pick]))
      nearest_sq = np.minimum(nearest_sq, np.sum((points_world - points_world[pick]) ** 2, axis=1))
      nearest_sq[pick] = -np.inf
  return chosen


def in_frame(u, v, z, image_wh):
  return (z > 0) & (u >= -0.5) & (u < image_wh[0] - 0.5) & (v >= -0.5) & (v < image_wh[1] - 0.5)


def verdict(track, visible, inside, z, at_query):
  status = "VISIBLE" if visible else "NOT VISIBLE"
  if z <= 0:
    status += " | behind camera"
  elif not inside:
    status += " | outside image"
  if visible and not inside:
    status += " | INCONSISTENT"
  if at_query:
    status += " | QUERY FRAME"
  return f"{track} | {status}"


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


def blueprint(review, inspect, fps):
  center = review.tracks_3d[0, inspect[0]]
  extent = float(np.linalg.norm(np.ptp(review.tracks_3d.reshape(-1, 3), axis=0)))

  return rrb.Blueprint(
    rrb.Vertical(
      rrb.Spatial3DView(
        name="3D tracks",
        origin="/",
        line_grid=False,
        background=rrb.Background(kind="GradientDark"),
        eye_controls=rrb.EyeControls3D(
          kind="Orbital",
          tracking_entity=f"/inspect/{inspect[0]}/point",
          look_target=center,
          position=center + np.array([1.2, -1.4, 0.9]) * max(extent * 0.4, 0.3),
          eye_up=[0, 0, 1],
        ),
        overrides={f"/inspect/{track}/rays": rrb.EntityBehavior(visible=False) for track in inspect[1:]},
      ),
      rrb.Horizontal(
        *[
          rrb.Spatial2DView(
            name=f"Camera {view}",
            origin=f"/views/{view}",
            visual_bounds=rrb.VisualBounds2D(x_range=[0, int(wh[0])], y_range=[0, int(wh[1])]),
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

  n_scene_points, n_off_image = 0, 0
  for view, cam_data in enumerate(episode["camera"].values()):
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

    for t in range(review.n_frames):
      rec.set_time("frame", sequence=t)
      img_rgb = cam_data["video_rgb"][t]
      T_cam2world = review.T_cam2world[view, t]

      rec.log(f"/views/{view}/rgb", rr.Image(img_rgb).compress(jpeg_quality=JPEG_QUALITY))
      rec.log(
        f"/cameras/{view}/pose",
        rr.Transform3D(translation=T_cam2world[:3, 3], mat3x3=T_cam2world[:3, :3]),
      )

      points_world, colors = depth_cloud(
        cam_data["raw_depth"][t], img_rgb, K, T_cam2world, cfg.depth_stride, cfg.max_depth
      )
      rec.log(f"/scene/{view}", rr.Points3D(points_world, colors=colors, radii=SCENE_RADIUS_M))
      n_scene_points += len(points_world)

      u, v, z = core.geometry.project_points(review.tracks_3d[t], K, T_cam2world)
      inside = in_frame(u, v, z, review.image_wh[view])
      visible = review.vis[view, t]
      n_off_image += int((visible & ~inside).sum())
      rec.log(
        f"/views/{view}/tracks",
        rr.Points2D(
          np.stack([u[inside], v[inside]], axis=1) + 0.5,
          colors=np.where(visible[inside, None], VISIBLE, NOT_VISIBLE).astype(np.uint8),
          radii=rr.Radius.ui_points(3),
          labels=[str(point) for point in np.flatnonzero(inside)],
          show_labels=False,
          draw_order=20,
        ),
      )

    rec.flush(timeout_sec=120)

  return n_scene_points, n_off_image


def log_tracks(rec, review):
  colors = np.where(np.arange(review.n_points)[:, None] < review.n_robot, ROBOT, STATIC).astype(np.uint8)
  labels = [str(point) for point in range(review.n_points)]

  for t, points_world in enumerate(review.tracks_3d):
    rec.set_time("frame", sequence=t)
    rec.log(
      "/tracks",
      rr.Points3D(points_world, colors=colors, radii=TRACK_RADIUS_M, labels=labels, show_labels=False),
    )


def log_inspect(rec, review, inspect):
  centers = review.T_cam2world[:, :, :3, 3]

  for track in inspect:
    tracks_3d = review.tracks_3d[:, track]
    for t, point_world in enumerate(tracks_3d):
      rec.set_time("frame", sequence=t)
      rec.log(f"/inspect/{track}/point", rr.Points3D(point_world[None], colors=INSPECT, radii=INSPECT_RADIUS_M))
      rec.log(
        f"/inspect/{track}/trail",
        rr.LineStrips3D([tracks_3d[max(0, t - TRAIL_FRAMES + 1) : t + 1]], colors=INSPECT, radii=RAY_RADIUS_M),
      )
      rec.log(
        f"/inspect/{track}/rays",
        rr.LineStrips3D(
          [[point_world, centers[view, t]] for view in range(review.n_views)],
          colors=[VISIBLE if review.vis[view, t, track] else NOT_VISIBLE for view in range(review.n_views)],
          radii=RAY_RADIUS_M,
        ),
      )


def log_inspect_views(rec, review, inspect):
  query_frame = review.queries[:, 2].astype(int)
  query_view = review.queries[:, 3].astype(int)

  for view in range(review.n_views):
    width, height = review.image_wh[view]
    for t in range(review.n_frames):
      rec.set_time("frame", sequence=t)
      u, v, z = core.geometry.project_points(
        review.tracks_3d[t, inspect], review.K[view], review.T_cam2world[view, t]
      )
      inside = in_frame(u, v, z, review.image_wh[view])

      for i, track in enumerate(inspect):
        visible = bool(review.vis[view, t, track])
        color = VISIBLE if visible else NOT_VISIBLE
        at_query = t == query_frame[track] and view == query_view[track]
        marker = [[float(np.clip(u[i] + 0.5, 0, width)), float(np.clip(v[i] + 0.5, 0, height))]]
        prefix = f"/views/{view}/inspect/{track}"

        rec.log(
          prefix + "/marker_outline",
          rr.Points2D(
            marker if inside[i] else np.empty((0, 2)),
            colors=WHITE,
            radii=rr.Radius.ui_points(7),
            draw_order=21,
          ),
        )
        rec.log(
          prefix + "/marker",
          rr.Points2D(
            marker,
            colors=color,
            radii=rr.Radius.ui_points(5) if inside[i] else 0,
            labels=[verdict(track, visible, inside[i], z[i], at_query)],
            show_labels=True,
            draw_order=22,
          ),
        )
        rec.log(
          prefix + "/query",
          rr.LineStrips2D(query_cross(review, track) if at_query else [], colors=QUERY, radii=0.8, draw_order=23),
        )


def build_recording(episode, review, episode_id, review_root, cfg):
  rrd = os.path.abspath(os.path.expanduser(os.path.join(review_root, f"{episode_id}.rrd")))
  os.makedirs(os.path.dirname(rrd), exist_ok=True)
  staging = rrd + ".partial"
  inspect = inspect_tracks(review, cfg.n_inspect)

  rec = rr.RecordingStream(APP_ID)
  try:
    rec.save(staging, default_blueprint=blueprint(review, inspect, cfg.fps))
    n_scene_points, n_off_image = log_cameras(rec, review, episode, cfg)
    log_tracks(rec, review)
    log_inspect(rec, review, inspect)
    log_inspect_views(rec, review, inspect)
    rec.flush(timeout_sec=120)
  finally:
    rec.disconnect()
  os.replace(staging, rrd)

  print(
    f"  {review.n_points} tracks ({review.n_robot} robot, {review.n_points - review.n_robot} static)"
    f" | {review.n_frames} frames | {review.n_views} views"
    f" | {n_scene_points / 1e6:.1f}M scene points"
    f" | annotated visible {100 * review.vis.mean():.0f}%"
    f" | visible off-image {n_off_image}"
    f" | inspecting {inspect}"
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

  target = core.runner.shard_episodes(
    core.runner.list_episode_dirs(config.paths.tracks),
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
