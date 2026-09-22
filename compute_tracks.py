import functools
import os

import cv2
import numpy as np
from absl import app
from ml_collections import config_flags

import config
import core.geometry
import core.io
import core.physics
import core.runner


def resize_mask(mask, margin):
  """Grow the mask by margin pixels, or shrink it when margin is negative."""
  kernel = np.ones((abs(margin), abs(margin)), np.uint8)
  morph = cv2.dilate if margin > 0 else cv2.erode
  return morph(mask.astype(np.uint8), kernel).astype(bool)


def query_frames(episode, poses, pb_renderer, config):
  """One set of query frames for the episode: the first frame, and then in every equal stretch of the
  rest of it, the frame where the camera that sees the least of the arm sees the most of it.

  The first frame is in whatever it offers, because a model that only runs forwards has nothing to
  track from without it, and a query there is the only one that covers the whole episode. It takes the
  first stretch's place rather than being added to the set, so that the episode still holds as many
  query times as it is asked for and no two of them sit a tenth of a second apart. On an episode where
  the arm starts half out of view that quota comes back crowded, or short -- the points it does place
  are no worse for it, there are just fewer places to put them.

  The stretches cover the whole episode. Ground truth does not run out: the arm's points come from
  kinematics and the background's stand still, so a point born on the last frame has its whole track
  before it, and a model that reads the clip both ways is asked for exactly that. Only a forward-only
  model needs a query with an episode still ahead of it, and the first frame is that query.

  Inside a stretch the frame is chosen rather than spaced, because an evenly spaced one takes whatever
  the arm happened to be doing: on one episode here the first frame left a camera 53 candidates to
  fill a quota of 20 from, while another frame in the same stretch offered 2243.

  The cameras share the frames instead of each taking its own best. Sharing costs almost nothing -- a
  camera lands within a few percent of its own best moment -- and it thirds the number of distinct query
  times in an episode, which is what a tracker has to run a pass for, and what lets a multi-view method
  read all three videos at the same instant. Each camera is scored against its own best frame, because
  one riding on the wrist fills its image with the arm and one across the room never can."""
  robot = episode["robot"]
  reach = len(robot["joint_positions"])

  seen = np.zeros((len(episode["camera"]), reach))
  for frame in range(reach):
    pb_renderer.update_robot_pose(robot["joint_positions"][frame], gripper_state=robot["gripper_positions"][frame])
    for view, (cam_id, cam_data) in enumerate(episode["camera"].items()):
      height, width = cam_data["raw_depth"][frame].shape
      drawn = pb_renderer.render_depth(poses[cam_id]["extrinsics"][frame], cam_data["K"], width, height)
      seen[view, frame] = resize_mask(drawn > 0, -config.tracks.mask_margin).sum()

  worst = (seen / np.maximum(seen.max(axis=1, keepdims=True), 1)).min(axis=0)
  # The first stretch spends itself on frame 0 rather than on its best frame. Adding frame 0 on top of
  # the five instead puts two queries a tenth of a second apart, which is one arm pose, not two.
  edges = np.linspace(0, reach, config.tracks.num_query_frames + 1).astype(int)
  return np.array([0] + [lo + int(np.argmax(worst[lo:hi])) for lo, hi in zip(edges[1:-1], edges[2:])])


def spread_cells(points, min_gap):
  """One candidate per min_gap cube of surface.

  In metres rather than in pixels, because a grid on the image measures the camera, not the scene: the
  wrist camera sits 15 cm from the gripper and the room cameras 65 cm from the arm, so the same pixel
  grid offers a candidate every 5 mm on the gripper and every 3 cm on everything else. Sampling then
  runs out of arm to pick from long before it runs out of budget, and spends the rest on the gripper."""
  order = np.random.permutation(len(points))
  _, first = np.unique(np.round(points[order] / min_gap).astype(np.int64), axis=0, return_index=True)
  return order[first]


def sensor_slack(depth, u, v, z_pred, config):
  """How far behind the point the measured surface sits, counted in what the depth map can be trusted to.
  Stereo error grows with range, so a centimetre at arm's length is not a centimetre across the room: past
  one the surface is really behind the point, under minus one something is really standing in front of it.
  Infinite where the stereo left a hole, which is not the same as empty space, and not a number off-frame."""
  z = core.geometry.sample_depth(depth, u, v, z_pred)
  gap = np.where(z == 0, np.inf, z) - z_pred
  return gap / (config.tracks.sensor_tolerance_base + config.tracks.sensor_tolerance_slope * z_pred)


def urdf_gap(depth, u, v, z_pred):
  """The same against the rendered robot, which is exact, so only the pose can be wrong."""
  z = core.geometry.sample_depth(depth, u, v, z_pred)
  return np.where(z == 0, np.inf, z) - z_pred


def part_masks(parts):
  """Which candidates sit on which robot link."""
  return [(part, (parts == part).all(axis=1)) for part in map(tuple, np.unique(parts, axis=0).tolist())]


def link_local(points_world, parts):
  """Candidates in the frame of the link they sit on, so forward kinematics can carry them."""
  homogeneous = np.hstack([points_world, np.ones((len(points_world), 1))]).T
  local = np.zeros_like(homogeneous)
  for part, on_part in part_masks(parts):
    local[:, on_part] = np.linalg.inv(core.physics.link_transform(*part)) @ homogeneous[:, on_part]
  return local


def find_robot_candidates(episode, poses, pb_renderer, queries, config):
  """Robot surface on each camera's own query frames, in the frame of the link it sits on.

  A candidate this camera cannot really see is dropped here. The rendered arm only says the surface
  faces the camera; a real object standing in front of it is not in that render, and a point hidden
  on the one frame it is born cannot be a query."""
  robot = episode["robot"]

  local, parts, query_view, query_frame = [], [], [], []
  for view, (src_cam, cam_data) in enumerate(episode["camera"].items()):
    for frame in queries:
      pb_renderer.update_robot_pose(robot["joint_positions"][frame], gripper_state=robot["gripper_positions"][frame])

      K = cam_data["K"]
      height, width = cam_data["raw_depth"][frame].shape
      T_cam2world = poses[src_cam]["extrinsics"][frame]

      obj_ids, link_ids, urdf_depth = pb_renderer.render_segmentation(T_cam2world, K, width, height)
      vs, us = np.where(resize_mask(obj_ids == pb_renderer.robot_id, -config.tracks.mask_margin))

      u, v, z = us.astype(np.float32), vs.astype(np.float32), urdf_depth[vs, us]
      sensor = sensor_slack(cam_data["raw_depth"][frame], u, v, z, config)
      lit = np.isinf(sensor) | (sensor >= -1)

      surface = core.geometry.unproject_pixels(u[lit], v[lit], z[lit], K, T_cam2world)
      cell = spread_cells(surface, config.tracks.min_gap)
      on_parts = np.stack([obj_ids[vs, us][lit], link_ids[vs, us][lit]], axis=1)[cell]

      local.append(link_local(surface[cell], on_parts))
      parts.append(on_parts)
      query_view.append(np.full(len(cell), view, dtype=np.int8))
      query_frame.append(np.full(len(cell), frame, dtype=np.int32))

  return (
    np.concatenate(local, axis=1),
    np.concatenate(parts),
    np.concatenate(query_view),
    np.concatenate(query_frame),
  )


def carry_robot(local, parts, robot, pb_renderer, frames):
  """Where each arm point sits on the frames asked for, by forward kinematics.

  Asked for one frame to sample on and for every frame once the sampling is done, so the whole episode
  is only ever carried for the points that were kept."""
  carried = np.zeros((len(frames), local.shape[1], 3), dtype=np.float32)
  on_links = part_masks(parts)
  for step, t in enumerate(frames):
    pb_renderer.update_robot_pose(robot["joint_positions"][t], gripper_state=robot["gripper_positions"][t])
    for part, on_part in on_links:
      carried[step, on_part] = (core.physics.link_transform(*part) @ local[:, on_part])[:3].T
  return carried


def depth_steps(depth, window=5):
  """How far apart the nearest and the farthest surface within a window of each pixel are.

  A pixel where that distance is large sits on the edge between two things, and stereo cannot hold an
  edge still: the boundary moves a pixel between frames and the reading jumps from one surface to the
  other. Everything read off such a pixel inherits the coin toss -- where the point is, and whether
  anything is in front of it. Where nothing was measured nearby the answer is infinite, which reads as
  an edge and is meant to: nothing there can be trusted either."""
  kernel = np.ones((window, window), np.uint8)
  far = cv2.dilate(depth, kernel)
  near = -cv2.dilate(np.where(depth > 0, -depth, -np.inf).astype(np.float32), kernel)
  return np.where(far > 0, far - near, np.inf)


def find_static_candidates(episode, poses, pb_renderer, queries, config):
  """Background within reach on the query frames, in the places every camera that can see it agrees on.

  Two things are asked of a candidate beyond standing still. Its pixel must not sit on a depth edge in
  any camera, because a reading taken there is a coin toss. And every camera that can see the place must
  put the surface where this one does: a camera abstains when it measured nothing there, when the arm is
  in the way, or when the place is off its image, but one that does read a surface somewhere else is
  reporting that the point is not where we think it is. Asking only that some camera agree let a point
  through that a third camera put four centimetres away -- which is four centimetres of error in its
  position and several pixels of error in every projection of it."""
  robot = episode["robot"]
  match_radius, max_depth = config.tracks.match_radius, config.tracks.max_depth
  max_step = config.tracks.max_edge_step

  points_3d, query_view, query_frame = [], [], []
  for frame in queries:
    pb_renderer.update_robot_pose(robot["joint_positions"][frame], gripper_state=robot["gripper_positions"][frame])

    drawn, steps = {}, {}
    for cam_id, cam_data in episode["camera"].items():
      height, width = cam_data["raw_depth"][frame].shape
      drawn[cam_id] = pb_renderer.render_depth(poses[cam_id]["extrinsics"][frame], cam_data["K"], width, height)
      steps[cam_id] = depth_steps(cam_data["raw_depth"][frame])

    for view, (src_cam, cam_data) in enumerate(episode["camera"].items()):
      depth = cam_data["raw_depth"][frame]
      K, T_cam2world = cam_data["K"], poses[src_cam]["extrinsics"][frame]

      on_env = ~resize_mask(drawn[src_cam] > 0, config.tracks.mask_margin) & (depth > 0) & (depth <= max_depth)
      vs, us = np.where(on_env & (steps[src_cam] <= max_step))

      points = core.geometry.unproject_pixels(
        us.astype(np.float32), vs.astype(np.float32), depth[vs, us], K, T_cam2world
      )

      confirmed = np.zeros(len(points), dtype=bool)
      doubted = np.zeros(len(points), dtype=bool)
      for other_cam, other_data in episode["camera"].items():
        if other_cam == src_cam:
          continue
        u, v, z = core.geometry.project_points(points, other_data["K"], poses[other_cam]["extrinsics"][frame])
        z_other = core.geometry.sample_depth(other_data["raw_depth"][frame], u, v, z)
        step = core.geometry.sample_depth(steps[other_cam], u, v, z)
        behind_arm = core.geometry.sample_depth(drawn[other_cam], u, v, z) > 0

        speaks = np.isfinite(z_other) & (z_other > 0) & ~behind_arm
        agrees = np.abs(z_other - z) < match_radius
        confirmed |= speaks & agrees & (z_other <= max_depth)
        doubted |= speaks & (~agrees | ~(step <= max_step))

      cell = spread_cells(points[confirmed & ~doubted], config.tracks.min_gap)
      points_3d.append(points[confirmed & ~doubted][cell])
      query_view.append(np.full(len(cell), view, dtype=np.int8))
      query_frame.append(np.full(len(cell), frame, dtype=np.int32))

  return np.concatenate(points_3d).astype(np.float32), np.concatenate(query_view), np.concatenate(query_frame)


def project_tracks(tracks_3d, n_robot, episode, poses, pb_renderer, config):
  """Every point in every view: where it lands, whether that view can see it, and what the depth map measured."""
  robot = episode["robot"]
  n_frames, n_points, _ = tracks_3d.shape
  n_views = len(episode["camera"])

  uv = np.zeros((n_views, n_frames, n_points, 2), dtype=np.float32)
  margin = np.zeros((n_views, n_frames, n_points), dtype=np.float32)
  inside = np.zeros((n_views, n_frames, n_points), dtype=bool)
  slack = np.zeros((n_views, n_frames, n_points), dtype=np.float32)

  for t in range(n_frames):
    pb_renderer.update_robot_pose(robot["joint_positions"][t], gripper_state=robot["gripper_positions"][t])

    drawn = [
      pb_renderer.render_depth(poses[cam_id]["extrinsics"][t], cam_data["K"], *cam_data["raw_depth"][t].shape[::-1])
      for cam_id, cam_data in episode["camera"].items()
    ]

    for view, (cam_id, cam_data) in enumerate(episode["camera"].items()):
      K = cam_data["K"]
      T_cam2world = poses[cam_id]["extrinsics"][t]
      urdf_depth = drawn[view]

      height, width = cam_data["raw_depth"][t].shape
      u, v, z_pred = core.geometry.project_points(tracks_3d[t], K, T_cam2world)
      uv[view, t] = np.stack([u, v], axis=1)
      inside[view, t] = core.geometry.in_frame(u, v, width, height) & (z_pred > 0)

      urdf = urdf_gap(urdf_depth, u, v, z_pred)
      sensor = sensor_slack(cam_data["raw_depth"][t], u, v, z_pred, config)

      # A frame where this camera measured nothing says nothing, and is left as no reading at all. The
      # label then stays where it was rather than being answered by something else: asking the other
      # cameras along the ray instead came back with "blocked" three times as often as this camera's
      # own depth did, so a point flipped every time the map dropped out under it for a frame.
      # The rendered arm is exact and still answers: fmin ignores the missing reading, not the robot.
      margin[view, t] = np.fmin(urdf / config.tracks.urdf_tolerance, np.where(np.isinf(sensor), np.nan, sensor))
      slack[view, t] = sensor

  # Two lines for the arm, one for the background: the arm passes behind things and comes back, so its
  # labels have to survive a reading that rests on the cut, while the background is read at the single
  # cut it always was. Both hold their label through a frame with no reading.
  read = np.concatenate(
    [latch(margin[:, :, :n_robot], config.tracks.hysteresis), latch(margin[:, :, n_robot:], 0.0)], axis=2
  )
  return uv, settle(read, inside), slack


def latch(margin, band):
  """Read the arm's margin with two lines instead of one, so a point sitting on the cut stops flickering.

  The cut is at -1: below it something stands in front of the point. A point exactly on it is a coin
  toss decided by measurement noise, frame after frame, while nothing in the scene has moved. So it
  takes band past the cut to call a point hidden, and band the other way to call it visible again;
  between the two lines the label stays where it was. A real occlusion clears both lines and starts on
  the same frame either way -- what the band removes is the stutter, not the event.

  A frame with no reading at all arrives as a nan, which is under neither line, so the label holds
  there too. With band at zero that is all this does: one cut, and no opinion where nothing was
  measured."""
  hidden, shown = margin < -1 - band, margin > -1 + band

  vis = np.empty(margin.shape, dtype=bool)
  vis[:, 0] = ~(margin[:, 0] < -1)  # a first frame with no reading starts visible: nothing said otherwise
  for t in range(1, margin.shape[1]):
    vis[:, t] = np.where(hidden[:, t], False, np.where(shown[:, t], True, vis[:, t - 1]))
  return vis


def settle(vis, inside):
  """Drop labels that change for a single frame and change straight back.

  Nothing on a rigid arm is revealed and hidden again in a thirtieth of a second, so a lone frame that
  disagrees with both its neighbours is a threshold being grazed, not something moving. There is no
  threshold that avoids this: how far a point sits behind the drawn surface is spread evenly over the
  first two centimetres, so any cut runs through the middle of a crowd, and moving it only changes
  which points sit on the edge.

  Settling can turn a frame visible as well as hidden, so it is held to the same floor as everything
  else: a point that landed outside the image was seen by nobody, whatever its neighbours did."""
  settled = vis
  for _ in range(4):
    middle = settled[:, 1:-1]
    alone = (middle != settled[:, :-2]) & (middle != settled[:, 2:])
    if not alone.any():
      break
    settled = settled.copy()
    settled[:, 1:-1] = np.where(alone, ~middle, middle)
  return settled & inside


def never_seen_through(slack, max_seen_through):
  """A point the depth map keeps looking straight through sits on something that moved away."""
  seen_through = np.isfinite(slack) & (slack > 1)
  clear_line = np.isfinite(slack) & (slack >= -1)
  return ~(seen_through.sum(axis=1) / np.maximum(clear_line.sum(axis=1), 1) > max_seen_through).any(axis=0)


def out_of_reach(points_3d, episode, pb_renderer, clearance):
  """Background points the gripper ever closes on are the ones it carries away."""
  robot = episode["robot"]

  closest = np.full(len(points_3d), np.inf, dtype=np.float32)
  for t in range(len(robot["joint_positions"])):
    pb_renderer.update_robot_pose(robot["joint_positions"][t], gripper_state=robot["gripper_positions"][t])
    links = np.array(
      [core.physics.link_transform(pb_renderer.robot_id, link)[:3, 3] for link in pb_renderer.gripper_links]
    )
    closest = np.minimum(closest, np.linalg.norm(links[:, None] - points_3d[None], axis=-1).min(axis=0))

  return closest > clearance


def take(picked, home, group, quota):
  """Pick quota more points out of group, as far from each other and from everything already picked as
  they go. Distances are measured at the first frame's pose: the same spot on a link comes back to the
  same place there whatever the arm is doing, so a gripper that is in view on every query frame is
  covered once and then left alone, and later frames spend their quota on surface that has just turned
  towards a camera."""
  if quota <= 0 or not len(group):
    return picked
  return np.concatenate([picked, group[core.geometry.farthest_points(home[group], quota, seeds=home[picked])]])


def sample_tracks(keep, home, query_view, query_frame, is_robot, queries, config):
  """Every camera, on every one of its query frames: points_per_class on the arm and as many again on
  the scene, spread over the surface that camera can see.

  A query belongs to a camera -- it is a pixel in one video -- so a camera is what a quota is handed
  to. Pooling the arm's quota across the cameras instead and letting them compete on distance sounds
  fairer and is not: it hands out points by surface area, and the surface the wrist camera can see is
  a few percent of the arm, so that camera came away with eight annotated points for a whole episode.
  What the pooled version was trying to fix -- the gripper holding half the arm's points, because it
  is the one thing the wrist camera ever looks at -- is not a bias to fix here. It is what that camera
  films. Reporting one 2D number over three cameras this different is what makes it look like a bias,
  and a per-camera number does not need saving from it.

  Nothing here reads how much of a frame something fills. Area decides how many candidates there are,
  not how many points are wanted, and while it did decide the split the background -- always the
  larger surface -- spent the arm's share on most frames."""
  on_arm = in_scene = np.empty(0, dtype=int)
  for frame in queries:
    for view in range(int(query_view.max()) + 1):
      born_here = keep & (query_view == view) & (query_frame == frame)

      on_arm = take(on_arm, home, np.flatnonzero(born_here & is_robot), config.tracks.points_per_class)
      in_scene = take(in_scene, home, np.flatnonzero(born_here & ~is_robot), config.tracks.points_per_class)

  return np.sort(np.concatenate([on_arm, in_scene]))


def export_tracks(episode, tracks_3d, uv, vis, query_view, query_frame, n_robot, export_root):
  episode_id = episode["meta"]["episode_id"]
  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(export_root, episode_id)))
  os.makedirs(ep_dir, exist_ok=True)

  np.savez_compressed(os.path.join(ep_dir, "tracks_3d.npz"), tracks_3d=tracks_3d.astype(np.float32))

  for view, cam_id in enumerate(episode["camera"]):
    cam_dir = os.path.join(ep_dir, cam_id)
    os.makedirs(cam_dir, exist_ok=True)

    np.savez_compressed(os.path.join(cam_dir, "tracks_2d.npz"), tracks_2d=uv[view], vis_2d=vis[view])

  np.savez_compressed(
    os.path.join(ep_dir, "track_metadata.npz"),
    n_robot=np.array(n_robot),
    n_static=np.array(uv.shape[2] - n_robot),
    query_view=query_view,
    query_frame=query_frame,
  )


def process_episode(episode_id, pb_renderer, config):
  episode = core.io.load_depth_data(episode_id, config.paths.depth)
  poses = core.io.load_extrinsics(episode, config.paths.extrinsics)
  robot = episode["robot"]
  n_frames = len(robot["joint_positions"])

  queries = query_frames(episode, poses, pb_renderer, config)
  local, parts, robot_view, robot_frame = find_robot_candidates(episode, poses, pb_renderer, queries, config)
  static_3d, static_view, static_frame = find_static_candidates(episode, poses, pb_renderer, queries, config)

  # Sample before tracking, not after. Every candidate carried through the episode and projected into
  # every view costs the same as one that is kept, and a hundred and forty of them are thrown away for
  # each one that survives. Choosing needs only where a candidate sits and which camera and frame it
  # was born on, all of which is already here.
  query_view = np.concatenate([robot_view, static_view])
  query_frame = np.concatenate([robot_frame, static_frame])
  is_robot = np.arange(len(query_view)) < local.shape[1]
  home = np.concatenate([carry_robot(local, parts, robot, pb_renderer, [0])[0], static_3d])

  keep = np.ones(len(query_view), dtype=bool)
  keep[~is_robot] &= out_of_reach(static_3d, episode, pb_renderer, config.tracks.gripper_clearance)

  idx = sample_tracks(keep, home, query_view, query_frame, is_robot, queries, config)
  on_arm, in_scene = idx[is_robot[idx]], idx[~is_robot[idx]] - local.shape[1]

  tracks_3d = np.concatenate(
    [
      carry_robot(local[:, on_arm], parts[on_arm], robot, pb_renderer, range(n_frames)),
      np.broadcast_to(static_3d[in_scene], (n_frames, len(in_scene), 3)),
    ],
    axis=1,
  )
  query_view, query_frame = query_view[idx], query_frame[idx]
  n_robot = len(on_arm)

  uv, vis, slack = project_tracks(tracks_3d, n_robot, episode, poses, pb_renderer, config)

  # What the sampling could not know without the whole episode: a background point the depth map keeps
  # looking through sat on something that has since been moved, and a query the final labels call
  # hidden on its own frame is not a query.
  keep = vis[query_view, query_frame, np.arange(len(query_view))]
  keep[n_robot:] &= never_seen_through(slack[:, :, n_robot:], config.tracks.max_seen_through)

  n_robot = int(keep[:n_robot].sum())
  print(
    f"  {int(keep.sum())} points: {n_robot} robot, {int(keep.sum()) - n_robot} static"
    f" | per view {np.bincount(query_view[keep])}"
  )

  export_tracks(
    episode,
    tracks_3d[:, keep],
    uv[:, :, keep],
    vis[:, :, keep],
    query_view[keep],
    query_frame[keep],
    n_robot,
    config.paths.tracks,
  )


def main(_):
  config = config_flag.value
  pb_renderer = core.physics.PyBulletRenderer(config.paths.urdf, gpu=config.render.gpu)

  available = core.runner.list_episode_dirs(config.paths.extrinsics)
  if config.paths.episode_list:
    available &= core.io.read_episode_list(config.paths.episode_list)

  target = core.runner.shard_episodes(
    available,
    config.runner.rank,
    config.runner.world_size,
    config.runner.limit,
  )
  export_root = os.path.abspath(os.path.expanduser(config.paths.tracks))
  done = {e for e in target if os.path.exists(os.path.join(export_root, e, "tracks_3d.npz"))}

  def run_one(episode_id):
    process_episode(episode_id, pb_renderer, config)

  core.runner.run_episodes(
    target,
    run_one,
    rank=config.runner.rank,
    world_size=config.runner.world_size,
    done=done,
    stage="Stage 3",
  )


if __name__ == "__main__":
  config_flag = config_flags.DEFINE_config_file("config", config.__file__)
  app.run(main)
