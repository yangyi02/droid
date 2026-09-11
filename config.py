import os

import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  repo = os.path.dirname(os.path.abspath(__file__))
  data = os.path.join(repo, "data")
  output = os.path.join(data, "output", "droid")

  config.paths = ml_collections.ConfigDict()
  config.paths.meta = os.path.join(data, "meta", "1.0.1")
  config.paths.raw = os.path.join(data, "input", "robotics", "droid_raw", "1.0.1")
  config.paths.urdf = os.path.join(repo, "assets", "franka_description", "franka_panda_robotiq_2f85_og.urdf")
  config.paths.depth = os.path.join(output, "depth")
  config.paths.extrinsics = os.path.join(output, "extrinsics")
  config.paths.tracks = os.path.join(output, "tracks")
  config.paths.metrics = os.path.join(output, "metrics")

  config.urls = ml_collections.ConfigDict()
  config.urls.meta = "https://huggingface.co/KarlP/droid/resolve/main"

  config.runner = ml_collections.ConfigDict()
  config.runner.rank = 0
  config.runner.world_size = 1
  config.runner.limit = -1

  config.render = ml_collections.ConfigDict()
  config.render.gpu = True

  config.depth = ml_collections.ConfigDict()
  config.depth.min_frames = 48
  config.depth.max_frames = 250
  config.depth.conf_thresh = 0.95
  config.depth.max_depth_thresh = 0.15
  config.depth.gripper_closed_thresh = 0.05
  config.depth.mask_area_min = 0.02
  config.depth.mask_area_max = 0.45
  config.depth.consensus_thresh = 0.5

  config.extrinsics = ml_collections.ConfigDict()
  config.extrinsics.outer_steps = 5
  config.extrinsics.inner_steps = 100
  config.extrinsics.lr = 0.001
  config.extrinsics.n_steps = 500
  config.extrinsics.n_points = 2000
  config.extrinsics.chamfer_match_radius = 0.05
  config.extrinsics.max_depth = 1.5

  config.tracks = ml_collections.ConfigDict()
  config.tracks.num_static_points_per_view = 100
  config.tracks.num_robot_points_per_view = 100
  config.tracks.match_radius = 0.005
  config.tracks.max_depth = 1.5
  config.tracks.min_run_fraction = 0.10
  config.tracks.flicker = 0.10
  config.tracks.depth_tolerance = 0.01

  return config
