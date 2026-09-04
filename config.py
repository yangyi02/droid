import os

import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  repo = os.path.dirname(os.path.abspath(__file__))
  data = os.path.join(repo, "data")
  output = os.path.join(data, "output", "droid")

  config.paths = ml_collections.ConfigDict()
  config.paths.repo = repo
  config.paths.data = data
  config.paths.meta = os.path.join(data, "meta", "1.0.1")
  config.paths.raw = os.path.join(data, "input", "robotics", "droid_raw", "1.0.1")
  config.paths.output = output
  config.paths.depth = os.path.join(output, "depth")
  config.paths.extrinsics = os.path.join(output, "extrinsics")
  config.paths.tracks = os.path.join(output, "tracks")
  config.paths.metrics = os.path.join(output, "metrics")
  config.paths.tapvidmv = os.path.join(output, "tapvidmv")
  config.paths.urdf = os.path.join(
    repo, "assets", "franka_description", "franka_panda_robotiq_2f85_og.urdf"
  )

  config.urls = ml_collections.ConfigDict()
  config.urls.gcs_input = "gs://gresearch/robotics/droid_raw"
  config.urls.gcs_output = "gs://dm-tapnet/tmp/droid"
  config.urls.meta = "https://huggingface.co/KarlP/droid/resolve/main"

  config.runner = ml_collections.ConfigDict()
  config.runner.rank = 0
  config.runner.world_size = 1
  config.runner.limit = -1

  config.depth = ml_collections.ConfigDict()
  config.depth.min_frames = 48
  config.depth.max_frames = 250
  config.depth.conf_thresh = 0.95
  config.depth.consensus_thresh = 0.5
  config.depth.max_depth_thresh = 0.15

  config.extrinsics = ml_collections.ConfigDict()
  config.extrinsics.outer_steps = 1
  config.extrinsics.inner_steps = 500
  config.extrinsics.lr = 0.001
  config.extrinsics.n_steps = 500
  config.extrinsics.chamfer_weight = 1.0
  config.extrinsics.robot_weight = 1.0
  config.extrinsics.chamfer_n_points = 2000

  config.tracks = ml_collections.ConfigDict()
  config.tracks.num_static_points = 300
  config.tracks.max_robot_pts_per_cam = 100
  config.tracks.match_radius = 0.005
  config.tracks.safe_margin = 15
  config.tracks.robot_safe_margin = 7
  config.tracks.tau = 0.015
  config.tracks.min_run_frames = 30
  config.tracks.flicker = 0.10
  config.tracks.depth_tolerance = 0.05

  return config
