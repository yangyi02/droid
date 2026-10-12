import importlib
import os
import shutil
import sys

from absl import app
from absl import flags
from ml_collections import config_flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import core.runner

XM_SCRATCH = flags.DEFINE_string(
  "xm_scratch", "", "Local scratch dir for staging episodes before upload (e.g. /dev/disk-scratch/droid)."
)
XM_UPLOAD = flags.DEFINE_string(
  "xm_upload", "", "Optional '<mount_dir>=gs://bucket/prefix' for committing staged episodes via GCS API."
)


def episode_files(local_root, entries):
  files = []
  for name in entries:
    path = os.path.join(local_root, name)
    if os.path.isdir(path):
      for dirpath, _, filenames in os.walk(path):
        files += [(os.path.join(dirpath, f), os.path.relpath(os.path.join(dirpath, f), local_root)) for f in filenames]
    else:
      files.append((path, name))
  nested = [f for f in files if f[1].count(os.sep) >= 2]
  top = [f for f in files if f[1].count(os.sep) < 2]
  return nested, top


def upload_gcs(batches, bucket, prefix, workers=16):
  from google.cloud.storage import transfer_manager

  for batch in batches:
    if batch:
      pairs = [(path, bucket.blob(f"{prefix}/{rel}" if prefix else rel)) for path, rel in batch]
      transfer_manager.upload_many(
        pairs, max_workers=workers, worker_type=transfer_manager.THREAD, raise_exception=True
      )


def stage_episodes(process, cfg, key, scratch, upload):
  if not scratch:
    return process

  final_root = os.path.abspath(os.path.expanduser(cfg.paths[key]))
  local_root = os.path.join(os.path.abspath(os.path.expanduser(scratch)), key)
  os.makedirs(local_root, exist_ok=True)
  cfg.paths[key] = local_root

  gcs_root, bucket, gcs_prefix = None, None, ""
  if upload:
    mount, _, gcs = upload.partition("=")
    mount = os.path.abspath(os.path.expanduser(mount))
    if not gcs.startswith("gs://"):
      raise ValueError(f"--xm_upload must look like <dir>=gs://bucket/prefix, got {upload!r}")
    if final_root == mount or final_root.startswith(mount + os.sep):
      from google.cloud import storage

      gcs_root = gcs.rstrip("/") + final_root[len(mount) :]
      bucket_name, _, gcs_prefix = gcs_root.removeprefix("gs://").partition("/")
      gcs_prefix = gcs_prefix.rstrip("/")
      bucket = storage.Client().bucket(bucket_name)

  print(f"Staging {key}: writing to {local_root}, committing each episode to {gcs_root or final_root}")

  def entries(episode_id):
    return [n for n in os.listdir(local_root) if n == episode_id or n.startswith(episode_id + ".")]

  def discard(episode_id):
    for name in entries(episode_id):
      path = os.path.join(local_root, name)
      if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
      else:
        os.remove(path)

  def run(episode_id):
    try:
      process(episode_id)
    except BaseException:
      discard(episode_id)
      raise
    nested, top = episode_files(local_root, entries(episode_id))
    if bucket is not None:
      upload_gcs([nested, top], bucket, gcs_prefix)
    else:
      for path, rel in nested + top:
        dst = os.path.join(final_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(path, dst)
    discard(episode_id)

  return run


def patch_zed():
  import pyzed.sl as sl

  orig_camera, orig_init = sl.Camera, sl.InitParameters

  class InitParamsProxy:
    def __init__(self):
      super().__setattr__("_inner", orig_init())
      super().__setattr__("_svo_file", None)
      self._inner.depth_mode = sl.DEPTH_MODE.NONE

    def set_from_svo_file(self, svo_file):
      super().__setattr__("_svo_file", svo_file)
      return self._inner.set_from_svo_file(svo_file)

    def __getattr__(self, name):
      return getattr(self._inner, name)

    def __setattr__(self, name, value):
      setattr(self._inner, name, value)

  class CameraProxy:
    def __init__(self):
      self._inner = orig_camera()

    def open(self, init_params):
      svo_file = getattr(init_params, "_svo_file", None)
      raw_params = getattr(init_params, "_inner", init_params)
      if svo_file and os.environ.get("ZED_REQUIRE_LOCAL_CALIB"):
        cam_id = os.path.splitext(os.path.basename(svo_file))[0]
        calib = f"/usr/local/zed/settings/SN{cam_id}.conf"
        if not os.path.exists(calib):
          raise core.runner.SkipEpisode(f"{cam_id}: no ZED calibration at {calib}")
      status = self._inner.open(raw_params)
      if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"cannot open {svo_file}: {status}")
      return status

    def __getattr__(self, name):
      return getattr(self._inner, name)

  sl.InitParameters = InitParamsProxy
  sl.Camera = CameraProxy


def install_zed_hook(depth_module):
  patched = False
  orig_extract = depth_module.extract_svo_video

  def extract_svo_video(*args, **kwargs):
    nonlocal patched
    if not patched:
      patch_zed()
      patched = True
    return orig_extract(*args, **kwargs)

  depth_module.extract_svo_video = extract_svo_video


def install_staging_hook(stage_key, cfg_flag):
  orig_run_episodes = core.runner.run_episodes

  def staged_run_episodes(episode_ids, process, rank, world_size, done, stage):
    wrapped = stage_episodes(process, cfg_flag.value, stage_key, XM_SCRATCH.value, XM_UPLOAD.value)
    return orig_run_episodes(episode_ids, wrapped, rank=rank, world_size=world_size, done=done, stage=stage)

  core.runner.run_episodes = staged_run_episodes


if __name__ == "__main__":
  if len(sys.argv) < 2:
    raise SystemExit("Usage: python xm/run_stage.py <depth|extrinsics|scene|tracks|metrics|review> [flags...]")
  stage_name = sys.argv.pop(1)
  stage_module = importlib.import_module(f"compute_{stage_name}")
  stage_module.config_flag = config_flags.DEFINE_config_file("config", config.__file__)
  if stage_name == "depth":
    install_zed_hook(stage_module)
  install_staging_hook(stage_name, stage_module.config_flag)
  app.run(stage_module.main)
