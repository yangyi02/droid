import os

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc

STAGES = ("depth", "extrinsics", "scene", "tracks", "metrics", "review")

STAGE = flags.DEFINE_enum("stage", None, STAGES, "Pipeline stage, i.e. compute_<stage>.py.", required=True)
NUM_WORKERS = flags.DEFINE_integer("num_workers", 8, "Work units (= GPUs = shards, config.runner.world_size).")
LIMIT = flags.DEFINE_integer("limit", -1, "config.runner.limit: only the first N episodes (before sharding).")
GPU = flags.DEFINE_string("gpu", "a100", "GPU type per worker, e.g. a100, a100_80gib.")
LOCATION = flags.DEFINE_string("location", "us-central1", "GCP region; us-central1 holds the dm-pool GPUs.")
OUTPUT_PREFIX = flags.DEFINE_string(
  "output_prefix", "tmp/droid", "Output prefix in gs://dm-tapnet. Use a fresh one (e.g. tmp/droid_xmtest) for tests."
)
DISK_GIB = flags.DEFINE_integer("disk_gib", 200, "Ephemeral disk per worker for /dev/disk-scratch.")
CONFIG = flags.DEFINE_multi_string("config", [], "Extra pipeline overrides, KEY=VALUE -> --config.KEY=VALUE.")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_BUCKET, INPUT_DIR = "gresearch", "robotics/droid_raw"
OUTPUT_BUCKET = "dm-tapnet"
INPUT_MOUNT = "/app/data/input/robotics/droid_raw"
OUTPUT_MOUNT = "/app/data/output/droid"
SCRATCH = "/dev/disk-scratch/droid"


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  stage, n, prefix = STAGE.value, NUM_WORKERS.value, OUTPUT_PREFIX.value.strip("/")
  title = f"droid {stage} {n}x{GPU.value} -> gs://{OUTPUT_BUCKET}/{prefix}"

  with xm_abc.create_experiment(experiment_title=title) as experiment:
    executor = xm_abc.Gcp(
      requirements=xm.JobRequirements(
        **{GPU.value: 1}, location=LOCATION.value, ephemeral_storage=DISK_GIB.value * xm.GiB
      ),
      gcs_volumes=[
        xm_abc.GcsVolume(
          bucket=INPUT_BUCKET, directory_path=INPUT_DIR, mount_path=INPUT_MOUNT, mount_options="implicit-dirs"
        ),
        xm_abc.GcsVolume(
          bucket=OUTPUT_BUCKET, directory_path=prefix, mount_path=OUTPUT_MOUNT, mount_options="implicit-dirs"
        ),
      ],
    )

    [executable] = experiment.package([
      xm.dockerfile_container(
        executor_spec=executor.Spec(trusted=False),
        path=REPO,
        dockerfile=os.path.join(REPO, "xm", "Dockerfile"),
      )
    ])

    common = [
      f"--config.runner.world_size={n}",
      f"--config.runner.limit={LIMIT.value}",
      f"--xm_scratch={SCRATCH}",
      f"--xm_upload={OUTPUT_MOUNT}=gs://{OUTPUT_BUCKET}/{prefix}",
      f"--config.paths.tracks={OUTPUT_MOUNT}/tracks",
      f"--config.paths.review={OUTPUT_MOUNT}/review",
    ] + [f"--config.{kv}" for kv in CONFIG.value]

    for rank in range(n):
      experiment.add(xm.Job(executable, executor, args=[stage, f"--config.runner.rank={rank}", *common]))


if __name__ == "__main__":
  app.run(main)
