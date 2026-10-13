import os

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc

STAGES = ("depth", "extrinsics", "scene", "tracks", "metrics", "review")

STAGE = flags.DEFINE_enum("stage", None, STAGES, "Pipeline stage.", required=True)
WORKERS = flags.DEFINE_integer("workers", 8, "Number of parallel GPU workers.")
LIMIT = flags.DEFINE_integer("limit", -1, "Only the first N episodes by name; -1 for all.")
GPU = flags.DEFINE_string("gpu", "a100", "GPU type per worker.")
LOCATION = flags.DEFINE_string("location", "us-central1", "GCP region.")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = "/app/data/input/robotics/droid_raw"
OUTPUT_DIR = "/app/data/output/droid"


def main(argv):
  stage = STAGE.value
  workers = WORKERS.value
  extra = argv[1:]

  with xm_abc.create_experiment(experiment_title=f"droid {stage} {workers}x{GPU.value}") as experiment:
    executor = xm_abc.Gcp(
      requirements=xm.JobRequirements(**{GPU.value: 1}, location=LOCATION.value),
      gcs_volumes=[
        xm_abc.GcsVolume(
          bucket="gresearch", directory_path="robotics/droid_raw", mount_path=INPUT_DIR, mount_options="implicit-dirs"
        ),
        xm_abc.GcsVolume(
          bucket="dm-tapnet", directory_path="tmp/droid_v2", mount_path=OUTPUT_DIR, mount_options="implicit-dirs"
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

    for rank in range(workers):
      args = [
        f"compute_{stage}.py",
        f"--config.runner.rank={rank}",
        f"--config.runner.world_size={workers}",
        f"--config.runner.limit={LIMIT.value}",
        f"--config.paths.tracks={OUTPUT_DIR}/tracks",
        f"--config.paths.review={OUTPUT_DIR}/review",
        *extra,
      ]
      experiment.add(xm.Job(executable, executor, args=args))


if __name__ == "__main__":
  app.run(main)
