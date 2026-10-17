import os

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc

STAGES = ("depth", "extrinsics", "tracks", "review")

STAGE = flags.DEFINE_enum("stage", None, STAGES, "Pipeline stage.", required=True)
WORKERS = flags.DEFINE_integer("workers", 8, "Number of parallel GPU workers.")
RANKS = flags.DEFINE_list("ranks", None, "Only launch these ranks of --workers; default all.")
LIMIT = flags.DEFINE_integer("limit", -1, "Only the first N episodes by name; -1 for all.")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = "/app/data/input/robotics/droid_raw"
OUTPUT_DIR = "/app/data/output/droid"


def main(argv):
  stage = STAGE.value
  workers = WORKERS.value
  ranks = [int(rank) for rank in RANKS.value or range(workers)]
  extra = argv[1:]

  with xm_abc.create_experiment(experiment_title=f"droid {stage} {len(ranks)}xb40") as experiment:
    executor = xm_abc.Gcp(
      requirements=xm.JobRequirements(b40_48th=48, location="dmgcp-us-west4"),
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

    for rank in ranks:
      args = [
        f"compute_{stage}.py",
        f"--config.runner.rank={rank}",
        f"--config.runner.world_size={workers}",
        f"--config.runner.limit={LIMIT.value}",
        *extra,
      ]
      experiment.add(xm.Job(executable, executor, args=args))


if __name__ == "__main__":
  app.run(main)
