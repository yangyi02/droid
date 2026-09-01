"""Batch-runner plumbing shared by the pipeline stages.

Every stage has the same shape around its actual work: decide which episodes
it can process, split them across ranks, skip the ones already finished, then
run the rest one at a time without letting a single bad episode take the job
down. This module is that shape; what a stage does to an episode stays in the
stage.
"""

import os
import random
import traceback


def add_sharding_args(parser):
  """Add the --rank / --world_size / --limit trio every stage shares."""
  parser.add_argument("--rank", type=int, default=0,
                      help="Rank of this process (for multi-worker sharding)")
  parser.add_argument("--world_size", type=int, default=1,
                      help="Total number of parallel workers")
  parser.add_argument("--limit", type=int, default=-1,
                      help="Limit total episodes to process (-1 = all)")
  return parser


def list_episode_dirs(root):
  """The episode ids directly under `root`, as a set.

  A bare listdir, deliberately: these roots are normally gcsfuse mounts where
  every os.path.isdir is a separate GCS request, and thousands of them stall
  for minutes. Entries that turn out not to be usable episodes are left for the
  stage to fail on individually.

  A missing root reads as no episodes rather than raising, so a stage can ask
  about output that does not exist yet.
  """
  root = os.path.abspath(os.path.expanduser(root))
  return set(os.listdir(root)) if os.path.isdir(root) else set()


def shard_episodes(episode_ids, rank, world_size, limit=-1, seed=42):
  """This rank's share of `episode_ids`, shuffled and optionally truncated.

  The shuffle comes before the split because it is what balances the load:
  episode ids sort by lab and date, and episode length varies several-fold, so
  a contiguous slice can hand one rank a run of long ones while another idles.
  A fixed seed lets every rank derive the same ordering without coordinating,
  and a private Random keeps that draw off the global RNG, which stages use for
  their own sampling.

  --limit truncates the shuffled pool before sharding, so it means "this many
  episodes across the job", not "this many per rank".
  """
  episode_ids = sorted(episode_ids)
  random.Random(seed).shuffle(episode_ids)
  if limit > 0:
    episode_ids = episode_ids[:limit]
  return episode_ids[rank::world_size]


def run_episodes(episode_ids, process, rank=0, world_size=1, done=(),
                 stage="Pipeline"):
  """Run `process(episode_id)` over `episode_ids`, one episode at a time.

  An episode counts as succeeded if `process` returns without raising. A raise
  is reported with its traceback and the run carries on: these are long batch
  jobs over thousands of episodes, and one unreadable SVO should not cost the
  rest of the shard.

  Args:
    process: callable taking an episode id. Anything it needs beyond that --
        models, roots, options -- belongs in a closure or a partial.
    done: episode ids already finished, skipped silently and left out of the
        totals. Build it from one directory listing (see list_episode_dirs)
        rather than a stat per episode.
    stage: name used in the closing summary.

  Returns:
    The ids that succeeded, in the order they ran.
  """
  done = set(done)
  todo = [ep for ep in episode_ids if ep not in done]
  skipped = len(episode_ids) - len(todo)
  print(f"Rank {rank}/{world_size}: {len(todo)} episodes to process"
        + (f" ({skipped} already done)" if skipped else ""))

  succeeded = []
  for idx, ep_id in enumerate(todo):
    print(f"\n[{idx + 1}/{len(todo)}] Episode: {ep_id}")
    try:
      process(ep_id)
      succeeded.append(ep_id)
    except Exception as e:
      print(f"  [FAIL] Episode {ep_id} failed: {e}")
      traceback.print_exc()

  print(f"\n{stage} complete! {len(succeeded)}/{len(todo)} episodes succeeded.")
  return succeeded
