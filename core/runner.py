import os
import random


def add_sharding_args(parser):
  parser.add_argument(
    "--rank", type=int, default=0, help="Rank of this process (for multi-worker sharding)"
  )
  parser.add_argument("--world_size", type=int, default=1, help="Total number of parallel workers")
  parser.add_argument(
    "--limit", type=int, default=-1, help="Limit total episodes to process (-1 = all)"
  )
  return parser


def list_episode_dirs(root):
  root = os.path.abspath(os.path.expanduser(root))
  return set(os.listdir(root)) if os.path.isdir(root) else set()


def shard_episodes(episode_ids, rank, world_size, limit=-1, seed=42):
  episode_ids = sorted(episode_ids)
  random.Random(seed).shuffle(episode_ids)
  if limit > 0:
    episode_ids = episode_ids[:limit]
  return episode_ids[rank::world_size]


def run_episodes(episode_ids, process, rank=0, world_size=1, done=(), stage="Pipeline"):
  done = set(done)
  todo = [ep for ep in episode_ids if ep not in done]
  skipped = len(episode_ids) - len(todo)
  print(
    f"Rank {rank}/{world_size}: {len(todo)} episodes to process"
    + (f" ({skipped} already done)" if skipped else "")
  )

  succeeded = []
  for idx, ep_id in enumerate(todo):
    print(f"\n[{idx + 1}/{len(todo)}] Episode: {ep_id}")
    process(ep_id)
    succeeded.append(ep_id)

  print(f"\n{stage} complete! {len(succeeded)}/{len(todo)} episodes succeeded.")
  return succeeded
