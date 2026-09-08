import os
import random


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
  todo = [episode_id for episode_id in episode_ids if episode_id not in done]
  skipped = len(episode_ids) - len(todo)
  print(
    f"Rank {rank}/{world_size}: {len(todo)} episodes to process"
    + (f" ({skipped} already done)" if skipped else "")
  )

  succeeded = []
  for idx, episode_id in enumerate(todo):
    print(f"\n[{idx + 1}/{len(todo)}] Episode: {episode_id}")
    process(episode_id)
    succeeded.append(episode_id)

  print(f"\n{stage} complete! {len(succeeded)}/{len(todo)} episodes succeeded.")
  return succeeded
