import os
import random
import traceback


class SkipEpisode(Exception):
  pass


def list_episode_dirs(root):
  root = os.path.abspath(os.path.expanduser(root))
  return set(os.listdir(root)) if os.path.isdir(root) else set()


def shard_episodes(episode_ids, rank, world_size, limit, seed=0):
  episode_ids = sorted(episode_ids)
  random.Random(seed).shuffle(episode_ids)
  if limit > 0:
    episode_ids = episode_ids[:limit]
  return episode_ids[rank::world_size]


def run_episodes(episode_ids, process, rank, world_size, done, stage):
  done = set(done)
  todo = [episode_id for episode_id in episode_ids if episode_id not in done]
  already_done = len(episode_ids) - len(todo)
  print(
    f"Rank {rank}/{world_size}: {len(todo)} episodes to process"
    + (f" ({already_done} already done)" if already_done else "")
  )

  succeeded, skipped, failed = [], [], []
  for idx, episode_id in enumerate(todo):
    print(f"\n[{idx + 1}/{len(todo)}] Episode: {episode_id}")
    try:
      process(episode_id)
      succeeded.append(episode_id)
    except SkipEpisode as reason:
      print(f"  SKIP {episode_id}: {reason}")
      skipped.append(episode_id)
    except Exception:
      failed.append(episode_id)
      traceback.print_exc()

  print(f"\n{stage}: {len(succeeded)} succeeded, {len(skipped)} skipped, {len(failed)} failed of {len(todo)}.")
  for episode_id in failed:
    print(f"  FAILED {episode_id}")
  return succeeded
