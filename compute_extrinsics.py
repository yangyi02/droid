import copy
import gc
import itertools
import json
import os

import numpy as np
import torch
from absl import app
from ml_collections import config_flags
import torch.optim as optim

import config
import core.alignment
import core.geometry
import core.io
import core.physics
import core.runner


def world_extrinsics(base, ee_poses, is_wrist):
  if is_wrist:
    return ee_poses @ base
  return np.tile(base, (len(ee_poses), 1, 1))


def init_camera_states(episode, extrinsics_db):
  print("  Initializing camera 3D states...")
  wrist_cam_id = episode["meta"]["wrist_serial"]
  robot_data = episode["robot"]
  n_frames = len(robot_data["T_ee_base_all"])

  episode_id = episode["meta"]["episode_id"]
  episode_extrinsics = extrinsics_db.get(episode_id, {})

  poses = {}

  for cam_id in episode["camera"].keys():
    if cam_id == wrist_cam_id:
      T_cam2mount = robot_data["T_cam_ee_init"]
      cam_trajectory = robot_data["T_ee_base_all"] @ T_cam2mount
    elif cam_id in episode_extrinsics:
      db_entry = episode_extrinsics[cam_id]
      euler_6d = db_entry.get("extrinsics", db_entry) if isinstance(db_entry, dict) else db_entry
      T_cam2mount = core.geometry.pose_from_euler(euler_6d)
      cam_trajectory = np.tile(T_cam2mount, (n_frames, 1, 1))
      print(f"    Loaded pre-calibrated extrinsics for camera [{cam_id}] from metadata.")
    else:
      T_cam2mount = None
      cam_trajectory = None

    poses[cam_id] = {"base_extrinsic": T_cam2mount, "extrinsics": cam_trajectory}

  return poses


def per_camera_alignment(
  episode, pb_renderer, prev_poses, device, outer_steps=5, inner_steps=100, lr=0.001
):
  print("\nUnified camera-robot alignment (external + wrist)...")
  wrist_cam_id = episode['meta']['wrist_serial']
  poses = copy.deepcopy(prev_poses)
  T_ee_base_all = episode['robot']['T_ee_base_all']
  n_frames = len(episode['robot']['joint_positions'])

  for cam_id in episode['camera'].keys():
    is_wrist = cam_id == wrist_cam_id
    mode = "wrist (gripper-only)" if is_wrist else "external (full body)"
    print(f"\n  Optimizing [{mode}] camera: [{cam_id}] ...")

    cam_data = episode['camera'][cam_id]
    K = torch.tensor(cam_data['K'], dtype=torch.float32, device=device)
    T_cam2mount_init = torch.tensor(
      prev_poses[cam_id]['base_extrinsic'], dtype=torch.float32, device=device
    )
    depth_full = torch.tensor(
      np.asarray(cam_data['raw_depth'], dtype=np.float32), device=device
    ).unsqueeze(1)
    delta = torch.zeros(6, requires_grad=True, device=device)
    optimizer = optim.Adam([delta], lr=lr)
    loss_rob = None

    print(f"      {outer_steps} x {inner_steps} steps, re-rendering the cloud between them...")
    for outer in range(outer_steps):
      with torch.no_grad():
        T_cam2mount = (
          (T_cam2mount_init @ core.geometry.pose_from_axis_angle(delta, device)).cpu().numpy()
        )
      robot_points, depth_batch = core.alignment.extract_robot_clouds(
        cam_id, episode, pb_renderer, T_cam2mount, device, depth_full
      )
      for _ in range(inner_steps):
        optimizer.zero_grad()
        loss_rob = core.alignment.depth_loss_batched(
          robot_points,
          T_cam2mount_init @ core.geometry.pose_from_axis_angle(delta, device),
          K,
          depth_batch,
        )
        loss_rob.backward()
        optimizer.step()

      with torch.no_grad():
        rot_deg = torch.norm(delta[3:]).item() * (180.0 / np.pi)
        shift_mm = torch.norm(delta[:3]).item() * 1000.0
      print(
        f"        Outer {outer + 1}/{outer_steps} | frames "
        f"{len(robot_points)} | Loss: {loss_rob.item():.4f} | "
        f"Shift: {shift_mm:.2f}mm | Rot: {rot_deg:.2f}°"
      )

    if loss_rob is None:
      continue

    with torch.no_grad():
      T_cam2mount_final = (
        (T_cam2mount_init @ core.geometry.pose_from_axis_angle(delta, device)).cpu().numpy()
      )
      shift_mm = torch.norm(delta[:3]).item() * 1000.0
      rot_deg = torch.norm(delta[3:]).item() * (180.0 / np.pi)
      print(
        f"  [{cam_id}] Alignment done! Loss: {loss_rob.item():.4f} "
        f"(shift: {shift_mm:.2f}mm, rot: {rot_deg:.2f}°)"
      )

      poses[cam_id]['base_extrinsic'] = T_cam2mount_final
      poses[cam_id]['extrinsics'] = world_extrinsics(T_cam2mount_final, T_ee_base_all, is_wrist)

  return poses


def global_joint_alignment(
  episode,
  prev_poses,
  pb_renderer,
  device,
  lr=0.001,
  n_steps=500,
  chamfer_weight=1.0,
  robot_weight=1.0,
  chamfer_n_points=2000,
  max_depth=1.5,
  match_radius=0.05,
):
  print(f"\nGlobal joint optimization (Chamfer + Robot + Wrist, lr={lr})...")
  wrist_cam_id = episode['meta']['wrist_serial']
  cam_ids = [c for c in episode['camera'] if c != wrist_cam_id] + [wrist_cam_id]
  pairs = list(itertools.combinations(cam_ids, 2))
  base = {
    c: torch.tensor(prev_poses[c]['base_extrinsic'], dtype=torch.float32, device=device)
    for c in cam_ids
  }

  robot_points, depth_batch, K = core.alignment.robot_clouds(
    episode, prev_poses, pb_renderer, device
  )
  env, ee_poses = core.alignment.scene_clouds(episode, device, chamfer_n_points, max_depth)

  fixed_cam_ids = [c for c in cam_ids if c != wrist_cam_id]
  label = {c: str(i + 1) for i, c in enumerate(fixed_cam_ids)} | {wrist_cam_id: "W"}

  delta = {cam_id: torch.zeros(6, requires_grad=True, device=device) for cam_id in cam_ids}
  optimizer = optim.Adam(list(delta.values()), lr=lr)

  print(f"  Data ready! Launching GPU joint optimization engine ({n_steps} steps)...")
  for step in range(n_steps):
    optimizer.zero_grad()

    pose = {
      cam_id: base[cam_id] @ core.geometry.pose_from_axis_angle(delta[cam_id], device)
      for cam_id in cam_ids
    }
    chamfer, overlap = core.alignment.chamfer_overlap(
      env, ee_poses, pose, wrist_cam_id, pairs, match_radius
    )
    robot = core.alignment.robot_depth_loss(robot_points, depth_batch, K, pose, max_depth)

    loss_total = chamfer_weight * sum(chamfer.values()) + robot_weight * sum(robot.values())
    loss_total.backward()
    optimizer.step()

    if step % 100 == 0 or step == n_steps - 1:
      parts = [f"Step {step:03d}"]
      parts += [f"Ch{label[a]}{label[b]}: {chamfer[a, b].item():.4f}" for a, b in pairs]
      parts += [f"Rob{label[cam_id]}: {robot[cam_id].item():.4f}" for cam_id in cam_ids]
      parts.append(f"Overlap: {sum(overlap.values()).item() / len(pairs) * 100:.1f}%")
      shifts = ", ".join(
        f"{label[cam_id]}: {torch.norm(delta[cam_id][:3]).item() * 1000:.2f}mm"
        for cam_id in cam_ids
      )
      parts.append(f"Shift: {shifts}")
      print(f"    {' | '.join(parts)}")

  with torch.no_grad():
    final = {
      cam_id: (base[cam_id] @ core.geometry.pose_from_axis_angle(delta[cam_id], device))
      .cpu()
      .numpy()
      for cam_id in cam_ids
    }

  print("\nGlobal joint optimization complete!")

  T_ee2base = episode['robot']['T_ee_base_all']
  return {
    cam_id: {
      "base_extrinsic": final[cam_id],
      "extrinsics": world_extrinsics(final[cam_id], T_ee2base, cam_id == wrist_cam_id),
    }
    for cam_id in episode['camera']
  }


def export_extrinsics(episode, poses, export_root):
  episode_id = episode["meta"]["episode_id"]
  wrist_cam_id = episode["meta"]["wrist_serial"]
  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(export_root, episode_id)))
  fname = "extrinsics.json"

  for cam_id, state in poses.items():
    if state.get("base_extrinsic") is None or state.get("extrinsics") is None:
      continue

    cam_dir = os.path.join(ep_dir, cam_id)
    os.makedirs(cam_dir, exist_ok=True)

    payload = {
      "base_extrinsic": state["base_extrinsic"].astype(np.float64).tolist(),
      "extrinsics": state["extrinsics"].astype(np.float64).tolist(),
      "is_wrist": (cam_id == wrist_cam_id),
    }

    out_path = os.path.join(cam_dir, fname)
    with open(out_path, "w") as f:
      json.dump(payload, f, indent=2)

  print(f"  Extrinsics saved to {ep_dir}/*/{fname}")
  return ep_dir


def _has_final_extrinsics(ep_dir):
  return os.path.isdir(ep_dir) and any(
    os.path.exists(os.path.join(ep_dir, cam_id, "extrinsics.json")) for cam_id in os.listdir(ep_dir)
  )


def process_episode(episode_id, pb_renderer, extrinsics_db, device, config):
  episode = core.io.load_depth_data(episode_id, config.paths.depth)

  init_state = init_camera_states(episode, extrinsics_db)

  aligned_state = per_camera_alignment(
    episode,
    pb_renderer,
    init_state,
    device,
    outer_steps=config.extrinsics.outer_steps,
    inner_steps=config.extrinsics.inner_steps,
    lr=config.extrinsics.lr,
  )

  joint_state = global_joint_alignment(
    episode,
    aligned_state,
    pb_renderer,
    device,
    lr=config.extrinsics.lr,
    n_steps=config.extrinsics.n_steps,
    chamfer_n_points=config.extrinsics.chamfer_n_points,
    max_depth=config.extrinsics.max_depth,
    match_radius=config.extrinsics.chamfer_match_radius,
  )

  export_extrinsics(episode, joint_state, export_root=config.paths.extrinsics)

  gc.collect()
  torch.cuda.empty_cache()


def main(_):
  config = config_flag.value
  device = core.io.get_accelerator()
  serials_db, _, _, extrinsics_db, _ = core.io.load_metadata(config)
  pb_renderer = core.physics.PyBulletRenderer(config.paths.urdf, gpu=config.render.gpu)

  target = core.runner.shard_episodes(
    core.runner.list_episode_dirs(config.paths.depth),
    config.runner.rank,
    config.runner.world_size,
    config.runner.limit,
  )
  export_abs = os.path.abspath(os.path.expanduser(config.paths.extrinsics))
  done = {
    episode_id
    for episode_id in target
    if _has_final_extrinsics(os.path.join(export_abs, episode_id))
  }

  def run_one(episode_id):
    process_episode(episode_id, pb_renderer, extrinsics_db, device, config)

  core.runner.run_episodes(
    target,
    run_one,
    rank=config.runner.rank,
    world_size=config.runner.world_size,
    done=done,
    stage="Stage 2",
  )


if __name__ == "__main__":
  config_flag = config_flags.DEFINE_config_file("config", config.__file__)
  app.run(main)
