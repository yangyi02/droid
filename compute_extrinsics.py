import argparse
import copy
import gc
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import core.geometry
import core.io
import core.physics
import core.runner


def init_camera_states(scene_constants, extrinsics_db):
  print("  Initializing camera 3D states...")
  wrist_serial = scene_constants["meta"]["wrist_serial"]
  robot_data = scene_constants["robot"]
  n_frames = len(robot_data["T_ee_base_all"])

  episode_id = scene_constants["meta"]["episode_id"]
  episode_extrinsics = extrinsics_db.get(episode_id, {})

  scene_state = {}

  for cam_id in scene_constants["camera"].keys():
    if cam_id == wrist_serial:
      base_ext = robot_data["T_cam_ee_init"]
      cam_trajectory = robot_data["T_ee_base_all"] @ base_ext
    elif cam_id in episode_extrinsics:
      ext_data = episode_extrinsics[cam_id]
      ext_vec = ext_data.get("extrinsics", ext_data) if isinstance(ext_data, dict) else ext_data
      base_ext = core.geometry.make_4x4(ext_vec)
      cam_trajectory = np.tile(base_ext, (n_frames, 1, 1))
      print(f"    Loaded pre-calibrated extrinsics for camera [{cam_id}] from metadata.")
    else:
      base_ext = None
      cam_trajectory = None

    scene_state[cam_id] = {
        "base_extrinsic": base_ext,
        "extrinsics": cam_trajectory,
    }

  return scene_state


def extract_robot_clouds(cam_id, scene_constants, pb_renderer, base_extrinsic,
                         device):
  is_wrist = (cam_id == scene_constants['meta']['wrist_serial'])
  T_ee_base_all = scene_constants['robot']['T_ee_base_all']
  cam_data = scene_constants['camera'][cam_id]
  K_mat = cam_data['K_mat']

  cache_X, cache_obs = [], []
  n_frames = len(scene_constants['robot']['joint_positions'])
  for t in range(n_frames):
    pb_renderer.update_robot_pose(
        scene_constants['robot']['joint_positions'][t],
        scene_constants['robot']['gripper_positions'][t])
    d_obs = cam_data['raw_depth'][t].astype(np.float32)

    if is_wrist:
      pts_cam = core.physics.get_foreground_gripper_points(
          T_ee_base_all[t] @ base_extrinsic, K_mat, d_obs, pb_renderer, device)
      if pts_cam is None:
        continue
      cache_X.append(torch.tensor((base_extrinsic @ pts_cam)[:3, :].T,
                                  dtype=torch.float32, device=device))
    else:
      pts_world = core.physics.get_foreground_robot_points(
          base_extrinsic, K_mat, d_obs, pb_renderer, device)
      if pts_world is None:
        continue
      cache_X.append(pts_world)
    cache_obs.append(torch.tensor(d_obs, dtype=torch.float32,
                                  device=device)[None, ...])

  if not cache_X:
    return None, None
  return torch.stack(cache_X), torch.stack(cache_obs)


def robot_depth_loss(batch_X, T_opt, K, batch_obs, is_wrist):
  if is_wrist:
    return core.physics.compute_wrist_loss_batched(batch_X, T_opt, K, batch_obs)
  return core.physics.compute_robot_loss_batched(batch_X, T_opt, K, batch_obs)


def per_camera_alignment(scene_constants, pb_renderer, prev_scene_state,
                         device, outer_steps=1, inner_steps=500):
  print("\nUnified camera-robot alignment (external + wrist)...")
  wrist_cam = scene_constants['meta']['wrist_serial']
  scene_state = copy.deepcopy(prev_scene_state)
  T_ee_base_all = scene_constants['robot']['T_ee_base_all']
  n_frames = len(scene_constants['robot']['joint_positions'])

  for cam in scene_constants['camera'].keys():
    is_wrist = (cam == wrist_cam)
    mode = "wrist (gripper-only)" if is_wrist else "external (full body)"
    print(f"\n  Optimizing [{mode}] camera: [{cam}] ...")

    K_t = torch.tensor(scene_constants['camera'][cam]['K_mat'],
                       dtype=torch.float32, device=device)
    T_init_t = torch.tensor(prev_scene_state[cam]['base_extrinsic'],
                            dtype=torch.float32, device=device)
    d_ext = torch.zeros(6, requires_grad=True, device=device)
    optimizer = optim.Adam([d_ext], lr=0.001)
    loss_rob = None

    print(f"      {outer_steps} x {inner_steps} steps, "
          f"re-rendering the cloud between them...")
    for outer in range(outer_steps):
      with torch.no_grad():
        T_cur = (T_init_t @ core.geometry.make_T(d_ext, device)).cpu().numpy()
      batch_X, batch_obs = extract_robot_clouds(
          cam, scene_constants, pb_renderer, T_cur, device)
      for _ in range(inner_steps):
        optimizer.zero_grad()
        loss_rob = robot_depth_loss(
            batch_X, T_init_t @ core.geometry.make_T(d_ext, device), K_t,
            batch_obs, is_wrist)
        loss_rob.backward()
        optimizer.step()

      with torch.no_grad():
        rot_deg = torch.norm(d_ext[:3]).item() * (180.0 / np.pi)
        shift_mm = torch.norm(d_ext[3:]).item() * 1000.0
      print(f"        Outer {outer + 1}/{outer_steps} | frames "
            f"{len(batch_X)} | Loss: {loss_rob.item():.4f} | "
            f"Shift: {shift_mm:.2f}mm | Rot: {rot_deg:.2f}°")

    if loss_rob is None:
      continue

    with torch.no_grad():
      T_final_np = (T_init_t @ core.geometry.make_T(d_ext, device)).cpu().numpy()
      shift_mm = torch.norm(d_ext[3:]).item() * 1000.0
      rot_deg = torch.norm(d_ext[:3]).item() * (180.0 / np.pi)
      print(f"  [{cam}] Alignment done! Loss: {loss_rob.item():.4f} "
            f"(shift: {shift_mm:.2f}mm, rot: {rot_deg:.2f}°)")

      scene_state[cam]['base_extrinsic'] = T_final_np
      scene_state[cam]['extrinsics'] = (
          T_ee_base_all @ T_final_np if is_wrist
          else np.tile(T_final_np, (n_frames, 1, 1))
      )

  return scene_state


def batched_chamfer_distance(p1, p2, device):
  dist_matrix = torch.cdist(p1, p2)
  min_dist_12 = torch.min(dist_matrix, dim=2)[0]
  min_dist_21 = torch.min(dist_matrix, dim=1)[0]

  valid_12 = min_dist_12 < 0.05
  valid_21 = min_dist_21 < 0.05

  loss = torch.tensor(0.0, device=device)
  if valid_12.any():
    loss += min_dist_12[valid_12].mean()
  if valid_21.any():
    loss += min_dist_21[valid_21].mean()

  overlap_ratio = (valid_12.sum() + valid_21.sum()) / (p1.shape[0] * (p1.shape[1] + p2.shape[1]) + 1e-6)
  return loss, overlap_ratio.item()


def get_cam_points_local_t(t, cam_data, device, n_points=2000):
  depth = cam_data["raw_depth"][t].astype(np.float32)
  K_mat_np = cam_data["K_mat"]

  valid_mask = (depth > 0.) & (depth < 1.5)
  vs, us = np.where(valid_mask)
  if len(us) < 100:
    return None

  zs_obs = depth[vs, us]
  x_c = (us - K_mat_np[0, 2]) * zs_obs / K_mat_np[0, 0]
  y_c = (vs - K_mat_np[1, 2]) * zs_obs / K_mat_np[1, 1]

  P_cam = np.stack([x_c, y_c, zs_obs, np.ones_like(zs_obs)], axis=0)
  if P_cam.shape[1] < 100:
    return None

  idx = np.random.choice(P_cam.shape[1], n_points, replace=(P_cam.shape[1] <= n_points))
  return torch.tensor(P_cam[:, idx], dtype=torch.float32, device=device)


def global_joint_alignment(scene_constants, prev_scene_state, pb_renderer,
                           device, lr=0.001, n_steps=500,
                           chamfer_weight=1.0, robot_weight=1.0,
                           chamfer_n_points=2000):
  print(f"\nGlobal joint optimization "
        f"(Chamfer + Robot + Wrist, lr={lr})...")
  wrist_cam = scene_constants['meta']['wrist_serial']
  ext_cams = [c for c in scene_constants['camera'].keys() if c != wrist_cam]
  cam1, cam2 = ext_cams[0], ext_cams[1]
  n_frames = len(scene_constants['robot']['joint_positions'])
  T_ee_all = scene_constants['robot']['T_ee_base_all']

  print(f"  Rendering robot physical point clouds...")
  batch_X1, batch_obs1 = extract_robot_clouds(
      cam1, scene_constants, pb_renderer,
      prev_scene_state[cam1]['base_extrinsic'], device)
  batch_X2, batch_obs2 = extract_robot_clouds(
      cam2, scene_constants, pb_renderer,
      prev_scene_state[cam2]['base_extrinsic'], device)
  batch_P_ee, batch_obs_w = extract_robot_clouds(
      wrist_cam, scene_constants, pb_renderer,
      prev_scene_state[wrist_cam]['base_extrinsic'], device)

  print(f"  Extracting Chamfer environment point clouds...")
  cache_Pc1, cache_Pc2, cache_Pcw, cache_Tee = [], [], [], []

  for t in range(n_frames):
    pc1 = get_cam_points_local_t(t, scene_constants['camera'][cam1], device, n_points=chamfer_n_points)
    pc2 = get_cam_points_local_t(t, scene_constants['camera'][cam2], device, n_points=chamfer_n_points)
    pcw = get_cam_points_local_t(t, scene_constants['camera'][wrist_cam], device, n_points=chamfer_n_points)

    if pc1 is not None and pc2 is not None and pcw is not None:
      cache_Pc1.append(pc1)
      cache_Pc2.append(pc2)
      cache_Pcw.append(pcw)
      cache_Tee.append(torch.tensor(T_ee_all[t], dtype=torch.float32, device=device))

  batch_Pc1, batch_Pc2, batch_Pcw = torch.stack(cache_Pc1), torch.stack(cache_Pc2), torch.stack(cache_Pcw)
  batch_Tee = torch.stack(cache_Tee)

  K_t1 = torch.tensor(scene_constants['camera'][cam1]['K_mat'], dtype=torch.float32, device=device)
  K_t2 = torch.tensor(scene_constants['camera'][cam2]['K_mat'], dtype=torch.float32, device=device)
  K_t_w = torch.tensor(scene_constants['camera'][wrist_cam]['K_mat'], dtype=torch.float32, device=device)

  d1 = torch.zeros(6, requires_grad=True, device=device)
  d2 = torch.zeros(6, requires_grad=True, device=device)
  dhe = torch.zeros(6, requires_grad=True, device=device)
  optimizer = optim.Adam([d1, d2, dhe], lr=lr)

  T1_init_t = torch.tensor(prev_scene_state[cam1]['base_extrinsic'], dtype=torch.float32, device=device)
  T2_init_t = torch.tensor(prev_scene_state[cam2]['base_extrinsic'], dtype=torch.float32, device=device)
  Tee_init_t = torch.tensor(prev_scene_state[wrist_cam]['base_extrinsic'], dtype=torch.float32, device=device)

  print(f"  Data ready! Launching GPU joint optimization engine ({n_steps} steps)...")
  for step in range(n_steps):
    optimizer.zero_grad()

    T1_opt = T1_init_t @ core.geometry.make_T(d1, device)
    T2_opt = T2_init_t @ core.geometry.make_T(d2, device)
    Tee_opt = Tee_init_t @ core.geometry.make_T(dhe, device)

    bc1 = (T1_opt @ batch_Pc1)[:, :3, :].transpose(1, 2)
    bc2 = (T2_opt @ batch_Pc2)[:, :3, :].transpose(1, 2)
    T_wrist_c2w = batch_Tee @ Tee_opt
    bcw = torch.bmm(T_wrist_c2w, batch_Pcw)[:, :3, :].transpose(1, 2)

    l12, o12 = batched_chamfer_distance(bc1, bc2, device)
    l1w, o1w = batched_chamfer_distance(bc1, bcw, device)
    l2w, o2w = batched_chamfer_distance(bc2, bcw, device)
    loss_chamfer = l12 + l1w + l2w

    l_rob1 = robot_depth_loss(batch_X1, T1_opt, K_t1, batch_obs1, is_wrist=False)
    l_rob2 = robot_depth_loss(batch_X2, T2_opt, K_t2, batch_obs2, is_wrist=False)
    l_wrist = robot_depth_loss(batch_P_ee, Tee_opt, K_t_w, batch_obs_w, is_wrist=True)

    loss_total = chamfer_weight * loss_chamfer + robot_weight * (l_rob1 + l_rob2 + l_wrist)

    loss_total.backward()
    optimizer.step()

    if step % 50 == 0 or step == n_steps - 1:
      bg_overlap = (o12 + o1w + o2w) / 3.0 * 100
      shift_c1 = torch.norm(d1[3:]).item() * 1000
      shift_c2 = torch.norm(d2[3:]).item() * 1000
      shift_w = torch.norm(dhe[3:]).item() * 1000
      log_parts = [
          f"Step {step:03d}",
          f"Chmf: {loss_chamfer.item():.4f}",
          f"Rob1: {l_rob1.item():.4f}", f"Rob2: {l_rob2.item():.4f}",
          f"Wrst: {l_wrist.item():.4f}",
      ]
      log_parts.extend([
          f"BG Overlap: {bg_overlap:.1f}%",
          f"Shift: C1: {shift_c1:.2f}mm, C2: {shift_c2:.2f}mm, W: {shift_w:.2f}mm",
      ])
      print(f"    {' | '.join(log_parts)}")

  with torch.no_grad():
    final_p1 = (T1_init_t @ core.geometry.make_T(d1, device)).cpu().numpy()
    final_p2 = (T2_init_t @ core.geometry.make_T(d2, device)).cpu().numpy()
    final_cam_ee = (Tee_init_t @ core.geometry.make_T(dhe, device)).cpu().numpy()

  print("\nGlobal joint optimization complete!")

  ultimate_scene_state = {c: {} for c in scene_constants['camera'].keys()}
  ultimate_scene_state[cam1].update({
      "base_extrinsic": final_p1,
      "extrinsics": np.tile(final_p1, (n_frames, 1, 1)),
  })
  ultimate_scene_state[cam2].update({
      "base_extrinsic": final_p2,
      "extrinsics": np.tile(final_p2, (n_frames, 1, 1)),
  })
  ultimate_scene_state[wrist_cam].update({
      "base_extrinsic": final_cam_ee,
      "extrinsics": T_ee_all @ final_cam_ee,
  })

  return ultimate_scene_state


def export_extrinsics(scene_constants, scene_state,
                      export_root=os.path.join(core.io.OUTPUT_ROOT, "extrinsics")):
  ep_str = scene_constants["meta"]["episode_id"]
  wrist_serial = scene_constants["meta"]["wrist_serial"]
  ep_dir = os.path.abspath(os.path.expanduser(os.path.join(export_root, ep_str)))
  fname = "extrinsics.json"

  for cam_id, state in scene_state.items():
    if state.get("base_extrinsic") is None or state.get("extrinsics") is None:
      continue

    cam_dir = os.path.join(ep_dir, cam_id)
    os.makedirs(cam_dir, exist_ok=True)

    payload = {
        "base_extrinsic": state["base_extrinsic"].astype(np.float64).tolist(),
        "extrinsics": state["extrinsics"].astype(np.float64).tolist(),
        "is_wrist": (cam_id == wrist_serial),
    }

    out_path = os.path.join(cam_dir, fname)
    with open(out_path, "w") as f:
      json.dump(payload, f, indent=2)

  print(f"  Extrinsics saved to {ep_dir}/*/{fname}")
  return ep_dir


def _has_final_extrinsics(ep_dir):
  return os.path.isdir(ep_dir) and any(
      os.path.exists(os.path.join(ep_dir, cam, "extrinsics.json"))
      for cam in os.listdir(ep_dir))


def process_episode(ep_id, pb_renderer, extrinsics_db, depth_root, export_root,
                    device):
  scene_constants = core.io.load_depth_data(ep_id, depth_root)

  init_state = init_camera_states(scene_constants, extrinsics_db)

  aligned_state = per_camera_alignment(
      scene_constants, pb_renderer, init_state, device)

  joint_state = global_joint_alignment(
      scene_constants, aligned_state, pb_renderer, device,
      lr=0.001, n_steps=500, robot_weight=1.0)

  export_extrinsics(scene_constants, joint_state, export_root=export_root)

  gc.collect()
  torch.cuda.empty_cache()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="DROID Stage 2: Camera Extrinsics Calibration")
  core.runner.add_sharding_args(parser)
  parser.add_argument("--depth_root", type=str,
                      default=os.path.join(core.io.OUTPUT_ROOT, "depth"),
                      help="Root directory of depth outputs")
  parser.add_argument("--export_root", type=str,
                      default=os.path.join(core.io.OUTPUT_ROOT, "extrinsics"),
                      help="Root directory for extrinsics output")
  args = parser.parse_args()

  print("DROID Stage 2: Camera Extrinsics Calibration Pipeline")
  device = core.io.get_accelerator()
  serials_db, _, _, extrinsics_db, _ = core.io.load_metadata()
  pb_renderer = core.physics.PyBulletRenderer(gpu=True)
  print(f"  Using PyBulletRenderer (EGL: {pb_renderer.gpu})")

  target = core.runner.shard_episodes(
      core.runner.list_episode_dirs(args.depth_root),
      args.rank, args.world_size, args.limit)
  export_abs = os.path.abspath(os.path.expanduser(args.export_root))
  done = {ep for ep in target
          if _has_final_extrinsics(os.path.join(export_abs, ep))}

  core.runner.run_episodes(
      target,
      lambda ep_id: process_episode(
          ep_id, pb_renderer, extrinsics_db,
          args.depth_root, args.export_root, device),
      rank=args.rank, world_size=args.world_size,
      done=done, stage="Stage 2")
