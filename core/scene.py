import cv2
import numpy as np
import torch


def is_fixed(extrinsics):
  return bool(np.allclose(extrinsics, extrinsics[0], atol=1e-6))


def unproject(depth, K, T):
  v, u = torch.nonzero(depth > 0, as_tuple=True)
  z = depth[v, u]
  points = torch.stack([(u - float(K[0, 2])) / float(K[0, 0]) * z, (v - float(K[1, 2])) / float(K[1, 1]) * z, z], -1)
  return points @ T[:3, :3].T + T[:3, 3]


def project(points, K, T, height, width):
  cam = (points - T[:3, 3]) @ T[:3, :3]
  z = cam[:, 2]
  u = torch.round(float(K[0, 0]) * cam[:, 0] / z + float(K[0, 2])).long()
  v = torch.round(float(K[1, 1]) * cam[:, 1] / z + float(K[1, 2])).long()
  inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
  return (v.clamp(0, height - 1) * width + u.clamp(0, width - 1)), z, inside


def tolerance(z, focal_baseline, pose_error, config):
  return config.scene.tolerance + focal_baseline * pose_error / z.clamp(min=1e-6) ** 2


def is_static(points, measured, K, poses, frames, focal_baseline, pose_error, config):
  height, width = measured.shape[1:]
  near = torch.zeros(len(points), device=points.device)
  free = torch.zeros_like(near)
  for e in frames:
    index, z, inside = project(points, K, poses[e], height, width)
    observed = torch.where(inside, measured[e].flatten()[index], 0.0)
    seen = observed > 0
    gap = focal_baseline / observed.clamp(min=1e-6) - focal_baseline / z.clamp(min=1e-6)
    allowed = tolerance(z, focal_baseline, pose_error, config)
    near += seen & (gap.abs() <= allowed)
    free += seen & (gap < -allowed)
  return (near >= 2) & (free <= config.scene.max_seen_through * (near + free))


def splat(points, K, T, focal_baseline, height, width, pose_error, config):
  index, z, inside = project(points, K, T, height, width)
  index, z = index[inside], z[inside]
  disparity = focal_baseline / z
  nearest = torch.full((height * width,), -torch.inf, device=points.device)
  nearest.scatter_reduce_(0, index, disparity, reduce="amax")
  member = disparity >= nearest[index] - 2 * tolerance(z, focal_baseline, pose_error, config)
  total = torch.zeros(height * width, device=points.device).scatter_add_(0, index[member], disparity[member])
  count = torch.zeros(height * width, device=points.device).scatter_add_(0, index[member], torch.ones_like(disparity[member]))
  return torch.where(count > 0, focal_baseline * count / total.clamp(min=1e-6), 0.0).reshape(height, width)


def static_depth(depth, arm, K, extrinsics, focal_baseline, config):
  n_frames, height, width = depth.shape
  measured = torch.where(arm, 0.0, depth)
  poses = torch.tensor(extrinsics, dtype=torch.float32, device=depth.device)
  frames = list(range(0, n_frames, config.scene.stride))
  fixed = is_fixed(extrinsics)
  pose_error = 0.0 if fixed else config.scene.pose_error
  cloud = []
  for s in frames:
    points = unproject(measured[s], K, poses[s])
    cloud.append(points[is_static(points, measured, K, poses, frames, focal_baseline, pose_error, config)])
  targets = [0] if fixed else range(n_frames)
  refined = []
  for t in targets:
    same = [np.allclose(extrinsics[s], extrinsics[t], atol=1e-6) for s in frames]
    own = torch.cat([points for points, keep in zip(cloud, same) if keep])
    other = torch.cat([points for points, keep in zip(cloud, same) if not keep] or [own[:0]])
    view = splat(own, K, poses[t], focal_baseline, height, width, 0.0, config)
    fill = splat(other, K, poses[t], focal_baseline, height, width, pose_error, config)
    refined.append(torch.where(view > 0, view, fill))
  return torch.stack(refined)


def disparity(depth, focal_baseline):
  return torch.where(depth > 0, focal_baseline / depth.clamp(min=1e-6), 0.0)


def grow(mask, radius):
  return torch.nn.functional.max_pool2d(mask[:, None].float(), 2 * radius + 1, stride=1, padding=radius)[:, 0] > 0


def align_links(arm, links, raw, interior, allowed):
  aligned = arm.clone()
  for link in torch.unique(links[links >= 0]).tolist():
    on = links == link
    usable = on & interior & (raw > 0) & ((raw - arm).abs() <= 4 * allowed)
    gap = torch.where(usable, raw - arm, torch.nan).flatten(1)[:, ::4]
    offset = torch.nan_to_num(torch.nanmedian(gap, dim=1).values)
    aligned = torch.where(on, arm + offset[:, None, None], aligned)
  return aligned


def warp(depth, K, T_source, T_target):
  height, width = depth.shape
  index, z, inside = project(unproject(depth, K, T_source), K, T_target, height, width)
  nearest = torch.full((height * width,), torch.inf, device=depth.device)
  nearest.scatter_reduce_(0, index[inside], z[inside], reduce="amin")
  return torch.where(torch.isfinite(nearest), nearest, 0.0).reshape(height, width)


def spatial_support(r, allowed):
  padded = torch.nn.functional.pad(r[:, None], (1, 1, 1, 1))[:, 0]
  height, width = r.shape[1:]
  count = torch.zeros_like(r)
  for dy in (0, 1, 2):
    for dx in (0, 1, 2):
      if (dy, dx) != (1, 1):
        neighbour = padded[:, dy : dy + height, dx : dx + width]
        count += (neighbour > 0) & ((neighbour - r).abs() <= allowed)
  return count >= 4


def settle(raw, keep, K, extrinsics, focal_baseline, allowed, config):
  fixed = is_fixed(extrinsics)
  poses = torch.tensor(extrinsics, dtype=torch.float32, device=raw.device)
  r = disparity(raw, focal_baseline)
  coherent = spatial_support(r, allowed)
  settled = raw.clone()
  for t in range(len(raw)):
    frames = sorted(range(max(0, t - config.scene.window), min(len(raw), t + config.scene.window + 1)), key=lambda s: abs(s - t))[1:]
    views = [[raw[s]] if fixed else [warp(raw[s], K, poses[s], poses[t]), raw[s]] for s in frames]
    views = [[disparity(view, focal_baseline) for view in group] for group in views]
    reference = r[t]
    for d in (d for group in views for d in group):
      reference = torch.where(reference > 0, reference, d)
    total, count = torch.where(keep[t], r[t], 0.0), (keep[t] & (r[t] > 0)).float()
    for group in views:
      best = torch.zeros_like(reference)
      for d in group:
        best = torch.where((d > 0) & ((d - reference).abs() < (best - reference).abs()), d, best)
      ok = keep[t] & (best > 0) & ((best - reference).abs() <= allowed[t])
      total, count = total + torch.where(ok, best, 0.0), count + ok
    averaged = torch.where(count > 1, focal_baseline * count / total.clamp(min=1e-6), torch.where(coherent[t], raw[t], 0.0))
    settled[t] = torch.where(keep[t], averaged, raw[t])
  return settled


def nearest_class(holes, seeds):
  found = np.zeros_like(holes)
  for t in range(len(holes)):
    _, labels = cv2.distanceTransformWithLabels(holes[t].astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    classes = np.zeros(labels.max() + 1, bool)
    classes[labels[~holes[t]]] = seeds[t][~holes[t]]
    found[t] = holes[t] & classes[labels]
  return found


def refine_depth(raw, static, segmentation, K, extrinsics, focal_baseline, config):
  static = static.expand_as(raw)
  pose_error = 0.0 if is_fixed(extrinsics) else config.scene.pose_error
  robot, links = segmentation[:, 0] > 0, segmentation[:, 1].long()
  reference = torch.where(raw > 0, raw, torch.where(robot, segmentation[:, 0], static)).clamp(min=0.05)
  allowed = tolerance(reference, focal_baseline, pose_error, config)
  r, s = disparity(raw, focal_baseline), disparity(static, focal_baseline)

  interior = robot & ~grow(~robot, config.scene.halo)
  edge = grow(robot, config.scene.halo) & ~interior
  a = align_links(disparity(segmentation[:, 0], focal_baseline), links, r, interior, allowed)
  nearby = torch.nn.functional.max_pool2d(a[:, None], 2 * config.scene.halo + 1, stride=1, padding=config.scene.halo)[:, 0]

  agrees = (r > 0) & (s > 0) & (r <= s + 3 * allowed)
  in_front = (r > 0) & (r > a + allowed)
  beyond = (r > 0) & (r < a - allowed) & edge
  halo = ~robot & edge & (r > 0) & ((r - nearby).abs() <= allowed)
  foreground = (r > 0) & ~agrees & ((~robot & ~halo) | (robot & in_front))
  holes = (raw == 0) & ~robot
  solid = grow(~grow(~foreground, config.scene.halo // 2), config.scene.halo // 2)
  on_foreground = torch.tensor(nearest_class(holes.cpu().numpy(), solid.cpu().numpy()), device=raw.device)

  refined = settle(raw, foreground | on_foreground, K, extrinsics, focal_baseline, allowed, config)
  refined = torch.where(foreground & (refined == 0), static, refined)
  refined = torch.where(robot & ((~in_front & ~beyond) | (in_front & (refined == 0))), focal_baseline / a.clamp(min=1e-6), refined)
  refined = torch.where(halo, static, refined)
  refined = torch.where(holes & ~on_foreground, static, refined)
  return torch.where(agrees, static, refined)


def shift(depth, focal_baseline, offset):
  return torch.where(depth > 0, focal_baseline / (focal_baseline / depth.clamp(min=1e-6) + offset), 0.0)


def pair_samples(source, target, config, stride=8, every=10):
  samples = []
  frames = [0] if is_fixed(source["extrinsics"]) and is_fixed(target["extrinsics"]) else range(0, len(source["extrinsics"]), every)
  for t in frames:
    depth = source["static"][min(t, len(source["static"]) - 1)]
    height, width = target["static"].shape[1:]
    v, u = torch.nonzero(depth[::stride, ::stride] > 0, as_tuple=True)
    u, v = u * stride, v * stride
    K = source["K"]
    ray = torch.stack([(u - float(K[0, 2])) / float(K[0, 0]), (v - float(K[1, 2])) / float(K[1, 1]), torch.ones_like(u, dtype=torch.float32)], -1)
    T_source = torch.tensor(source["extrinsics"][t], device=depth.device)
    T_target = torch.tensor(target["extrinsics"][t], device=depth.device)
    relative = torch.linalg.inv(T_target) @ T_source
    points = ray * depth[v, u, None]
    index, z, inside = project(points @ T_source[:3, :3].T + T_source[:3, 3], target["K"], T_target, height, width)
    measured = target["static"][min(t, len(target["static"]) - 1)].flatten()[index]
    keep = inside & (measured > 0)
    samples.append(torch.stack([
      ray[keep] @ relative[2, :3], relative[2, 3].expand(int(keep.sum())),
      disparity(depth[v, u], source["focal_baseline"])[keep], disparity(measured, target["focal_baseline"])[keep],
    ], -1))
  return torch.cat(samples)


def robot_samples(camera, config, stride=4, every=10):
  frames = list(range(0, len(camera["raw"]), every))
  urdf = camera["segmentation"][frames, 0]
  robot = urdf > 0
  interior = robot & ~grow(~robot, config.scene.halo)
  raw = camera["raw"][frames]
  keep = (interior & (raw > 0))[:, ::stride, ::stride]
  measured = disparity(raw, camera["focal_baseline"])[:, ::stride, ::stride][keep]
  rendered = disparity(urdf, camera["focal_baseline"])[:, ::stride, ::stride][keep]
  return torch.stack([measured, rendered], -1)


def depth_offsets(cameras, config):
  pairs = [(a, b, pair_samples(cameras[a], cameras[b], config)) for a in range(len(cameras)) for b in range(len(cameras)) if a != b]
  robots = [robot_samples(camera, config) for camera in cameras]
  offsets = torch.zeros(len(cameras), device=pairs[0][2].device, requires_grad=True)
  cauchy = lambda r: torch.log1p((r / config.scene.tolerance) ** 2).mean()

  def loss():
    total = 0.0
    for a, b, (alpha, beta, source, measured) in (p[:2] + (p[2].T,) for p in pairs):
      z = alpha * cameras[a]["focal_baseline"] / (source + offsets[a]) + beta
      total = total + cauchy(measured + offsets[b] - cameras[b]["focal_baseline"] / z.clamp(min=1e-3))
    for c, (measured, rendered) in enumerate(r.T for r in robots):
      total = total + cauchy(measured + offsets[c] - rendered)
    return total

  optimizer = torch.optim.LBFGS([offsets], max_iter=100, line_search_fn="strong_wolfe")

  def closure():
    optimizer.zero_grad()
    value = loss()
    value.backward()
    return value

  optimizer.step(closure)
  return offsets.detach()
