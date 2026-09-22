"""What the other cameras can say about a hole in this one.

Stereo leaves holes exactly where surfaces are dark, thin or textureless, and a depth map that measured
nothing at a pixel has said nothing -- neither that the point is visible nor that it is hidden. The
other cameras are looking at the same scene on the same frame from somewhere else, so the question can
be put to them instead: walk the ray from this camera towards the point and ask whether anyone measured
a surface standing on it.

Only this frame is used. A model fused over the whole episode was measured and answered almost nothing
the other cameras had not already answered, while carrying the surfaces of everything that was ever
moved as occluders that are no longer there.
"""

import numpy as np

import core.geometry


def ray_samples(origins, targets, clearance, step, near):
  """Points along each ray, from near in front of its camera up to the clearance short of its target.

  Both ends are left out for the same reason: something is always there and it is never the answer. At
  the far end the target sits on a surface of its own. At the near end sits the camera -- the wrist one
  rides on the arm, and its body and cable are not in the URDF, so every ray it casts starts by running
  into itself."""
  direction = targets - origins
  length = np.linalg.norm(direction, axis=1, keepdims=True)
  reach = np.maximum(length - clearance[:, None], 0.0)

  distance = (np.arange(max(int(np.ceil(float(reach.max()) / step)), 1), dtype=np.float32) + 0.5) * step
  samples = origins[:, None, :] + direction[:, None, :] / np.maximum(length, 1e-9)[:, None, :] * distance[None, :, None]
  return samples, (distance[None, :] < reach) & (distance[None, :] >= near)


def blocked(origins, targets, clearance, step, near, episode, poses, t, robot_mask, slack, tolerance):
  """Whether any camera measured a surface between each camera centre and its point, on this frame.

  What a camera reads inside the robot's own silhouette says nothing about this: it is reading the arm,
  and it cannot see past it either. Counting those readings makes the arm its own occluder and the wrist
  camera never sees its gripper again, so each camera is only asked about the samples it sees past the
  robot. Where every camera is looking at the arm, nobody can say, and nothing is claimed.

  A camera also cannot tell a sample from the point itself unless it sees the two apart, and it is its
  own line of sight that decides that, not the ray being walked. The two cross at an angle, so backing
  off along the ray can leave a sample a few millimetres from the point as a side camera reads it -- well
  inside what that camera can resolve -- and the point's own surface comes back as the thing hiding it."""
  samples, along = ray_samples(origins, targets, clearance, step, near)
  flat = samples.reshape(-1, 3)

  hit = np.zeros(samples.shape[:-1], dtype=bool)
  for view, (cam_id, cam_data) in enumerate(episode["camera"].items()):
    K, T_cam2world = cam_data["K"], poses[cam_id]["extrinsics"][t]
    u, v, z = core.geometry.project_points(flat, K, T_cam2world)
    reading = slack(cam_data["raw_depth"][t], u, v, z).reshape(samples.shape[:-1])

    height, width = robot_mask[view].shape
    ui = np.clip(np.round(u).astype(int), 0, width - 1)
    vi = np.clip(np.round(v).astype(int), 0, height - 1)
    on_robot = robot_mask[view][vi, ui].reshape(samples.shape[:-1])

    _, _, z_target = core.geometry.project_points(targets, K, T_cam2world)
    in_front = z.reshape(samples.shape[:-1]) < (z_target - tolerance(z_target))[:, None]

    hit |= np.isfinite(reading) & (np.abs(reading) <= 1) & ~on_robot & in_front

  return (hit & along).any(axis=1)
