"""PyBullet physics renderer for Franka + Robotiq gripper.

Test:
  python -c "from droid.physics import PyBulletRenderer_Robotiq; r = PyBulletRenderer_Robotiq(); print('✅ physics OK')"
"""

import importlib.util
import os

import numpy as np
import pybullet as p
import pybullet_data

# Default URDF path (relative to droid_workspace)
_DEFAULT_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "PointWorld/assets/franka_description/franka_panda_robotiq_2f85_og.urdf"
)


class PyBulletRenderer_Robotiq:
    """Dual-body physics renderer: thin Franka arm + Robotiq 2F-85 gripper.

    The 'robot' body renders the arm links (hand/finger hidden).
    The 'ghost' body renders the gripper (arm links hidden).
    Together they form the complete visual model.
    """

    def __init__(self, ghost_urdf=None):
        if ghost_urdf is None:
            ghost_urdf = _DEFAULT_URDF

        if p.isConnected():
            p.disconnect()
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Try to load EGL plugin for headless rendering
        egl_spec = importlib.util.find_spec('eglRendererPlugin')
        if egl_spec:
            p.loadPlugin(egl_spec.origin, "_eglRendererPlugin")

        # Real body: thin arm (hand/finger hidden)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.arm_joints = [
            i for i in range(p.getNumJoints(self.robot_id))
            if "panda_joint" in p.getJointInfo(self.robot_id, i)[1].decode('utf-8')
            and p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED
        ]

        self.hidden_robot_links = []
        for i in range(-1, p.getNumJoints(self.robot_id)):
            name = (p.getBodyInfo(self.robot_id)[0].decode('utf-8') if i == -1
                    else p.getJointInfo(self.robot_id, i)[12].decode('utf-8'))
            if "hand" in name or "finger" in name:
                p.changeVisualShape(self.robot_id, i, rgbaColor=[0, 0, 0, 0])
                self.hidden_robot_links.append(i)

        # Ghost body: Robotiq gripper (arm links hidden)
        self.ghost_id = p.loadURDF(ghost_urdf, useFixedBase=True)
        self.ghost_arm_joints = [
            i for i in range(p.getNumJoints(self.ghost_id))
            if "panda_joint" in p.getJointInfo(self.ghost_id, i)[1].decode('utf-8')
            and p.getJointInfo(self.ghost_id, i)[2] != p.JOINT_FIXED
        ]

        self.gripper_joints = []
        self.gripper_signs = []
        for i in range(p.getNumJoints(self.ghost_id)):
            info = p.getJointInfo(self.ghost_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            if joint_type != p.JOINT_FIXED and "panda_joint" not in joint_name:
                self.gripper_joints.append(i)
                base_sign = -1 if "right" in joint_name else 1
                if "inner_finger" in joint_name or "follower" in joint_name or "finger_tip" in joint_name:
                    self.gripper_signs.append(base_sign * -1)
                else:
                    self.gripper_signs.append(base_sign)

        self.hidden_ghost_links = []
        for i in range(-1, p.getNumJoints(self.ghost_id)):
            name = (p.getBodyInfo(self.ghost_id)[0].decode('utf-8') if i == -1
                    else p.getJointInfo(self.ghost_id, i)[12].decode('utf-8'))
            if "panda_link" in name:
                p.changeVisualShape(self.ghost_id, i, rgbaColor=[0, 0, 0, 0])
                self.hidden_ghost_links.append(i)

    def _get_projection_matrix(self, intrinsics, width, height):
        """Convert camera intrinsics to OpenGL projection matrix."""
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        near, far = 0.01, 10.0
        return [
            2.0 * fx / width, 0.0, 0.0, 0.0,
            0.0, 2.0 * fy / height, 0.0, 0.0,
            1.0 - 2.0 * cx / width, 2.0 * cy / height - 1.0,
            (far + near) / (near - far), -1.0,
            0.0, 0.0, 2.0 * far * near / (near - far), 0.0,
        ]

    def update_robot_pose(self, joint_angles, gripper_state=None, gripper_width_offset=0.08):
        """Synchronize both bodies to the given joint configuration."""
        for i, angle in zip(self.arm_joints, joint_angles):
            p.resetJointState(self.robot_id, i, angle)
        for i, angle in zip(self.ghost_arm_joints, joint_angles):
            p.resetJointState(self.ghost_id, i, angle)

        if gripper_state is not None and len(self.gripper_joints) > 0:
            raw_val = gripper_state[0] if isinstance(gripper_state, (list, np.ndarray)) else gripper_state
            raw_val = np.clip(raw_val, 0.0, 1.0)
            max_urdf_radian = 0.8028
            angle = (raw_val * max_urdf_radian) - gripper_width_offset
            for i, sign in zip(self.gripper_joints, self.gripper_signs):
                p.resetJointState(self.ghost_id, i, angle * sign)

        p.performCollisionDetection()

    def render_depth(self, extrinsics, intrinsics, width, height):
        """Render physical depth map from camera pose."""
        cam_pos = extrinsics[:3, 3]
        target_pos = cam_pos + extrinsics[:3, 2]
        view_matrix = p.computeViewMatrix(cam_pos, target_pos, -extrinsics[:3, 1])
        proj_matrix = self._get_projection_matrix(intrinsics, width, height)
        _, _, _, depth_buffer, _ = p.getCameraImage(
            width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        metric_depth = 0.1 / (10.0 - 9.99 * np.reshape(depth_buffer, (height, width)))
        return np.where(metric_depth < 9.9, metric_depth, 0.0)

    def render_mask(self, extrinsics, intrinsics, width, height):
        """Render binary robot segmentation mask from camera pose."""
        cam_pos = extrinsics[:3, 3]
        target_pos = cam_pos + extrinsics[:3, 2]
        view_matrix = p.computeViewMatrix(cam_pos, target_pos, -extrinsics[:3, 1])
        proj_matrix = self._get_projection_matrix(intrinsics, width, height)
        _, _, _, _, seg_buffer = p.getCameraImage(
            width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        )
        seg_array = np.reshape(seg_buffer, (height, width)).astype(np.int32)
        obj_ids = seg_array & 0xFFFFFF
        link_ids = (seg_array >> 24) - 1
        valid_robot = (obj_ids == self.robot_id) & ~np.isin(link_ids, self.hidden_robot_links)
        valid_ghost = (obj_ids == self.ghost_id) & ~np.isin(link_ids, self.hidden_ghost_links)
        return valid_robot | valid_ghost

    def render_segmentation(self, extrinsics, intrinsics, width, height):
        """Render full segmentation buffer (obj_ids, link_ids, depth)."""
        cam_pos = extrinsics[:3, 3]
        target_pos = cam_pos + extrinsics[:3, 2]
        view_matrix = p.computeViewMatrix(cam_pos, target_pos, -extrinsics[:3, 1])
        proj_matrix = self._get_projection_matrix(intrinsics, width, height)
        _, _, _, depth_buffer, seg_buffer = p.getCameraImage(
            width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        )
        metric_depth = 0.1 / (10.0 - 9.99 * np.reshape(depth_buffer, (height, width)))
        metric_depth = np.where(metric_depth < 9.9, metric_depth, 0.0)
        seg_array = np.reshape(seg_buffer, (height, width)).astype(np.int32)
        obj_ids = seg_array & 0xFFFFFF
        link_ids = (seg_array >> 24) - 1
        return obj_ids, link_ids, metric_depth
