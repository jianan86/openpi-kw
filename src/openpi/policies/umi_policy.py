"""UMI policy transforms for pi0.5 fine-tuning.

Supports two dataset formats:
  - euler_input=False: pos(3) + rot6d(6) + grip(1) = 10 dims per robot (native 6D)
  - euler_input=True:  pos(3) + euler(3) + grip(1) =  7 dims per robot → auto-convert to 10 dims

FastUMI dataset (FastUMI_100k_lerobot) uses euler_input=True (7 dims per arm).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from openpi.shared import rotation_utils

if TYPE_CHECKING:
    from openpi.models.model import ModelType


def euler_rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert roll, pitch, yaw to rotation matrix as Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    rpy = np.asarray(rpy, dtype=np.float32)
    roll = rpy[..., 0]
    pitch = rpy[..., 1]
    yaw = rpy[..., 2]

    sr = np.sin(roll)
    cr = np.cos(roll)
    sp = np.sin(pitch)
    cp = np.cos(pitch)
    sy = np.sin(yaw)
    cy = np.cos(yaw)

    rot = np.empty(rpy.shape[:-1] + (3, 3), dtype=np.float32)
    rot[..., 0, 0] = cy * cp
    rot[..., 0, 1] = cy * sp * sr - sy * cr
    rot[..., 0, 2] = cy * sp * cr + sy * sr
    rot[..., 1, 0] = sy * cp
    rot[..., 1, 1] = sy * sp * sr + cy * cr
    rot[..., 1, 2] = sy * sp * cr - cy * sr
    rot[..., 2, 0] = -sp
    rot[..., 2, 1] = cp * sr
    rot[..., 2, 2] = cp * cr
    return rot


def matrix_to_rot6d(rot_mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to 6D rotation using the first two columns."""
    rot_mat = np.asarray(rot_mat, dtype=np.float32)
    if rot_mat.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrix shape (..., 3, 3), got {rot_mat.shape}")
    return np.concatenate([rot_mat[..., :, 0], rot_mat[..., :, 1]], axis=-1).astype(np.float32, copy=False)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation to rotation matrix via Gram-Schmidt."""
    return rotation_utils.rot6d_to_mat_np(np.asarray(rot6d, dtype=np.float32)).astype(np.float32, copy=False)


def _matrix_to_euler_rpy(rot_mat: np.ndarray) -> np.ndarray:
    """Convert a single Rz @ Ry @ Rx rotation matrix to roll, pitch, yaw."""
    rot_mat = np.asarray(rot_mat, dtype=np.float32)
    if rot_mat.shape != (3, 3):
        raise ValueError(f"Expected rotation matrix shape (3, 3), got {rot_mat.shape}")

    pitch = np.arcsin(np.clip(-rot_mat[2, 0], -1.0, 1.0))
    cos_pitch = np.cos(pitch)
    if abs(float(cos_pitch)) > 1e-6:
        roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        roll = np.float32(0.0)
        yaw = np.arctan2(-rot_mat[0, 1], rot_mat[1, 1])
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def pose7d_to_transform(pose7d: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, roll, pitch, yaw, gripper] to T_world_ee."""
    pose7d = np.asarray(pose7d, dtype=np.float32)
    if pose7d.shape != (7,):
        raise ValueError(f"Expected pose7d shape (7,), got {pose7d.shape}")
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = euler_rpy_to_matrix(pose7d[3:6])
    transform[:3, 3] = pose7d[:3]
    return transform


def transform_to_pose10d(transform: np.ndarray, gripper: float | np.float32) -> np.ndarray:
    """Convert a 4x4 transform and gripper distance to [xyz, rot6d, gripper]."""
    transform = np.asarray(transform, dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4, 4), got {transform.shape}")
    return np.concatenate(
        [transform[:3, 3], matrix_to_rot6d(transform[:3, :3]), np.asarray([gripper], dtype=np.float32)],
        axis=0,
    ).astype(np.float32, copy=False)


def pose7d_to_pose10d(pose7d: np.ndarray) -> np.ndarray:
    """Convert absolute 7D pose to absolute 10D pose."""
    pose7d = np.asarray(pose7d, dtype=np.float32)
    return transform_to_pose10d(pose7d_to_transform(pose7d), pose7d[6])


def relative_pose7d(target_pose7d: np.ndarray, base_pose7d: np.ndarray) -> np.ndarray:
    """Compute target pose relative to base pose, preserving target absolute gripper distance."""
    target_pose7d = np.asarray(target_pose7d, dtype=np.float32)
    base_pose7d = np.asarray(base_pose7d, dtype=np.float32)
    relative_transform = np.linalg.inv(pose7d_to_transform(base_pose7d)) @ pose7d_to_transform(target_pose7d)
    return np.concatenate(
        [
            relative_transform[:3, 3],
            _matrix_to_euler_rpy(relative_transform[:3, :3]),
            np.asarray([target_pose7d[6]], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def relative_pose7d_to_pose10d(target_pose7d: np.ndarray, base_pose7d: np.ndarray) -> np.ndarray:
    """Compute relative 10D pose from target/base absolute 7D poses."""
    relative_transform = np.linalg.inv(pose7d_to_transform(base_pose7d)) @ pose7d_to_transform(target_pose7d)
    return transform_to_pose10d(relative_transform, np.asarray(target_pose7d, dtype=np.float32)[6])


def euler_to_rot6d(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert roll, pitch, yaw angles in radians to 6D rotation.

    Args:
        euler_xyz: (..., 3) array of roll, pitch, yaw angles

    Returns:
        (..., 6) array of 6D rotation representation
    """
    return matrix_to_rot6d(euler_rpy_to_matrix(euler_xyz))


class UMIInputs:
    """Map UMI LeRobot dataset keys to openpi canonical keys (DataTransformFn protocol).

    Expected LeRobot dataset format:
      - observation.state:  (7,) or (14,) — pos(3)+euler(3)+grip(1) per robot
      - action:             (7,) or (14,)
      - observation.images.cam_high:      (H, W, 3) — main camera
      - observation.images.cam_left_wrist: (H, W, 3) — left wrist
      - observation.images.cam_right_wrist: (H, W, 3) — right wrist
      - task: language instruction string

    If euler_input=True, converts Euler angles → 6D rotation on the fly.
    """

    def __init__(
        self,
        *,
        model_type: ModelType | None = None,
        action_horizon: int | None = None,
        euler_input: bool = False,
        num_robots: int = 1,
    ):
        self._model_type = model_type
        self._action_horizon = action_horizon
        self._euler_input = euler_input
        self._num_robots = num_robots

    def _euler_to_6d(self, arr: np.ndarray) -> np.ndarray:
        """Convert 7-dim per robot (Euler) to 10-dim per robot (6D)."""
        per_robot_in = 7   # pos(3) + euler(3) + grip(1)
        per_robot_out = 10  # pos(3) + rot6d(6) + grip(1)
        total_out = self._num_robots * per_robot_out

        if arr.shape[-1] == total_out:
            return arr  # already in 6D format

        out = np.zeros(arr.shape[:-1] + (total_out,), dtype=arr.dtype)
        for r in range(self._num_robots):
            i0 = r * per_robot_in
            o0 = r * per_robot_out
            # Position: copy as-is
            out[..., o0:o0 + 3] = arr[..., i0:i0 + 3]
            # Euler → 6D
            euler = arr[..., i0 + 3:i0 + 6]
            out[..., o0 + 3:o0 + 9] = euler_to_rot6d(euler)
            # Gripper: copy as-is
            out[..., o0 + 9] = arr[..., i0 + 6]
        return out

    def __call__(self, data: dict) -> dict:
        # Map observation state
        data["state"] = data["observation.state"]
        if "action" in data:
            data["actions"] = data["action"]

        # Euler → 6D conversion
        if self._euler_input:
            data["state"] = self._euler_to_6d(data["state"])
            if "actions" in data:
                data["actions"] = self._euler_to_6d(data["actions"])

        # Map images — read from top-level keys (post-repack) or nested observation.images
        images_out = {}
        obs_images = data.get("observation.images", {})
        # First try top-level (repack flattens observation.images.cam_* to cam_*)
        for src_key, dst_key in [
            ("cam_high", "base_0_rgb"),
            ("cam_left_wrist", "left_wrist_0_rgb"),
            ("cam_right_wrist", "right_wrist_0_rgb"),
        ]:
            if src_key in data:
                images_out[dst_key] = data.pop(src_key)
            elif src_key in obs_images:
                images_out[dst_key] = obs_images[src_key]

        if images_out:
            data["image"] = images_out
            data["image_mask"] = {k: True for k in images_out}

        # UMI actions are delta by nature — no additional DeltaActions needed
        return data


class UMIOutputs:
    """Extract UMI action from model output (DataTransformFn protocol)."""

    def __init__(self, action_dim: int):
        self._action_dim = action_dim

    def __call__(self, data: dict) -> dict:
        data["actions"] = data["actions"][..., :self._action_dim]
        return data
