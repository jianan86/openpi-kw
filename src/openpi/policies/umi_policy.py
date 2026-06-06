"""UMI policy transforms for pi0.5 fine-tuning.

Supports two dataset formats:
  - euler_input=False: pos(3) + rot6d(6) + grip(1) = 10 dims per robot (native 6D)
  - euler_input=True:  pos(3) + euler(3) + grip(1) =  7 dims per robot → auto-convert to 10 dims

FastUMI dataset (FastUMI_100k_lerobot) uses euler_input=True (7 dims per arm).
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from openpi.models.model import ModelType
from openpi.shared import rotation_utils


def euler_to_rot6d(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert Euler angles (x,y,z in radians) to 6D rotation.

    Args:
        euler_xyz: (..., 3) array of Euler angles in XYZ convention

    Returns:
        (..., 6) array of 6D rotation representation
    """
    mat = R.from_euler("xyz", euler_xyz).as_matrix()  # (..., 3, 3)
    # First two columns flattened
    rot6d = np.concatenate([mat[..., 0], mat[..., 1]], axis=-1)
    return rot6d


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
        data["actions"] = data["action"]

        # Euler → 6D conversion
        if self._euler_input:
            data["state"] = self._euler_to_6d(data["state"])
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
