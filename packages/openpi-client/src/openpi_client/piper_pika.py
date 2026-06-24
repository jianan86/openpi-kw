"""Utilities for running OpenPI UMI policies on Piper arms with Pika grippers."""

from __future__ import annotations

import socket
import subprocess
import time
from typing import Dict, Optional, Tuple

import numpy as np
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client import image_tools


RIGHT_ARM_SLICE = slice(0, 7)
LEFT_ARM_SLICE = slice(7, 14)
SINGLE_ARM_POSE_DIM = 7
SINGLE_ARM_MODEL_DIM = 10

R_EE_TCP = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
T_EE_TCP = np.eye(4, dtype=np.float32)
T_EE_TCP[:3, :3] = R_EE_TCP
T_EE_TCP[:3, 3] = np.array([0.0, 0.0, 0.1943], dtype=np.float32)
T_TCP_EE = np.linalg.inv(T_EE_TCP).astype(np.float32)


def euler_xyz_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in np.asarray(rpy)]
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float32,
    )


def matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    sy = float(np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    if sy >= 1e-6:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        pitch = np.arctan2(-matrix[2, 0], sy)
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = np.arctan2(-matrix[1, 2], matrix[1, 1])
        pitch = np.arctan2(-matrix[2, 0], sy)
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1).astype(np.float32, copy=False)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    a1, a2 = rot6d[..., :3], rot6d[..., 3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32, copy=False)


def pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (SINGLE_ARM_POSE_DIM,):
        raise ValueError(f"expected pose shape (7,), got {pose.shape}")
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = euler_xyz_to_matrix(pose[3:6])
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_to_pose7(matrix: np.ndarray, gripper: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return np.concatenate(
        [matrix[:3, 3], matrix_to_euler_xyz(matrix[:3, :3]), np.asarray([gripper], dtype=np.float32)]
    ).astype(np.float32, copy=False)


def pose7_to_pose10(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    matrix = pose7_to_matrix(pose)
    return np.concatenate([pose[:3], matrix_to_rot6d(matrix[:3, :3]), pose[6:7]]).astype(
        np.float32, copy=False
    )


def pose10_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (SINGLE_ARM_MODEL_DIM,):
        raise ValueError(f"expected pose shape (10,), got {pose.shape}")
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rot6d_to_matrix(pose[3:9])
    matrix[:3, 3] = pose[:3]
    return matrix


def ee_pose7_to_tcp_pose7(ee_pose: np.ndarray) -> np.ndarray:
    ee_pose = np.asarray(ee_pose, dtype=np.float32)
    return matrix_to_pose7(pose7_to_matrix(ee_pose) @ T_EE_TCP, float(ee_pose[6]))


def tcp_pose7_to_ee_pose7(tcp_pose: np.ndarray) -> np.ndarray:
    tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
    return matrix_to_pose7(pose7_to_matrix(tcp_pose) @ T_TCP_EE, float(tcp_pose[6]))


def relative_actions_to_absolute_tcp(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Convert a (T, 20) relative UMI chunk into absolute (T, 14) TCP targets."""
    actions = np.asarray(actions, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 20:
        raise ValueError(f"expected action shape (T, 20), got {actions.shape}")
    if state.shape != (20,):
        raise ValueError(f"expected state shape (20,), got {state.shape}")

    arm_chunks = []
    identity_rot6d = matrix_to_rot6d(np.eye(3, dtype=np.float32))
    for arm_index in range(2):
        start = arm_index * SINGLE_ARM_MODEL_DIM
        base = pose10_to_matrix(state[start : start + SINGLE_ARM_MODEL_DIM])
        relative = actions[:, start : start + SINGLE_ARM_MODEL_DIM]
        arm_targets = []
        for row in relative:
            rotation = row[3:9]
            if float(np.linalg.norm(rotation)) < 1e-8:
                rotation = identity_rot6d
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = rot6d_to_matrix(rotation)
            transform[:3, 3] = row[:3]
            arm_targets.append(matrix_to_pose7(base @ transform, float(row[9])))
        arm_chunks.append(np.stack(arm_targets))
    return np.concatenate(arm_chunks, axis=1).astype(np.float32, copy=False)


def preprocess_fisheye(image: np.ndarray, size: int = 224) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got {image.shape}")
    resized = image_tools.resize_with_pad(image, size, size)
    return np.ascontiguousarray(image_tools.convert_to_uint8(resized))


def limit_pose_step(
    current: np.ndarray,
    target: np.ndarray,
    max_pos_step: float,
    max_rot_step: float,
    max_gripper_step: float,
    gripper_range: Tuple[float, float],
) -> np.ndarray:
    current = np.asarray(current, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32).copy()
    if current.shape != (7,) or target.shape != (7,):
        raise ValueError("current and target must both have shape (7,)")
    target[:3] = current[:3] + np.clip(target[:3] - current[:3], -max_pos_step, max_pos_step)
    rotation_delta = (target[3:6] - current[3:6] + np.pi) % (2 * np.pi) - np.pi
    target[3:6] = current[3:6] + np.clip(rotation_delta, -max_rot_step, max_rot_step)
    target[6] = current[6] + float(np.clip(target[6] - current[6], -max_gripper_step, max_gripper_step))
    target[6] = float(np.clip(target[6], gripper_range[0], gripper_range[1]))
    return target


class RelativeActionPolicy(_base_policy.BasePolicy):
    """Postprocess a remote UMI policy chunk using the state sent with that request."""

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: Optional[int] = 10) -> None:
        self._policy = policy
        self._action_horizon = action_horizon

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        state = np.asarray(obs["observation.state"], dtype=np.float32).copy()
        result = self._policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if self._action_horizon is not None and actions.shape[0] != self._action_horizon:
            raise ValueError(f"expected action horizon {self._action_horizon}, got {actions.shape}")
        result["actions"] = relative_actions_to_absolute_tcp(actions, state)
        return result

    @override
    def reset(self) -> None:
        self._policy.reset()


class SshTunnel:
    def __init__(
        self,
        server: str,
        ssh_port: int,
        remote_host: str,
        remote_port: int,
        local_port: int = 0,
        connect_timeout: float = 10.0,
    ) -> None:
        self.server = server
        self.ssh_port = ssh_port
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.local_port = local_port or find_free_port()
        self.connect_timeout = connect_timeout
        self._process: Optional[subprocess.Popen] = None

    def start(self) -> int:
        self._process = subprocess.Popen(
            [
                "ssh",
                "-N",
                "-L",
                f"{self.local_port}:{self.remote_host}:{self.remote_port}",
                self.server,
                "-p",
                str(self.ssh_port),
            ]
        )
        try:
            _wait_for_local_port(self.local_port, self.connect_timeout)
        except Exception:
            self.close()
            raise
        if self._process.poll() is not None:
            raise RuntimeError(f"ssh tunnel exited with code {self._process.returncode}")
        return self.local_port

    def close(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None

    def __enter__(self) -> int:
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_local_port(port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            try:
                sock.connect(("127.0.0.1", port))
                return
            except OSError as error:
                last_error = error
        time.sleep(0.05)
    raise RuntimeError(f"ssh tunnel local port 127.0.0.1:{port} is not ready: {last_error}")


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(norm, 1e-12)
