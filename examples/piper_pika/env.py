from __future__ import annotations

import time

import numpy as np
from openpi_client import piper_pika
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.piper_pika import hardware as _hardware


class PiperPikaEnvironment(_environment.Environment):
    def __init__(
        self,
        hardware: _hardware.PiperPikaHardware,
        prompt: str,
        home: dict | None,
        max_pos_step: float,
        max_rot_step: float,
        max_gripper_step: float,
        min_gripper: float,
        max_gripper: float,
        *,
        dry_run: bool,
    ) -> None:
        self._hardware = hardware
        self._prompt = prompt
        self._home = home
        self._max_pos_step = max_pos_step
        self._max_rot_step = max_rot_step
        self._max_gripper_step = max_gripper_step
        self._gripper_range = (min_gripper, max_gripper)
        self._dry_run = dry_run
        self._sensors = _hardware.SensorWorker(hardware)
        self._sensors.start()
        self._last_target = None

    @override
    def reset(self) -> None:
        if self._home is not None:
            self._hardware.move_home(self._home)
        snapshot = self._wait_for_snapshot()
        self._last_target = snapshot.tcp_pose.copy()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        snapshot = self._wait_for_snapshot()
        right_state = piper_pika.pose7_to_pose10(snapshot.tcp_pose[piper_pika.RIGHT_ARM_SLICE])
        left_state = piper_pika.pose7_to_pose10(snapshot.tcp_pose[piper_pika.LEFT_ARM_SLICE])
        state = np.concatenate([right_state, left_state]).astype(np.float32, copy=False)
        black = np.zeros_like(snapshot.left_rgb)
        return {
            "observation.state": state,
            "cam_high": black,
            "cam_left_wrist": snapshot.left_rgb,
            "cam_right_wrist": snapshot.right_rgb,
            "prompt": self._prompt,
            "_debug.ee_pose": snapshot.ee_pose.copy(),
            "_debug.tcp_pose": snapshot.tcp_pose.copy(),
        }

    @override
    def apply_action(self, action: dict) -> None:
        target = np.asarray(action["actions"], dtype=np.float32)
        if target.shape != (14,):
            raise ValueError(f"expected absolute TCP action shape (14,), got {target.shape}")
        if self._last_target is None:
            self._last_target = self._wait_for_snapshot().tcp_pose.copy()
        limited = self._last_target.copy()
        for arm_slice in (piper_pika.RIGHT_ARM_SLICE, piper_pika.LEFT_ARM_SLICE):
            limited[arm_slice] = piper_pika.limit_pose_step(
                self._last_target[arm_slice],
                target[arm_slice],
                self._max_pos_step,
                self._max_rot_step,
                self._max_gripper_step,
                self._gripper_range,
            )
        if self._dry_run:
            ee = np.concatenate([piper_pika.tcp_pose7_to_ee_pose7(limited[piper_pika.RIGHT_ARM_SLICE]), piper_pika.tcp_pose7_to_ee_pose7(limited[piper_pika.LEFT_ARM_SLICE])])
            print(f"[dry-run control] tcp={limited.round(6).tolist()} ee={ee.round(6).tolist()}", flush=True)
        else:
            self._hardware.execute(limited)
        self._last_target = limited

    def close(self) -> None:
        self._sensors.stop()
        self._hardware.close()

    def _wait_for_snapshot(self) -> _hardware.SensorSnapshot:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            snapshot = self._sensors.latest.get()
            if snapshot is not None:
                return snapshot
            time.sleep(0.01)
        raise RuntimeError("timed out waiting for Piper/Pika sensor data")
