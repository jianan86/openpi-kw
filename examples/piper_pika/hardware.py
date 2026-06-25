from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time

import numpy as np
from openpi_client import piper_pika

_PIPER_FATAL_STATUS_TERMS = (
    "NO_SOLUTION",
    "SINGULARITY_POINT",
    "TARGET_POS_EXCEEDS_LIMIT",
    "EMERGENCY_STOP",
    "JOINT_COMMUNICATION_ERR",
    "JOINT_BRAKE_NOT_RELEASED",
    "COLLISION_OCCURRED",
    "JOINT_STATUS_ERR",
    "OTHER_ERR",
    "MAIN_CONTROLLER_NTC_OVER_TEMPERATURE",
    "RELEASE_RESISTOR_NTC_OVER_TEMPERATURE",
)


@dataclass
class SensorSnapshot:
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    ee_pose: np.ndarray
    tcp_pose: np.ndarray
    timestamp: float


class LatestValue:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = None

    def set(self, value) -> None:
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value


class PiperRobot:
    def __init__(self, side: str, can_name: str, *, dry_run: bool, disabled: bool) -> None:
        self.side = side
        self.can_name = can_name
        self.dry_run = dry_run
        self.robot = None
        if disabled:
            return
        from piper_sdk import C_PiperInterface  # noqa: PLC0415

        self.robot = C_PiperInterface(can_name=can_name)
        self.robot.ConnectPort()
        while not self.robot.EnablePiper():
            time.sleep(0.01)
        self.robot.MotionCtrl_2(0x01, 0x00, 100, 0x00)

    def read_ee_pose(self) -> np.ndarray:
        if self.robot is None:
            return np.zeros(6, dtype=np.float32)
        pose = self.robot.GetArmEndPoseMsgs().end_pose
        xyz = np.asarray([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float32) / 1_000_000.0
        rpy = np.deg2rad(np.asarray([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float32) / 1000.0)
        return np.concatenate([xyz, rpy]).astype(np.float32, copy=False)

    def close(self) -> None:
        if self.robot is None:
            return
        self.robot.DisconnectPort()
        self.robot = None

    def execute_pose(self, target: np.ndarray) -> None:
        if self.robot is None or self.dry_run:
            return
        x, y, z, roll, pitch, yaw = np.asarray(target, dtype=np.float32)[:6]
        self.robot.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.robot.EndPoseCtrl(
            round(float(x) * 1_000_000.0),
            round(float(y) * 1_000_000.0),
            round(float(z) * 1_000_000.0),
            round(float(np.degrees(roll)) * 1000.0),
            round(float(np.degrees(pitch)) * 1000.0),
            round(float(np.degrees(yaw)) * 1000.0),
        )
        self.raise_for_status()

    def execute_joints(self, joints: np.ndarray) -> None:
        if self.robot is None or self.dry_run:
            return
        values = tuple(round(float(np.degrees(value)) * 1000.0) for value in joints)
        self.robot.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.robot.JointCtrl(*values)
        self.raise_for_status()

    def raise_for_status(self) -> None:
        if self.robot is None:
            return
        snapshots = []
        for name in ("GetArmStatus", "GetArmStatusMsgs"):
            method = getattr(self.robot, name, None)
            if method is not None:
                snapshots.append(str(method()))
        text = json.dumps(snapshots)
        errors = [term for term in _PIPER_FATAL_STATUS_TERMS if term in text]
        if "TARGET_POS_EXCEEDS_LIMIT" in errors:
            print(
                f"[piper-error] Piper {self.side} reported TARGET_POS_EXCEEDS_LIMIT; continuing",
                flush=True,
            )
            errors = [error for error in errors if error != "TARGET_POS_EXCEEDS_LIMIT"]
        if errors:
            raise RuntimeError(f"Piper {self.side} reported fatal status: {errors}")


class PikaGripper:
    def __init__(
        self,
        side: str,
        port: str,
        fisheye_device,
        width: int,
        height: int,
        fps: int,
        *,
        dry_run: bool,
        disabled: bool,
    ) -> None:
        self.side = side
        self.width = width
        self.height = height
        self.dry_run = dry_run
        self.device = None
        self.fisheye = None
        if disabled:
            return
        from pika.gripper import Gripper  # noqa: PLC0415

        self.device = Gripper(port)
        if not self.device.connect():
            raise RuntimeError(f"failed to connect Pika {side} gripper: {port}")
        self.device.enable()
        self.device.set_camera_param(width, height, fps)
        self.device.set_fisheye_camera_index(fisheye_device)
        self.fisheye = self.device.get_fisheye_camera()
        self._wait_for_frame()

    def _wait_for_frame(self) -> None:
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if self.fisheye is not None and getattr(self.fisheye, "is_connected", False):
                ok, frame = self.fisheye.get_frame()
                if ok and frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
                    return
            time.sleep(0.02)
        raise RuntimeError(f"failed to read initial Pika {self.side} fisheye frame")

    def close(self) -> None:
        if self.device is not None:
            self.device.disconnect()

    def read_width(self) -> float:
        if self.device is None:
            return 0.0
        return max(float(self.device.get_gripper_distance()) / 1000.0, 0.0)

    def read_rgb(self) -> np.ndarray:
        if self.fisheye is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        ok, frame = self.fisheye.get_frame()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read Pika {self.side} fisheye frame")
        return np.ascontiguousarray(frame[..., ::-1])

    def execute_width(self, width_m: float) -> None:
        if self.device is None or self.dry_run:
            return
        self.device.set_gripper_distance(float(np.clip(width_m * 1000.0, 0.0, 90.0)))


class PiperPikaHardware:
    def __init__(self, config: dict, arm_mode: str, *, dry_run: bool, no_piper: bool, no_pika: bool) -> None:
        self.arm_mode = arm_mode
        camera = config.get("camera", {})
        self.image_size = int(camera.get("image_size", 224))
        width = int(camera.get("width", 640))
        height = int(camera.get("height", 480))
        fps = int(camera.get("fps", 30))

        self.right_arm = PiperRobot("right", config["right"]["piper_can"], dry_run=dry_run, disabled=no_piper)
        self.right_gripper = PikaGripper(
            "right",
            config["right"]["gripper_port"],
            config["right"]["fisheye_device"],
            width,
            height,
            fps,
            dry_run=dry_run,
            disabled=no_pika,
        )
        self.left_arm: PiperRobot | None = None
        self.left_gripper: PikaGripper | None = None
        if arm_mode == "dual":
            self.left_arm = PiperRobot("left", config["left"]["piper_can"], dry_run=dry_run, disabled=no_piper)
            self.left_gripper = PikaGripper(
                "left",
                config["left"]["gripper_port"],
                config["left"]["fisheye_device"],
                width,
                height,
                fps,
                dry_run=dry_run,
                disabled=no_pika,
            )

    def close(self) -> None:
        self.right_arm.close()
        self.right_gripper.close()
        if self.left_arm is not None:
            self.left_arm.close()
        if self.left_gripper is not None:
            self.left_gripper.close()

    def move_home(self, home: dict) -> None:
        self.right_arm.execute_joints(np.asarray(home["right_arm"], dtype=np.float32))
        if self.arm_mode == "dual":
            assert self.left_arm is not None
            self.left_arm.execute_joints(np.asarray(home["left_arm"], dtype=np.float32))

    def read_poses(self) -> tuple[np.ndarray, np.ndarray]:
        right_ee = np.concatenate(
            [self.right_arm.read_ee_pose(), np.asarray([self.right_gripper.read_width()], dtype=np.float32)]
        )
        right_tcp = piper_pika.ee_pose7_to_tcp_pose7(right_ee)
        if self.arm_mode == "single":
            ee_pose = np.concatenate([right_ee, right_ee]).astype(np.float32, copy=False)
            tcp_pose = np.concatenate([right_tcp, right_tcp]).astype(np.float32, copy=False)
            return ee_pose, tcp_pose
        assert self.left_arm is not None
        assert self.left_gripper is not None
        left_ee = np.concatenate(
            [self.left_arm.read_ee_pose(), np.asarray([self.left_gripper.read_width()], dtype=np.float32)]
        )
        left_tcp = piper_pika.ee_pose7_to_tcp_pose7(left_ee)
        ee_pose = np.concatenate([right_ee, left_ee]).astype(np.float32, copy=False)
        tcp_pose = np.concatenate([right_tcp, left_tcp]).astype(np.float32, copy=False)
        return ee_pose, tcp_pose

    def read_snapshot(self) -> SensorSnapshot:
        right = piper_pika.preprocess_fisheye(self.right_gripper.read_rgb(), self.image_size)
        if self.arm_mode == "single":
            left = right.copy()
        else:
            assert self.left_gripper is not None
            left = piper_pika.preprocess_fisheye(self.left_gripper.read_rgb(), self.image_size)
        ee_pose, tcp_pose = self.read_poses()
        return SensorSnapshot(
            left_rgb=left,
            right_rgb=right,
            ee_pose=ee_pose,
            tcp_pose=tcp_pose,
            timestamp=time.time(),
        )

    def execute(self, tcp_target: np.ndarray) -> None:
        right = tcp_target[piper_pika.RIGHT_ARM_SLICE]
        right_ee = piper_pika.tcp_pose7_to_ee_pose7(right)
        print(f"[control] right tcp={right.round(6).tolist()} ee={right_ee.round(6).tolist()}", flush=True)
        self.right_arm.execute_pose(right_ee)
        self.right_gripper.execute_width(float(right[6]))
        if self.arm_mode == "single":
            return
        assert self.left_arm is not None
        assert self.left_gripper is not None
        left = tcp_target[piper_pika.LEFT_ARM_SLICE]
        left_ee = piper_pika.tcp_pose7_to_ee_pose7(left)
        print(f"[control] left tcp={left.round(6).tolist()} ee={left_ee.round(6).tolist()}", flush=True)
        self.left_arm.execute_pose(left_ee)
        self.left_gripper.execute_width(float(left[6]))


class SensorWorker:
    def __init__(self, hardware: PiperPikaHardware) -> None:
        self.hardware = hardware
        self.latest = LatestValue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="piper-pika-sensors", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.latest.set(self.hardware.read_snapshot())
            except Exception as error:
                print(f"[sensor] read failed: {error}")
            time.sleep(0.001)
