import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

_HARDWARE_PATH = Path(__file__).with_name("hardware.py")
_SPEC = importlib.util.spec_from_file_location("piper_pika_hardware", _HARDWARE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
hardware = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = hardware
_SPEC.loader.exec_module(hardware)


BASE_SUMMARY = {
    "groups": [
        {
            "device_name": "right_gripper",
            "physical_device_id": "260622271788",
            "status": "ok",
            "realsense": {"serial": "260622271788"},
            "usb_port": {"devnode": "/dev/ttyUSB2"},
            "fisheye": {"devnode": "/dev/video25"},
        },
        {
            "device_name": "left_gripper",
            "physical_device_id": "412622273326",
            "status": "ok",
            "realsense": {"serial": "412622273326"},
            "usb_port": {"devnode": "/dev/ttyUSB3"},
            "fisheye": {"devnode": "/dev/video33"},
        },
    ]
}


def _write_summary(tmp_path, payload):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_pika_device_bindings_reads_gripper_usb_and_fisheye(tmp_path):
    bindings = hardware.load_pika_device_bindings(_write_summary(tmp_path, BASE_SUMMARY))

    assert bindings["right"].group_name == "right_gripper"
    assert bindings["right"].physical_device_id == "260622271788"
    assert bindings["right"].gripper_port == "/dev/ttyUSB2"
    assert bindings["right"].fisheye_device == "/dev/video25"
    assert bindings["left"].group_name == "left_gripper"
    assert bindings["left"].gripper_port == "/dev/ttyUSB3"
    assert bindings["left"].fisheye_device == "/dev/video33"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload["groups"].pop(), "left_gripper"),
        (lambda payload: payload["groups"][0].__setitem__("status", "missing"), "status is not ok"),
        (lambda payload: payload["groups"][0]["usb_port"].pop("devnode"), "usb_port.devnode"),
        (lambda payload: payload["groups"][0]["fisheye"].pop("devnode"), "fisheye.devnode"),
    ],
)
def test_load_pika_device_bindings_rejects_invalid_summary(tmp_path, mutate, message):
    payload = copy.deepcopy(BASE_SUMMARY)
    mutate(payload)

    with pytest.raises((RuntimeError, ValueError), match=message):
        hardware.load_pika_device_bindings(_write_summary(tmp_path, payload))


def test_load_pika_device_bindings_requires_summary_file(tmp_path):
    with pytest.raises(RuntimeError, match="device group summary not found"):
        hardware.load_pika_device_bindings(tmp_path / "missing.json")


def test_piper_pika_hardware_no_pika_skips_device_group_summary(monkeypatch):
    def fail_if_called():
        raise AssertionError("summary should not be read when no_pika=True")

    monkeypatch.setattr(hardware, "load_pika_device_bindings", fail_if_called)
    config = {
        "camera": {},
        "right": {"piper_can": "can0", "gripper_port": "/dev/old0", "fisheye_device": "/dev/video-old0"},
        "left": {"piper_can": "can1", "gripper_port": "/dev/old1", "fisheye_device": "/dev/video-old1"},
    }

    robot = hardware.PiperPikaHardware(config, "dual", dry_run=True, no_piper=True, no_pika=True)

    assert robot.right_gripper.device is None
    assert robot.left_gripper is not None
    assert robot.left_gripper.device is None
