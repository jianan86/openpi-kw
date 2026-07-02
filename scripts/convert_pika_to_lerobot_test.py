from pathlib import Path
import time

import numpy as np
from PIL import Image
import pytest

from scripts import convert_pika_to_lerobot as converter


def _arm_series(frame_count: int) -> dict[str, np.ndarray]:
    poses7d = np.zeros((frame_count, 7), dtype=np.float32)
    poses7d[:, 0] = np.arange(frame_count, dtype=np.float32)
    return {"poses7d": poses7d}


def test_relative_state_is_previous_pose_in_current_frame() -> None:
    previous = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2], dtype=np.float32)
    current = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2, 0.8], dtype=np.float32)
    arm = {"poses7d": np.stack([previous, current])}

    state = converter.build_relative_state(arm, arm, 1)
    expected_arm = np.asarray(
        [0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.2],
        dtype=np.float32,
    )

    np.testing.assert_allclose(state[:10], expected_arm, atol=1e-6)
    np.testing.assert_allclose(state[10:], expected_arm, atol=1e-6)


def test_prepare_frame_resizes_with_black_padding(tmp_path: Path) -> None:
    image = np.full((4, 8, 3), (10, 20, 30), dtype=np.uint8)
    left_path = tmp_path / "left.png"
    right_path = tmp_path / "right.png"
    Image.fromarray(image).save(left_path)
    Image.fromarray(image).save(right_path)

    files = {"left_fisheye": [left_path] * 3, "right_fisheye": [right_path] * 3}
    frame = converter.prepare_frame(files, _arm_series(3), _arm_series(3), "task", 1, 1, 6)

    for key in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        assert frame[key].shape == (6, 6, 3)
        assert frame[key].dtype == np.uint8
    assert np.all(frame["observation.images.cam_high"] == 0)
    assert np.all(frame["observation.images.cam_left_wrist"][0] == 0)
    assert np.all(frame["observation.images.cam_left_wrist"][1:4] == (10, 20, 30))
    assert np.all(frame["observation.images.cam_left_wrist"][4:] == 0)


def test_serial_and_parallel_frames_are_equal_and_ordered(tmp_path: Path) -> None:
    paths = []
    for frame_idx in range(5):
        path = tmp_path / f"{frame_idx}.png"
        Image.fromarray(np.full((4, 8, 3), frame_idx, dtype=np.uint8)).save(path)
        paths.append(path)
    files = {"left_fisheye": paths, "right_fisheye": paths}
    right = _arm_series(6)
    left = _arm_series(6)

    serial = list(converter.iter_prepared_frames(files, right, left, "task", 4, 1, 6, 1))
    parallel = list(converter.iter_prepared_frames(files, right, left, "task", 4, 1, 6, 3))

    assert len(serial) == len(parallel) == 4
    for frame_idx, (serial_frame, parallel_frame) in enumerate(zip(serial, parallel, strict=True), start=1):
        assert serial_frame.keys() == parallel_frame.keys()
        for key in serial_frame:
            if isinstance(serial_frame[key], np.ndarray):
                assert np.array_equal(serial_frame[key], parallel_frame[key])
            else:
                assert serial_frame[key] == parallel_frame[key]
        assert np.max(parallel_frame["observation.images.cam_left_wrist"]) == frame_idx


def test_parallel_frame_order_and_exception_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    def prepare_in_reverse_completion_order(*args):
        frame_idx = args[4]
        time.sleep((4 - frame_idx) * 0.005)
        if frame_idx == 3:
            raise RuntimeError("bad frame")
        return {"frame_idx": frame_idx}

    monkeypatch.setattr(converter, "prepare_frame", prepare_in_reverse_completion_order)
    iterator = converter.iter_prepared_frames({}, {}, {}, "task", 5, 1, 6, 3)
    assert next(iterator) == {"frame_idx": 1}
    assert next(iterator) == {"frame_idx": 2}
    with pytest.raises(RuntimeError, match="bad frame"):
        next(iterator)
