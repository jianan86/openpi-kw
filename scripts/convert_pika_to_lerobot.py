#!/usr/bin/env python3
"""Convert Pika dual-arm episodes to an OpenPI-compatible LeRobot v3.0 dataset.

Input layout is the Pika/UMI directory format used by 0616_dex:
  episode*/localization/pose/pika_{r,l}/sync.txt
  episode*/gripper/encoder/pika_{r,l}/sync.txt
  episode*/camera/color/pikaFisheyeCamera_{r,l}/sync.txt

Output schema matches the current OpenPI pi05_umi_bimanual config:
  observation.state: (20,)    [right(10), left(10)]
  action:            (10, 20) future relative pose chunk
  three RGB cameras: cam_high, cam_left_wrist, cam_right_wrist
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from openpi.policies.umi_policy import pose7d_to_pose10d, relative_pose7d_to_pose10d
from openpi.shared.image_tools import resize_with_pad

DEFAULT_INPUT_ROOT = Path("/home/jianan/workspace/data/0616_dex")
DEFAULT_OUTPUT_ROOT = Path("/home/jianan/workspace/data/lerobot_pika_0616_dex_v30")
DEFAULT_REPO_ID = "local/pika-0616-dex-v30"
DEFAULT_TASK = "pika dex episode"
DEFAULT_FPS = 30
DEFAULT_HORIZON = 10
DEFAULT_IMAGE_SIZE = 224
DEFAULT_FRAME_WORKERS = min(2, os.cpu_count() or 1)

POSE_DIR = Path("localization/pose/pika")
GRIPPER_DIR = Path("gripper/encoder/pika")
FISHEYE_DIR = Path("camera/color/pikaFisheyeCamera")
ARM_SUFFIXES = {"right": "_r", "left": "_l"}


def load_synced_files(directory: Path) -> list[Path]:
    sync_path = directory / "sync.txt"
    if not sync_path.exists():
        raise FileNotFoundError(f"Required sync file not found: {sync_path}")

    files: list[Path] = []
    with sync_path.open() as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(f"File listed in sync.txt does not exist: {path}")
            try:
                float(path.stem)
            except ValueError as exc:
                raise ValueError(f"Synced file name must start with a numeric timestamp: {path.name}") from exc
            files.append(path)

    if not files:
        raise ValueError(f"No synced files listed in: {sync_path}")
    return files


def load_pose_json(path: Path) -> np.ndarray:
    with path.open() as f:
        payload = json.load(f)
    return np.asarray([payload[k] for k in ("x", "y", "z", "roll", "pitch", "yaw")], dtype=np.float32)


def load_gripper_distance(path: Path) -> np.float32:
    with path.open() as f:
        payload = json.load(f)
    return np.float32(payload["distance"])


def load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def arm_dir(episode_dir: Path, base: Path, side: str) -> Path:
    return episode_dir / base.with_name(f"{base.name}{ARM_SUFFIXES[side]}")


def discover_episode_files(episode_dir: Path) -> dict[str, list[Path]]:
    files = {
        "right_pose": load_synced_files(arm_dir(episode_dir, POSE_DIR, "right")),
        "right_gripper": load_synced_files(arm_dir(episode_dir, GRIPPER_DIR, "right")),
        "left_pose": load_synced_files(arm_dir(episode_dir, POSE_DIR, "left")),
        "left_gripper": load_synced_files(arm_dir(episode_dir, GRIPPER_DIR, "left")),
        "right_fisheye": load_synced_files(arm_dir(episode_dir, FISHEYE_DIR, "right")),
        "left_fisheye": load_synced_files(arm_dir(episode_dir, FISHEYE_DIR, "left")),
    }
    lengths = {name: len(paths) for name, paths in files.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Synced modality counts do not match for {episode_dir}: {lengths}")
    return files


def build_arm_series(pose_files: list[Path], gripper_files: list[Path]) -> dict[str, np.ndarray]:
    poses7d = []
    for pose_path, gripper_path in zip(pose_files, gripper_files, strict=True):
        pose = load_pose_json(pose_path)
        gripper = load_gripper_distance(gripper_path)
        poses7d.append(np.concatenate([pose, np.asarray([gripper], dtype=np.float32)], axis=0))

    poses7d_arr = np.asarray(poses7d, dtype=np.float32)
    states = np.stack([pose7d_to_pose10d(pose7d) for pose7d in poses7d_arr], axis=0).astype(np.float32)
    return {"poses7d": poses7d_arr, "states": states}


def build_relative_action_chunk(
    right: dict[str, np.ndarray],
    left: dict[str, np.ndarray],
    frame_idx: int,
    horizon: int,
) -> np.ndarray:
    per_step = []
    for step in range(1, horizon + 1):
        future_idx = frame_idx + step
        arms = []
        for arm in (right, left):
            arms.append(relative_pose7d_to_pose10d(arm["poses7d"][future_idx], arm["poses7d"][frame_idx]))
        per_step.append(np.concatenate(arms, axis=0))
    return np.asarray(per_step, dtype=np.float32)


def _extract_instruction_strings(payload: Any) -> list[str]:
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, list):
        values: list[str] = []
        for item in payload:
            values.extend(_extract_instruction_strings(item))
        return values
    if isinstance(payload, dict):
        values: list[str] = []
        for key in ("instruction", "instructions", "task", "tasks", "full-instructions", "segment-instructions"):
            if key in payload:
                values.extend(_extract_instruction_strings(payload[key]))
        return values
    return []


def get_episode_task(episode_dir: Path, task_override: str | None) -> str:
    if task_override:
        return task_override
    instructions_path = episode_dir / "instructions.json"
    if not instructions_path.exists():
        return DEFAULT_TASK
    with instructions_path.open() as f:
        payload = json.load(f)
    for item in _extract_instruction_strings(payload):
        normalized = item.strip()
        if normalized and normalized.lower() != "null":
            return normalized
    return DEFAULT_TASK


def create_features(image_shape: tuple[int, int, int], horizon: int) -> dict[str, dict[str, Any]]:
    state_names = [
        "right_pos_x", "right_pos_y", "right_pos_z",
        "right_rot6d_0", "right_rot6d_1", "right_rot6d_2",
        "right_rot6d_3", "right_rot6d_4", "right_rot6d_5",
        "right_gripper",
        "left_pos_x", "left_pos_y", "left_pos_z",
        "left_rot6d_0", "left_rot6d_1", "left_rot6d_2",
        "left_rot6d_3", "left_rot6d_4", "left_rot6d_5",
        "left_gripper",
    ]
    return {
        "observation.state": {"dtype": "float32", "shape": (20,), "names": state_names},
        "action": {"dtype": "float32", "shape": (horizon, 20), "names": None},
        "observation.images.cam_high": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["channels", "height", "width"],
        },
        "observation.images.cam_left_wrist": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["channels", "height", "width"],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["channels", "height", "width"],
        },
    }


def create_dataset(
    repo_id: str,
    output_root: Path,
    fps: int,
    features: dict[str, dict[str, Any]],
    overwrite: bool,
    image_writer_processes: int,
    image_writer_threads: int,
) -> Any:
    from lerobot.datasets import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != "v3.0":
        raise RuntimeError(f"This converter targets LeRobot v3.0, got CODEBASE_VERSION={CODEBASE_VERSION!r}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_root,
        robot_type="pika_bimanual",
        use_videos=False,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )


def prepare_frame(
    files: dict[str, list[Path]],
    right: dict[str, np.ndarray],
    left: dict[str, np.ndarray],
    task: str,
    frame_idx: int,
    horizon: int,
    image_size: int,
) -> dict[str, Any]:
    state = np.concatenate([right["states"][frame_idx], left["states"][frame_idx]], axis=0).astype(np.float32)
    left_image = np.asarray(resize_with_pad(load_rgb_image(files["left_fisheye"][frame_idx]), image_size, image_size))
    right_image = np.asarray(resize_with_pad(load_rgb_image(files["right_fisheye"][frame_idx]), image_size, image_size))
    return {
        "task": task,
        "observation.state": state,
        "action": build_relative_action_chunk(right, left, frame_idx, horizon),
        "observation.images.cam_high": np.zeros((image_size, image_size, 3), dtype=np.uint8),
        "observation.images.cam_left_wrist": left_image,
        "observation.images.cam_right_wrist": right_image,
    }


def iter_prepared_frames(
    files: dict[str, list[Path]],
    right: dict[str, np.ndarray],
    left: dict[str, np.ndarray],
    task: str,
    written: int,
    horizon: int,
    image_size: int,
    frame_workers: int,
):
    args = (files, right, left, task)
    if frame_workers == 1:
        for frame_idx in range(written):
            yield prepare_frame(*args, frame_idx, horizon, image_size)
        return

    executor = ThreadPoolExecutor(max_workers=frame_workers)
    pending: deque[Future[dict[str, Any]]] = deque()
    next_frame_idx = 0
    try:
        while next_frame_idx < min(written, 2 * frame_workers):
            pending.append(executor.submit(prepare_frame, *args, next_frame_idx, horizon, image_size))
            next_frame_idx += 1

        while pending:
            yield pending.popleft().result()
            if next_frame_idx < written:
                pending.append(executor.submit(prepare_frame, *args, next_frame_idx, horizon, image_size))
                next_frame_idx += 1
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def add_episode_to_dataset(
    dataset: Any,
    episode_dir: Path,
    horizon: int,
    task_override: str | None,
    image_size: int,
    frame_workers: int,
) -> dict[str, Any]:
    files = discover_episode_files(episode_dir)
    frame_count = len(files["right_pose"])
    if frame_count <= horizon:
        raise ValueError(f"Episode {episode_dir} has {frame_count} frames, not enough for horizon={horizon}")

    right = build_arm_series(files["right_pose"], files["right_gripper"])
    left = build_arm_series(files["left_pose"], files["left_gripper"])
    task = get_episode_task(episode_dir, task_override)

    written = frame_count - horizon
    frames = iter_prepared_frames(files, right, left, task, written, horizon, image_size, frame_workers)
    for frame in frames:
        dataset.add_frame(frame)
    dataset.save_episode()
    return {"episode": str(episode_dir), "source_frames": frame_count, "written_frames": written, "task": task}


def episode_sort_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"episode(\d+)", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**9, path.name)


def discover_episode_dirs(input_root: Path, limit_episodes: int | None) -> list[Path]:
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")
    episode_dirs = sorted((p for p in input_root.iterdir() if p.is_dir()), key=episode_sort_key)
    if limit_episodes is not None:
        episode_dirs = episode_dirs[:limit_episodes]
    if not episode_dirs:
        raise ValueError(f"No episode directories found under: {input_root}")
    return episode_dirs


def convert_dataset(
    input_root: Path,
    output_root: Path,
    repo_id: str,
    fps: int,
    horizon: int,
    task: str | None,
    overwrite: bool,
    limit_episodes: int | None,
    image_writer_processes: int,
    image_writer_threads: int,
    image_size: int,
    frame_workers: int,
) -> dict[str, Any]:
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if frame_workers <= 0:
        raise ValueError(f"frame_workers must be positive, got {frame_workers}")

    episode_dirs = discover_episode_dirs(input_root, limit_episodes)
    image_shape = (3, image_size, image_size)
    features = create_features(image_shape, horizon)

    dataset = create_dataset(
        repo_id=repo_id,
        output_root=output_root,
        fps=fps,
        features=features,
        overwrite=overwrite,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )

    episode_manifests: list[dict[str, Any]] = []
    try:
        for idx, episode_dir in enumerate(episode_dirs, start=1):
            logging.info("Converting episode %s/%s: %s", idx, len(episode_dirs), episode_dir)
            episode_manifests.append(
                add_episode_to_dataset(dataset, episode_dir, horizon, task, image_size, frame_workers)
            )
        dataset.finalize()
    except Exception:
        dataset.finalize()
        raise

    manifest = {
        "repo_id": repo_id,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "lerobot_codebase_version": "v3.0",
        "fps": fps,
        "horizon": horizon,
        "image_size": image_size,
        "frame_workers": frame_workers,
        "image_writer_processes": image_writer_processes,
        "image_writer_threads": image_writer_threads,
        "total_episodes": len(episode_manifests),
        "total_written_frames": sum(ep["written_frames"] for ep in episode_manifests),
        "features": features,
        "episodes": episode_manifests,
    }
    manifest_path = output_root / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--task", help="Optional task text override for every episode.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output dataset root.")
    parser.add_argument("--limit-episodes", type=int, help="Convert only the first N episodes for testing.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--frame-workers", type=int, default=DEFAULT_FRAME_WORKERS)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    manifest = convert_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        repo_id=args.repo_id,
        fps=args.fps,
        horizon=args.horizon,
        task=args.task,
        overwrite=args.overwrite,
        limit_episodes=args.limit_episodes,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        image_size=args.image_size,
        frame_workers=args.frame_workers,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
