#!/usr/bin/env python3
"""
Convert BimanualUR5eExample WebDataset to OpenPI-compatible LeRobot v2 format.

Source: robotics-diffusion-transformer/BimanualUR5eExample (~57 GB, 72 tar shards)
Target: LeRobot v2 dataset (Parquet), ready for OpenPI pi05_umi_bimanual training

Usage:
    python convert_bimanual_to_lerobot.py \
        --shards-dir /home/ps/datasets/bimanual_ur5e/shards \
        --instructions /home/ps/datasets/bimanual_ur5e/instructions.json \
        --repo-id <your_hf_user>/bimanual_ur5e_lerobot \
        --episode-size 5000 \
        --image-writer-processes 4

Dependencies:
    conda activate openpi  (provides lerobot, numpy, PIL, torch)
"""

import argparse
import glob
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


def create_empty_dataset(
    repo_id: str,
    fps: int = 30,
    *,
    image_writer_processes: int = 4,
    image_writer_threads: int = 4,
) -> "LeRobotDataset":
    """Create an empty LeRobot dataset with the UMI bimanual feature schema."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (20,),
            "names": [
                "right_pos_x", "right_pos_y", "right_pos_z",
                "right_rot6d_0", "right_rot6d_1", "right_rot6d_2",
                "right_rot6d_3", "right_rot6d_4", "right_rot6d_5",
                "right_gripper",
                "left_pos_x", "left_pos_y", "left_pos_z",
                "left_rot6d_0", "left_rot6d_1", "left_rot6d_2",
                "left_rot6d_3", "left_rot6d_4", "left_rot6d_5",
                "left_gripper",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (10, 20),
            "names": None,
        },
        "observation.images.cam_high": {
            "dtype": "image",
            "shape": (3, 384, 384),
            "names": ["channels", "height", "width"],
        },
        "observation.images.cam_left_wrist": {
            "dtype": "image",
            "shape": (3, 384, 384),
            "names": ["channels", "height", "width"],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "image",
            "shape": (3, 384, 384),
            "names": ["channels", "height", "width"],
        },
    }

    # Remove existing dataset if present
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
    root = HF_LEROBOT_HOME / repo_id
    if root.exists():
        import shutil
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="bimanual_ur5e",
        features=features,
        use_videos=False,
        tolerance_s=1e-4,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )
    return dataset


def process_sample(
    tar: tarfile.TarFile,
    prefix: str,
    instructions: dict,
) -> dict | None:
    """
    Process one sample from the WebDataset tar shard.

    Returns a frame dict suitable for dataset.add_frame(), or None on error.
    """
    try:
        # 1. Read and split binocular image
        img_member = tar.getmember(f"{prefix}.image.jpg")
        img_bytes = tar.extractfile(img_member).read()
        img = Image.open(io.BytesIO(img_bytes))
        img_np = np.array(img)  # (H, W, 3) = (384, 768, 3), uint8

        if img_np.shape[1] != 768 or img_np.shape[0] != 384:
            print(f"  [WARN] {prefix}: unexpected image shape {img_np.shape}, expected (384, 768, 3)")

        # Split left/right halves
        mid = img_np.shape[1] // 2
        left_img = img_np[:, :mid, :]   # (384, 384, 3)
        right_img = img_np[:, mid:, :]  # (384, 384, 3)

        # HWC → CHW for LeRobot
        left_chw = np.transpose(left_img, (2, 0, 1))   # (3, 384, 384)
        right_chw = np.transpose(right_img, (2, 0, 1))  # (3, 384, 384)

        # Black placeholder for main camera (source has no scene camera)
        black_chw = np.zeros((3, 384, 384), dtype=np.uint8)

        # 2. Read action and slice to (10, 20)
        action_member = tar.getmember(f"{prefix}.action.npy")
        action = np.load(io.BytesIO(tar.extractfile(action_member).read()))  # (24, 20)
        action = action[1:11, :].astype(np.float32)  # skip action[0] (≈ zero), take 10 steps

        if action.shape != (10, 20):
            print(f"  [WARN] {prefix}: action shape {action.shape}, expected (10, 20) after slicing")

        # 3. Read meta and lookup instruction
        meta_member = tar.getmember(f"{prefix}.meta.json")
        meta = json.load(tar.extractfile(meta_member))
        instruction_key = meta.get("sub_task_instruction_key", "")
        instruction = instructions.get(instruction_key, instruction_key)

        # 4. Construct dummy state (identity pose per robot)
        identity_pose = np.array(
            [0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0, 1.0, 0.0,  0.088],  # right arm
            dtype=np.float32,
        )
        state = np.concatenate([identity_pose, identity_pose])  # (20,)

        frame = {
            "observation.state": state,
            "action": action,
            "observation.images.cam_high": black_chw,
            "observation.images.cam_left_wrist": left_chw,
            "observation.images.cam_right_wrist": right_chw,
            "task": instruction,
        }
        return frame

    except Exception as e:
        print(f"  [ERROR] {prefix}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert BimanualUR5eExample WebDataset to LeRobot v2 format"
    )
    parser.add_argument(
        "--shards-dir", required=True,
        help="Path to directory containing shard-*.tar files",
    )
    parser.add_argument(
        "--instructions", required=True,
        help="Path to instructions.json",
    )
    parser.add_argument(
        "--repo-id", required=True,
        help="HuggingFace repo ID for the output LeRobot dataset (e.g. <user>/bimanual_ur5e_lerobot)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second (default: 30)",
    )
    parser.add_argument(
        "--episode-size", type=int, default=5000,
        help="Number of frames per LeRobot episode (default: 5000). "
             "Data has no original episode boundaries, so we batch frames into synthetic episodes.",
    )
    parser.add_argument(
        "--image-writer-processes", type=int, default=4,
        help="Number of processes for async image encoding (default: 4)",
    )
    parser.add_argument(
        "--image-writer-threads", type=int, default=4,
        help="Number of threads per image writer process (default: 4)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of tar shards to process (for testing)",
    )
    args = parser.parse_args()

    shards_dir = Path(args.shards_dir)
    instructions_path = Path(args.instructions)

    if not shards_dir.is_dir():
        print(f"ERROR: shards-dir not found: {shards_dir}", file=sys.stderr)
        sys.exit(1)
    if not instructions_path.is_file():
        print(f"ERROR: instructions.json not found: {instructions_path}", file=sys.stderr)
        sys.exit(1)

    # Load instructions
    with open(instructions_path) as f:
        instructions = json.load(f)
    print(f"Loaded {len(instructions)} instructions")

    # Find all tar shards
    shard_files = sorted(glob.glob(str(shards_dir / "shard-*.tar")))
    if not shard_files:
        print(f"ERROR: no shard-*.tar files found in {shards_dir}", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        shard_files = shard_files[:args.limit]

    print(f"Found {len(shard_files)} shard files")
    print(f"Output repo: {args.repo_id}")
    print(f"Episode size: {args.episode_size}")
    print(f"FPS: {args.fps}")
    print("-" * 60)

    # Create the empty LeRobot dataset
    print("Creating LeRobot dataset...")
    dataset = create_empty_dataset(
        repo_id=args.repo_id,
        fps=args.fps,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    # Process all shards
    total_samples = 0
    total_errors = 0

    for shard_idx, shard_path in enumerate(shard_files):
        shard_name = os.path.basename(shard_path)
        print(f"\n[{shard_idx + 1}/{len(shard_files)}] {shard_name}")

        with tarfile.open(shard_path, "r") as tar:
            # Get all sample prefixes (from .meta.json files)
            meta_members = sorted(
                [m for m in tar.getmembers() if m.name.endswith(".meta.json")],
                key=lambda m: m.name,
            )

            for m in meta_members:
                prefix = m.name.replace(".meta.json", "")

                frame = process_sample(tar, prefix, instructions)
                if frame is None:
                    total_errors += 1
                    continue

                dataset.add_frame(frame)
                total_samples += 1

                # Save episode when buffer reaches episode_size
                if total_samples % args.episode_size == 0:
                    dataset.save_episode()
                    ep_idx = total_samples // args.episode_size
                    print(f"  Saved episode {ep_idx} ({total_samples:,} total frames)")

    # Save final partial episode
    if total_samples % args.episode_size != 0:
        dataset.save_episode()
        ep_idx = (total_samples // args.episode_size) + 1
        print(f"  Saved final episode {ep_idx} ({total_samples:,} total frames)")

    print(f"\n{'=' * 60}")
    print(f"Conversion complete!")
    print(f"  Total frames (samples): {total_samples:,}")
    print(f"  Total errors:           {total_errors}")
    print(f"  Dataset:                {args.repo_id}")
    print(f"\nNext steps:")
    print(f"  1. Compute norm stats:")
    print(f"     cd ../openpi && uv run scripts/compute_norm_stats.py \\")
    print(f"         --config-name pi05_umi_bimanual \\")
    print(f"         --data.repo-id {args.repo_id}")
    print(f"  2. (Optional) Push to HuggingFace Hub:")
    print(f"     python -c \"from lerobot.datasets.lerobot_dataset import LeRobotDataset; "
          f"ds = LeRobotDataset('{args.repo_id}'); ds.push_to_hub()\"")
    print(f"  3. Add pi05_umi_bimanual config to OpenPI config.py (see docs/convert_rdt2_webdataset_to_openpi.md)")
    print(f"  4. Train with OpenPI")


if __name__ == "__main__":
    main()
