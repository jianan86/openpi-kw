"""Download FastUMI dataset — git sparse-checkout (only specified task).

Usage:
    HF_ENDPOINT=https://hf-mirror.com python fastumi_download.py dual_arm/Arrange_Toothbrush_and_Toothpaste
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ID = "IPEC-COMMUNITY/FastUMI_100k_lerobot"
DEFAULT_LOCAL_DIR = "./fastumi_data"


def get_repo_url():
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    if "hf-mirror.com" in endpoint:
        return f"https://hf-mirror.com/datasets/{REPO_ID}"
    return f"https://huggingface.co/datasets/{REPO_ID}"


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)


def download_task(task_pattern: str, local_dir: str):
    local_path = Path(local_dir).resolve()
    task_name = task_pattern.replace("/", "_")
    clone_dir = local_path / task_name
    repo_url = get_repo_url()

    print(f"Task:     {task_pattern}")
    print(f"Repo:     {repo_url}")
    print(f"Clone to: {clone_dir}")
    print()

    # Step 1: clone repo (metadata only, no LFS files)
    if not (clone_dir / ".git").exists():
        print("[1/4] git clone --no-checkout --filter=blob:none (metadata only, fast)...")
        run([
            "git", "clone",
            "--no-checkout",
            "--filter=blob:none",
            "--depth=1",
            repo_url,
            str(clone_dir),
        ])
    else:
        print("[1/4] Repo exists, fetching latest...")
        run(["git", "-C", str(clone_dir), "fetch", "--depth=1"])

    # Step 2: sparse checkout only the target folder
    print(f"[2/4] git sparse-checkout: {task_pattern}/")
    run(["git", "-C", str(clone_dir), "sparse-checkout", "init", "--cone"])
    run(["git", "-C", str(clone_dir), "sparse-checkout", "set", task_pattern])
    # 先清掉根目录已存在的文件避免 "already exists" 警告
    for f in [".gitattributes", "README.md"]:
        fp = clone_dir / f
        if fp.exists():
            fp.unlink()
    run(["git", "-C", str(clone_dir), "checkout", "HEAD"])

    # Step 3: pull LFS files
    print("[3/4] git lfs pull (downloading large files)...")
    run(["git", "-C", str(clone_dir), "lfs", "install"], capture_output=True)
    run([
        "git", "-C", str(clone_dir), "lfs", "pull",
        "--include", f"{task_pattern}/*",
    ])

    # Step 4: verify
    print("[4/4] Verifying...")
    data_dir = clone_dir / task_pattern
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    video_files = sorted(data_dir.glob("**/*.mp4"))
    print(f"  Parquet: {len(parquet_files)}")
    print(f"  Videos:  {len(video_files)}")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(root=str(data_dir))
        frame = ds[0]
        print(f"  Frames:  {len(ds)}")
        print(f"  Action:  {frame['action'].shape}")
        print(f"  State:   {frame['observation.state'].shape}")
    except Exception as e:
        print(f"  Warning: {e}")

    print(f"\nDone! {data_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download FastUMI task via git sparse-checkout")
    parser.add_argument("task", nargs="?", help="e.g. dual_arm/Arrange_Toothbrush_and_Toothpaste")
    parser.add_argument("--local-dir", "-o", default=DEFAULT_LOCAL_DIR)
    args = parser.parse_args()

    if not args.task:
        parser.print_help()
        print("\nExample:")
        print("  HF_ENDPOINT=https://hf-mirror.com python fastumi_download.py dual_arm/Arrange_Toothbrush_and_Toothpaste")
        sys.exit(1)

    download_task(args.task, args.local_dir)


if __name__ == "__main__":
    main()
