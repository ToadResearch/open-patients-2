#!/usr/bin/env python3
"""
Push the enriched Open-Patients dataset to Hugging Face Hub.

Example:
  uv run python src/push_to_hf.py \
    --data_dir ./open_patients_enriched \
    --repo_name open-patients-enriched \
    --org my-organization \
    --private
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Push enriched Open-Patients dataset to Hugging Face Hub."
    )
    ap.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing the enriched JSONL shards (e.g., ./open_patients_enriched)",
    )
    ap.add_argument(
        "--repo_name",
        required=True,
        help="Name of the dataset repository on Hugging Face (e.g., open-patients-enriched)",
    )
    ap.add_argument(
        "--org",
        default=None,
        help="Organization or username to push to. If not provided, pushes to your personal account.",
    )
    ap.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset repository private",
    )
    ap.add_argument(
        "--commit_message",
        default="Upload enriched Open-Patients dataset",
        help="Commit message for the push",
    )
    ap.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token. If not provided, uses the cached token from `huggingface-cli login`.",
    )
    ap.add_argument(
        "--parquet",
        action="store_true",
        help="Convert to Parquet format before pushing (recommended for large datasets)",
    )
    ap.add_argument(
        "--max_shard_size",
        default="500MB",
        help="Maximum shard size for Parquet files (e.g., '500MB', '1GB')",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all JSONL files
    jsonl_files = list(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")

    print(f"Found {len(jsonl_files)} JSONL file(s) in {data_dir}")

    # Load dataset from JSONL shards
    print("Loading dataset...")
    ds = load_dataset(
        "json",
        data_files=str(data_dir / "*.jsonl"),
        split="train",
    )
    print(f"Loaded {len(ds)} records with columns: {ds.column_names}")

    # Build repo ID
    if args.org:
        repo_id = f"{args.org}/{args.repo_name}"
    else:
        repo_id = args.repo_name

    print(f"Pushing to: {repo_id}")

    # Create the repo if it doesn't exist
    api = HfApi(token=args.token)
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
    except Exception as e:
        print(f"Note: Could not create repo (may already exist): {e}")

    # Push to hub
    if args.parquet:
        print(f"Converting to Parquet and pushing (max shard size: {args.max_shard_size})...")
        ds.push_to_hub(
            repo_id,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message,
            max_shard_size=args.max_shard_size,
        )
    else:
        print("Pushing as JSONL...")
        ds.push_to_hub(
            repo_id,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message,
        )

    print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
