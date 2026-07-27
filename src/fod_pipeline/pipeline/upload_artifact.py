"""
Upload pipeline artifacts to their configured S3 locations.

This utility provides a single command-line interface for uploading
different types of project files:

    - YOLO detector weights
    - MobileCLIP model weights
    - Dataset manifests
    - Label mappings
    - Database CSV files
    - Join configurations
    - MobileCLIP category mappings

The destination S3 path is not hardcoded. It is resolved from the
project configuration module.

The script validates required arguments and uploads files using the
central S3 I/O utilities.
"""
from __future__ import annotations

import argparse

from fod_pipeline.config import get_config, require_s3_bucket
from fod_pipeline.core.s3_io import upload_to_s3

SPLIT_REQUIRED_KINDS = {"manifest", "label-map", "mobileclip-label-map", "database-csv"}
KIND_CHOICES = [
    "yolo-weights",
    "mobileclip-weights",
    "join-config",
    "mobileclip-mapping",
    *SPLIT_REQUIRED_KINDS,
]


def resolve_destination(config, kind: str, split: str | None) -> str:
    if kind in SPLIT_REQUIRED_KINDS and not split:
        raise ValueError(f"--split is required for --kind {kind}")

    if kind == "yolo-weights":
        return config.s3.yolo_weights_tar
    if kind == "mobileclip-weights":
        return config.s3.mobileclip_weights
    if kind == "manifest":
        return config.s3.manifest(split)
    if kind == "label-map":
        return config.s3.label_map(split)
    if kind == "mobileclip-label-map":
        return config.s3.mobileclip_label_map(split)
    if kind == "database-csv":
        return config.s3.database_csv(split)
    if kind == "join-config":
        return config.s3.join_config
    if kind == "mobileclip-mapping":
        return config.s3.mobileclip_mapping

    raise ValueError(f"Unknown kind: {kind}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("local_path", type=str)
    parser.add_argument("--kind", type=str, required=True, choices=KIND_CHOICES)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default=None,
        help=f"Required for --kind {sorted(SPLIT_REQUIRED_KINDS)}",
    )

    args = parser.parse_args()

    if args.kind in SPLIT_REQUIRED_KINDS and not args.split:
        parser.error(f"--split is required for --kind {args.kind}")

    return args


def main():
    args = parse_args()

    config = get_config()
    require_s3_bucket(config)

    destination = resolve_destination(config, args.kind, args.split)
    upload_to_s3(args.local_path, destination)

    print(f"Uploaded {args.local_path} -> {destination}")


if __name__ == "__main__":
    main()
