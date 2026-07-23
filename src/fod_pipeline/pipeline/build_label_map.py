"""Stage 0: build an object_id -> label label_map.json from a manifest plus
a lookup source.

Every other stage (fod-preprocess, fod-evaluate, their fod-sm-* equivalents)
only *reads* label_map.json - nothing produces it. This fills that gap.

The two ground truths this pipeline needs both come from this one builder:
  - classifier ground truth: --join-config (curated overrides) checked first,
    --csv (e.g. trainingdata_old.csv/testingdata_old.csv) fills the rest
  - MobileCLIP ground truth: --csv only, no --join-config

--manifest, --csv, --join-config, and --output each accept either a local
path or an s3:// URI - s3:// inputs are downloaded to a temp file before
reading, and an s3:// --output is uploaded after writing, so the whole
run-only-against-S3 workflow needs no manual download/upload step.

--manifest, --csv, --id-column, --join-config, and --output can additionally all be left
out entirely in favor of --split (+ --ground-truth for --output/--join-config)
- each then defaults to the manifests/<split>_*.json (or trainingdata_old.csv/
testingdata_old.csv/join_config.json) path under S3_BUCKET/S3_PROJECT_PREFIX
that fod-sm-embed/fod-sm-evaluate/fod-upload already read/write by default,
so nothing needs typing out once those files are uploaded once via
`fod-upload --kind database-csv/join-config`.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

from fod_pipeline.config import get_config, require_s3_bucket
from fod_pipeline.core.labels import (
    extract_object_ids_from_manifest,
    load_csv_id_name_map,
    load_join_config,
    map_object_ids_to_labels,
)
from fod_pipeline.core.s3_io import download_from_s3, is_s3_uri, upload_to_s3


def build_label_map(
    manifest_path: str,
    csv_path: str | None = None,
    id_column: str | None = None,
    name_column: str = "name",
    join_config_path: str | None = None,
) -> tuple:
    """Returns (object_id -> label dict, sorted list of unmapped object IDs)."""
    object_ids = extract_object_ids_from_manifest(manifest_path)

    join_config = load_join_config(join_config_path) if join_config_path else None
    csv_fallback = (
        load_csv_id_name_map(csv_path, id_column, name_column) if csv_path else None
    )

    return map_object_ids_to_labels(object_ids, join_config, csv_fallback)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Local path or s3:// URI. Omit in favor of --split to default to "
        "manifests/<split>_manifest.json under S3_BUCKET/S3_PROJECT_PREFIX.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="id->name database, e.g. trainingdata_old.csv/testingdata_old.csv. Omit in "
        "favor of --split to default to manifests/<train|test>data_old.csv under "
        "S3_BUCKET/S3_PROJECT_PREFIX.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="CSV column holding the object ID (e.g. trainingID, testingID). Defaults "
        "to trainingID/testingID from --split when --csv is in use.",
    )
    parser.add_argument("--name-column", type=str, default="name")
    parser.add_argument(
        "--join-config",
        type=str,
        default=None,
        help="Curated object_id->label JSON, checked before --csv. Omit this for the "
        "MobileCLIP ground truth, which resolves from --csv alone. With --split and "
        "--ground-truth classifier, defaults to manifests/join_config.json under "
        "S3_BUCKET/S3_PROJECT_PREFIX.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Local path or s3:// URI. Omit in favor of --split + --ground-truth to "
        "default to manifests/<split>_label_map.json (classifier) or "
        "manifests/<split>_mobileclip_label_map.json (mobileclip) under "
        "S3_BUCKET/S3_PROJECT_PREFIX.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default=None,
        help="Fills in the default --manifest, and (with --ground-truth) --output.",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        choices=["classifier", "mobileclip"],
        default=None,
        help="Which ground truth --split's default --output should target. "
        "Requires --split.",
    )

    args = parser.parse_args()

    if not args.manifest and not args.split:
        parser.error("either --manifest or --split is required")
    if not args.output and not (args.split and args.ground_truth):
        parser.error("either --output or --split + --ground-truth is required")

    return args


def resolve_manifest(args, config) -> str:
    return args.manifest or config.s3.manifest(args.split)


def resolve_output(args, config) -> str:
    if args.output:
        return args.output
    if args.ground_truth == "classifier":
        return config.s3.label_map(args.split)
    return config.s3.mobileclip_label_map(args.split)


def resolve_csv(args, config) -> str | None:
    if args.csv:
        return args.csv
    if args.split:
        return config.s3.database_csv(args.split)
    return None


def resolve_id_column(args) -> str | None:
    if args.id_column:
        return args.id_column
    if args.split:
        return "trainingID" if args.split == "train" else "testingID"
    return None


def resolve_join_config(args, config) -> str | None:
    if args.join_config:
        return args.join_config
    if args.split and args.ground_truth == "classifier":
        return config.s3.join_config
    return None


def _local_input(path: str | None, tmp_dir: str) -> str | None:
    if path is None or not is_s3_uri(path):
        return path

    local_path = os.path.join(tmp_dir, os.path.basename(path))
    download_from_s3(path, local_path)
    return local_path


def main():
    args = parse_args()

    config = get_config() if args.split else None
    if config:
        require_s3_bucket(config)

    manifest_uri = resolve_manifest(args, config)
    output_uri = resolve_output(args, config)
    csv_uri = resolve_csv(args, config)
    id_column = resolve_id_column(args)
    join_config_uri = resolve_join_config(args, config)

    if csv_uri and not id_column:
        raise SystemExit(
            "--id-column is required when --csv is given (or pass --split to default it)"
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = _local_input(manifest_uri, tmp_dir)
        csv_path = _local_input(csv_uri, tmp_dir)
        join_config_path = _local_input(join_config_uri, tmp_dir)

        resolved, unmapped = build_label_map(
            manifest_path=manifest_path,
            csv_path=csv_path,
            id_column=id_column,
            name_column=args.name_column,
            join_config_path=join_config_path,
        )

        if unmapped:
            print(f"WARNING: {len(unmapped)} object IDs have no label:")
            for object_id in unmapped[:10]:
                print(f"  {object_id}")

        if is_s3_uri(output_uri):
            local_output = os.path.join(tmp_dir, os.path.basename(output_uri))
            with open(local_output, "w", encoding="utf-8") as f:
                json.dump(resolved, f, indent=2)
            upload_to_s3(local_output, output_uri)
        else:
            with open(output_uri, "w", encoding="utf-8") as f:
                json.dump(resolved, f, indent=2)

    print(f"Wrote {len(resolved)} labels to {output_uri}")


if __name__ == "__main__":
    main()
