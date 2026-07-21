"""Stage 1+2 orchestration: manifest -> YOLO crop -> MobileCLIP embed -> per-batch JSON.

Replaces the three duplicated Process.py/Processing.py scripts previously
used for preparing classifier training data, classifier test data, and
MobileCLIP standalone evaluation data - one function, parameterized by
which manifest/label-map/output-dir are passed in.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from fod_pipeline.core.detection import (
    DEFAULT_EXPANSION,
    crop_yolo_detection,
    extract_yolo_weights,
    load_yolo,
)
from fod_pipeline.core.device import get_device
from fod_pipeline.core.embedding import embedding_to_list, encode_image, load_mobileclip
from fod_pipeline.core.labels import load_label_map
from fod_pipeline.core.s3_io import (
    extract_batch_id,
    extract_object_id,
    load_image_from_s3,
    load_manifest,
)


CHECKPOINT_EVERY = 200


def process_manifest(
    manifest_path: str,
    label_map_path: str,
    yolo_model,
    mobileclip_model,
    mobileclip_preprocess,
    device,
    expansion: float = DEFAULT_EXPANSION,
    max_images: int = -1,
    fp16: bool = False,
    output_dir: str | None = None,
) -> tuple:
    """Run Stage 1+2 over every image in a manifest.

    A single image's failure (corrupt file, transient S3 error, unexpected
    filename) is logged and skipped rather than aborting the whole run. If
    output_dir is given, results are checkpointed to disk every
    CHECKPOINT_EVERY images so a crash partway through a large manifest
    doesn't lose everything already processed.

    Returns (batch_results, failures), where batch_results is
    {batch_id: [record, ...]} and each record is
    {"image", "objectID", "label", "embedding"}.
    """
    label_map = load_label_map(label_map_path)
    prefix, image_paths = load_manifest(manifest_path)

    if max_images != -1:
        image_paths = image_paths[:max_images]

    batch_results = {}
    failures = []

    for i, image_path in enumerate(image_paths, start=1):
        try:
            object_id = extract_object_id(image_path)
            label = label_map.get(object_id, "UNKNOWN")

            image = load_image_from_s3(prefix + image_path)

            yolo_results = yolo_model(image)
            crop, _bbox = crop_yolo_detection(image, yolo_results, expansion=expansion)

            features = encode_image(
                mobileclip_model, mobileclip_preprocess, device, crop, fp16=fp16
            )
            embedding = embedding_to_list(features)

            record = {
                "image": os.path.basename(image_path),
                "objectID": object_id,
                "label": label,
                "embedding": embedding,
            }

        except Exception as e:
            failures.append({"image": image_path, "error": repr(e)})
            print(f"ERROR {image_path}: {e!r}")
            continue

        batch_id = extract_batch_id(image_path)
        batch_results.setdefault(batch_id, []).append(record)

        if i % CHECKPOINT_EVERY == 0 or i == len(image_paths):
            print(f"{i}/{len(image_paths)} processed ({len(failures)} failed)")
            if output_dir:
                save_batch_results(batch_results, output_dir)

    if failures and output_dir:
        with open(os.path.join(output_dir, "failures.json"), "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
        print(f"WARNING: {len(failures)}/{len(image_paths)} images failed - see failures.json")

    return batch_results, failures


def save_batch_results(batch_results: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for batch_id, records in batch_results.items():
        save_path = os.path.join(output_dir, f"{batch_id}_embeddings.json")
        with open(save_path, "w") as f:
            json.dump(records, f, indent=4)
        print(f"Saved {batch_id}: {len(records)} images")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--label-map", type=str, required=True)
    parser.add_argument("--yolo", type=str, default="best.pt")
    parser.add_argument(
        "--yolo-tar", type=str, default=None, help="SageMaker model.tar.gz containing best.pt"
    )
    parser.add_argument("--mobileclip", type=str, default="mobileclip_s0.pt")
    parser.add_argument("--output-dir", type=str, default="embeddings")
    parser.add_argument(
        "--max-images", type=int, default=-1, help="-1 processes every image in the manifest"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run YOLO/MobileCLIP in fp16 on GPU for faster inference (no effect on CPU)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("CUDA available:", torch.cuda.is_available())

    yolo_path = extract_yolo_weights(args.yolo_tar) if args.yolo_tar else args.yolo

    device = get_device()
    yolo_model = load_yolo(yolo_path, device=device, fp16=args.fp16)
    mobileclip_model, preprocess, _tokenizer, device = load_mobileclip(
        args.mobileclip, device=device, fp16=args.fp16
    )

    process_manifest(
        manifest_path=args.manifest,
        label_map_path=args.label_map,
        yolo_model=yolo_model,
        mobileclip_model=mobileclip_model,
        mobileclip_preprocess=preprocess,
        device=device,
        max_images=args.max_images,
        fp16=args.fp16,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
