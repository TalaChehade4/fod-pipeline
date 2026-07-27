"""
Generate MobileCLIP embeddings from images listed in an S3 manifest.

Pipeline stages:

    1. Load image paths from a manifest file.
    2. Retrieve images from S3.
    3. Extract object IDs and corresponding labels.
    4. Run YOLO object detection to locate the FOD object.
    5. Crop the detected region with an optional expansion margin.
    6. Encode cropped images using MobileCLIP into feature embeddings.
    7. Save embeddings grouped by batch as JSON files.

The generated embeddings are later used to train the downstream
classifier.

The pipeline supports:
    - GPU/CPU execution
    - FP16 inference on CUDA devices
    - Batched MobileCLIP encoding for faster processing
    - Checkpoint saving during long SageMaker jobs
    - Failure logging without stopping the entire run

Input:
    - S3 manifest containing image paths
    - Object ID -> label mapping
    - YOLO detector weights
    - MobileCLIP weights

Output:
    - {batch_id}_embeddings.json files containing:
        {
            "image": image filename,
            "objectID": object identifier,
            "label": ground truth label,
            "embedding": MobileCLIP feature vector
        }
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
from fod_pipeline.core.embedding import embeddings_to_list, encode_images, load_mobileclip
from fod_pipeline.core.labels import load_label_map
from fod_pipeline.core.s3_io import (
    extract_batch_id,
    extract_object_id,
    load_image_from_s3,
    load_manifest,
)


CHECKPOINT_EVERY = 200
DEFAULT_EMBED_BATCH_SIZE = 64


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
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> tuple:
    """Run over every image in a manifest.

    YOLO detection/cropping runs one image at a time, but the cropped
    detections are buffered and pushed through MobileCLIP embed_batch_size
    at a time - one batched forward pass is much faster than one per image.

    A single image's failure (corrupt file, transient S3 error, unexpected
    filename) is logged and skipped rather than aborting the whole run.
    """
    label_map = load_label_map(label_map_path)
    prefix, image_paths = load_manifest(manifest_path)

    if max_images != -1:
        image_paths = image_paths[:max_images]

    batch_results = {}
    failures = []

    pending_crops = []
    pending_records = []

    def flush_pending():
        if not pending_crops:
            return

        features = encode_images(
            mobileclip_model, mobileclip_preprocess, device, pending_crops, fp16=fp16
        )
        embeddings = embeddings_to_list(features)

        for (record, image_path), embedding in zip(pending_records, embeddings):
            record["embedding"] = embedding
            batch_id = extract_batch_id(image_path)
            batch_results.setdefault(batch_id, []).append(record)

        pending_crops.clear()
        pending_records.clear()

    for i, image_path in enumerate(image_paths, start=1):
        try:
            object_id = extract_object_id(image_path)
            label = label_map.get(object_id, "UNKNOWN")

            image = load_image_from_s3(prefix + image_path)

            yolo_results = yolo_model(image)
            crop, _bbox = crop_yolo_detection(image, yolo_results, expansion=expansion)

            record = {
                "image": os.path.basename(image_path),
                "objectID": object_id,
                "label": label,
            }
            pending_crops.append(crop)
            pending_records.append((record, image_path))

        except Exception as e:
            failures.append({"image": image_path, "error": repr(e)})
            print(f"ERROR {image_path}: {e!r}")
            continue

        if len(pending_crops) >= embed_batch_size:
            flush_pending()

        if i % CHECKPOINT_EVERY == 0 or i == len(image_paths):
            flush_pending()
            print(f"{i}/{len(image_paths)} processed ({len(failures)} failed)")
            if output_dir:
                save_batch_results(batch_results, output_dir)

    flush_pending()

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
        "--embed-batch-size",
        type=int,
        default=DEFAULT_EMBED_BATCH_SIZE,
        help="Number of cropped detections encoded per MobileCLIP forward pass",
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
        embed_batch_size=args.embed_batch_size,
    )


if __name__ == "__main__":
    main()
