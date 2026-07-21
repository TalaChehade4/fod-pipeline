"""Stage 4/5 orchestration: run the hybrid pipeline over a labeled test
manifest and produce the full metrics report (Sections 4 and 5).

Each image needs two ground truths, since MobileCLIP and the classifier
predict over different label spaces (see fod_pipeline.hybrid.metrics):
  - mobileclip_gt : Ground Truth A, MobileCLIP's own taxonomy
  - classifier_gt : Ground Truth B, the classifier's taxonomy

This single script replaces what used to be three separate pieces wired
together by hand-matched CSVs: MobileCLIP_Alone's evaluation loop,
TestingClassifier's evaluation loop, and resultsOfHybrid's CSV-combination
step. Running one HybridPipeline per image and comparing it against both
ground truths in the same pass removes that manual matching entirely.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from fod_pipeline.core.detection import extract_yolo_weights
from fod_pipeline.core.labels import load_label_map
from fod_pipeline.core.s3_io import extract_object_id, load_image_from_s3, load_manifest
from fod_pipeline.hybrid.metrics import build_metrics_report
from fod_pipeline.pipeline.infer import build_pipeline


def evaluate_manifest(
    pipeline,
    manifest_path: str,
    mobileclip_label_map_path: str,
    classifier_label_map_path: str,
) -> list:
    """Run the hybrid pipeline over every image in a manifest, pairing each
    prediction with its dual ground truth.

    Returns a list of per-image records shaped for fod_pipeline.hybrid.metrics.
    """
    mobileclip_labels = load_label_map(mobileclip_label_map_path)
    classifier_labels = load_label_map(classifier_label_map_path)

    prefix, image_paths = load_manifest(manifest_path)

    records = []

    for i, image_path in enumerate(image_paths, start=1):
        object_id = extract_object_id(image_path)
        mobileclip_gt = mobileclip_labels.get(object_id, "UNKNOWN")
        classifier_gt = classifier_labels.get(object_id, "UNKNOWN")

        image = load_image_from_s3(prefix + image_path)
        prediction = pipeline.predict(image)

        records.append(
            {
                "image": os.path.basename(image_path),
                "object_id": object_id,
                "mobileclip_gt": mobileclip_gt,
                "mobileclip_top1": prediction.mobileclip_top1,
                "mobileclip_top2": prediction.mobileclip_top2,
                "classifier_gt": classifier_gt,
                "classifier_pred": prediction.classifier_prediction,
                "yolo_detected": prediction.yolo_detected,
                "mobileclip_ms": prediction.mobileclip_ms,
                "classifier_ms": prediction.classifier_ms,
                "pipeline_ms": prediction.pipeline_ms,
            }
        )

        print(f"{i}/{len(image_paths)} {image_path} -> {prediction.candidates}")

    return records


def classifier_label_arrays(records: list, class_names: list) -> tuple:
    """Encode classifier_gt/classifier_pred to label-encoder indices for the
    Section 5.2 metrics, skipping any record whose ground truth falls
    outside the classifier's known classes.
    """
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    known = [r for r in records if r["classifier_gt"] in class_to_idx]
    skipped = len(records) - len(known)
    if skipped:
        print(f"WARNING: {skipped} records have a classifier_gt outside the known classes; excluded from Section 5.2 metrics")

    y_true = [class_to_idx[r["classifier_gt"]] for r in known]
    y_pred = [class_to_idx[r["classifier_pred"]] for r in known]

    return y_true, y_pred


def save_report(records: list, report: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(
        os.path.join(output_dir, "predictions.csv"), "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    with open(
        os.path.join(output_dir, "metrics_report.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument(
        "--mobileclip-label-map",
        type=str,
        required=True,
        help="Ground Truth A (MobileCLIP taxonomy)",
    )
    parser.add_argument(
        "--classifier-label-map",
        type=str,
        required=True,
        help="Ground Truth B (classifier taxonomy)",
    )
    parser.add_argument("--yolo", type=str, default="best.pt")
    parser.add_argument(
        "--yolo-tar", type=str, default=None, help="SageMaker model.tar.gz containing best.pt"
    )
    parser.add_argument("--mobileclip", type=str, required=True)
    parser.add_argument("--mobileclip-model-name", type=str, default="mobileclip_s0")
    parser.add_argument("--prompts", type=str, default=None)
    parser.add_argument("--classifier-weights", type=str, required=True)
    parser.add_argument("--label-encoder", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="evaluation-results")

    return parser.parse_args()


def main():
    args = parse_args()

    yolo_path = extract_yolo_weights(args.yolo_tar) if args.yolo_tar else args.yolo

    pipeline = build_pipeline(
        yolo_path=yolo_path,
        mobileclip_path=args.mobileclip,
        classifier_weights_path=args.classifier_weights,
        label_encoder_path=args.label_encoder,
        mobileclip_model_name=args.mobileclip_model_name,
        prompts_path=args.prompts,
    )

    records = evaluate_manifest(
        pipeline,
        manifest_path=args.manifest,
        mobileclip_label_map_path=args.mobileclip_label_map,
        classifier_label_map_path=args.classifier_label_map,
    )

    classifier_y_true, classifier_y_pred = classifier_label_arrays(
        records, pipeline.classifier_class_names
    )

    report = build_metrics_report(records, classifier_y_true, classifier_y_pred)

    save_report(records, report, args.output_dir)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
