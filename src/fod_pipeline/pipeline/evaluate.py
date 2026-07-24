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

from fod_pipeline.classifier.dataset import extract_classifier_weights, load_label_names
from fod_pipeline.core.detection import extract_yolo_weights
from fod_pipeline.core.labels import canonical_label, load_label_map
from fod_pipeline.core.s3_io import extract_object_id, load_image_from_s3, load_manifest
from fod_pipeline.hybrid.metrics import build_metrics_report, is_hybrid_correct
from fod_pipeline.pipeline.infer import build_pipeline

RECORD_FIELDS = [
    "image",
    "object_id",
    "mobileclip_gt",
    "mobileclip_top1",
    "mobileclip_top2",
    "classifier_gt",
    "classifier_pred",
    "yolo_detected",
    "mobileclip_ms",
    "classifier_ms",
    "pipeline_ms",
    "hybrid_correct",
]

PROGRESS_EVERY = 50


def evaluate_manifest(
    pipeline,
    manifest_path: str,
    mobileclip_label_map_path: str,
    classifier_label_map_path: str,
    output_dir: str,
    max_images: int = -1,
    classifier_fallback: bool = False,
) -> tuple:
    """Run the hybrid pipeline over every image in a manifest, pairing each
    prediction with its dual ground truth.

    Writes predictions.csv incrementally (one flush per image) so a crash
    partway through a large manifest doesn't lose everything already
    processed. A single image's failure (corrupt file, transient S3 error,
    unexpected filename) is logged and skipped rather than aborting the
    whole run - matching the resilience the original MobileCLIP_Alone
    evaluation script had, extended here to the classifier/hybrid path too.

    Each row also gets a hybrid_correct column (the same OR-rule used for
    the aggregate hybrid_accuracy metric) so individual predictions can be
    eyeballed directly in predictions.csv without cross-referencing the
    metrics report.

    Returns (records, failures) - records shaped for fod_pipeline.hybrid.metrics.
    """
    mobileclip_labels = load_label_map(mobileclip_label_map_path)
    classifier_labels = load_label_map(classifier_label_map_path)

    prefix, image_paths = load_manifest(manifest_path)

    if max_images != -1:
        image_paths = image_paths[:max_images]

    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "predictions.csv")

    records = []
    failures = []

    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        writer.writeheader()

        for i, image_path in enumerate(image_paths, start=1):
            try:
                object_id = extract_object_id(image_path)
                # Same synonym mapping the pipeline applies to its own
                # MobileCLIP predictions, so both sides of the Ground Truth A
                # comparison land in the same vocabulary (see HybridPipeline.predict).
                mobileclip_gt = canonical_label(
                    mobileclip_labels.get(object_id, "UNKNOWN"),
                    pipeline.mobileclip_synonym_mapping,
                )
                classifier_gt = classifier_labels.get(object_id, "UNKNOWN")

                image = load_image_from_s3(prefix + image_path)
                prediction = pipeline.predict(image)

                record = {
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
                record["hybrid_correct"] = is_hybrid_correct(record, classifier_fallback)

            except Exception as e:
                failures.append({"image": image_path, "error": repr(e)})
                print(f"ERROR {image_path}: {e!r}")
                continue

            records.append(record)
            writer.writerow(record)
            f.flush()

            if i % PROGRESS_EVERY == 0 or i == len(image_paths):
                print(
                    f"{i}/{len(image_paths)} processed "
                    f"({len(failures)} failed) - last: {prediction.candidates}"
                )

    if failures:
        with open(os.path.join(output_dir, "failures.json"), "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
        print(f"WARNING: {len(failures)}/{len(image_paths)} images failed - see failures.json")

    return records, failures


def load_records_from_csv(predictions_path: str) -> list:
    """Reload records from a predictions.csv written by a previous
    evaluate_manifest() run - lets metrics be recomputed (e.g. with a
    different classifier_fallback setting) without re-running inference.
    """
    records = []
    with open(predictions_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["yolo_detected"] = (
                row["yolo_detected"].lower() == "true" if row["yolo_detected"] else None
            )
            if row.get("hybrid_correct"):
                row["hybrid_correct"] = row["hybrid_correct"].lower() == "true"
            for ms_field in ("mobileclip_ms", "classifier_ms", "pipeline_ms"):
                row[ms_field] = float(row[ms_field]) if row[ms_field] else None
            records.append(row)
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


def save_metrics_report(report: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(
        os.path.join(output_dir, "metrics_report.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--from-predictions",
        type=str,
        default=None,
        help="Recompute the metrics report from a predictions.csv written by a previous "
        "run, instead of re-running the pipeline. Only --label-encoder, --output-dir, "
        "and --classifier-fallback are used in this mode.",
    )
    parser.add_argument("--manifest", type=str, required=False)
    parser.add_argument(
        "--mobileclip-label-map",
        type=str,
        required=False,
        help="Ground Truth A (MobileCLIP taxonomy)",
    )
    parser.add_argument(
        "--classifier-label-map",
        type=str,
        required=False,
        help="Ground Truth B (classifier taxonomy)",
    )
    parser.add_argument("--yolo", type=str, default="best.pt")
    parser.add_argument(
        "--yolo-tar", type=str, default=None, help="SageMaker model.tar.gz containing best.pt"
    )
    parser.add_argument("--mobileclip", type=str, required=False)
    parser.add_argument("--mobileclip-model-name", type=str, default="mobileclip_s0")
    parser.add_argument("--prompts", type=str, default=None)
    parser.add_argument(
        "--mobileclip-mapping",
        type=str,
        default=None,
        help="category_to_objects synonym map translating MobileCLIP's category "
        "vocabulary into the dataset's ground-truth vocabulary",
    )
    parser.add_argument("--classifier-weights", type=str, required=False)
    parser.add_argument(
        "--classifier-weights-tar",
        type=str,
        default=None,
        help="SageMaker model.tar.gz containing model.pth",
    )
    parser.add_argument("--label-encoder", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="evaluation-results")
    parser.add_argument(
        "--max-images", type=int, default=-1, help="-1 evaluates every image in the manifest"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run YOLO/MobileCLIP in fp16 on GPU for faster inference (no effect on CPU)",
    )
    parser.add_argument(
        "--classifier-fallback",
        action="store_true",
        help="Experimental: when MobileCLIP's top-1/top-2 miss Ground Truth A, also "
        "accept a match against Ground Truth B (classifier_gt) before calling it "
        "wrong. Recovers classes that used to be joined in MobileCLIP's ground "
        "truth and were recently split (e.g. manufactured wood/wood, bullet/bullet "
        "casings). Off by default - does not change existing behavior.",
    )

    args = parser.parse_args()

    if not args.from_predictions:
        required = {
            "--manifest": args.manifest,
            "--mobileclip-label-map": args.mobileclip_label_map,
            "--classifier-label-map": args.classifier_label_map,
            "--mobileclip": args.mobileclip,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(f"the following arguments are required: {', '.join(missing)}")
        if not args.classifier_weights and not args.classifier_weights_tar:
            parser.error(
                "one of the arguments --classifier-weights --classifier-weights-tar "
                "is required"
            )

    return args


def main():
    args = parse_args()

    if args.from_predictions:
        records = load_records_from_csv(args.from_predictions)
        failures = []
        class_names = load_label_names(args.label_encoder)
    else:
        yolo_path = extract_yolo_weights(args.yolo_tar) if args.yolo_tar else args.yolo
        classifier_weights_path = (
            extract_classifier_weights(args.classifier_weights_tar)
            if args.classifier_weights_tar
            else args.classifier_weights
        )

        pipeline = build_pipeline(
            yolo_path=yolo_path,
            mobileclip_path=args.mobileclip,
            classifier_weights_path=classifier_weights_path,
            label_encoder_path=args.label_encoder,
            mobileclip_model_name=args.mobileclip_model_name,
            prompts_path=args.prompts,
            mobileclip_mapping_path=args.mobileclip_mapping,
            fp16=args.fp16,
        )

        records, failures = evaluate_manifest(
            pipeline,
            manifest_path=args.manifest,
            mobileclip_label_map_path=args.mobileclip_label_map,
            classifier_label_map_path=args.classifier_label_map,
            output_dir=args.output_dir,
            max_images=args.max_images,
            classifier_fallback=args.classifier_fallback,
        )
        class_names = pipeline.classifier_class_names

    if not records:
        report = {"error": "no images processed successfully", "num_failures": len(failures)}
        save_metrics_report(report, args.output_dir)
        print(json.dumps(report, indent=2))
        return

    classifier_y_true, classifier_y_pred = classifier_label_arrays(records, class_names)

    report = build_metrics_report(
        records,
        classifier_y_true,
        classifier_y_pred,
        classifier_fallback=args.classifier_fallback,
    )
    report["num_failures"] = len(failures)

    save_metrics_report(report, args.output_dir)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
