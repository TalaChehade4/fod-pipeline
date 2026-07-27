"""
Evaluation utilities for the hybrid FOD classification pipeline.

This module compares MobileCLIP predictions and classifier predictions
against their corresponding ground truths and produces evaluation metrics.

The pipeline uses two classification sources:

    1. MobileCLIP:
        - Ground Truth A taxonomy
        - Evaluated using Top-1 and Top-2 matching

    2. MLP classifier:
        - Ground Truth B taxonomy
        - Evaluated using standard classification metrics

The hybrid evaluation follows an OR rule:
    An image is considered correctly classified if either:
        - MobileCLIP predicts the correct category
        - The classifier predicts the correct category

Metrics produced:
    - MobileCLIP Top-1 / Top-2 accuracy
    - Classifier accuracy, precision, recall, F1
    - Hybrid accuracy
    - Hybrid balanced accuracy
    - Model contribution breakdown
    - Average inference latency
    - YOLO detection rate

This module operates only on prediction records generated during
evaluation and does not perform model inference.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from fod_pipeline.classifier.evaluate import compute_classification_metrics
from fod_pipeline.core.labels import normalize_label


def mobileclip_correct(record: dict, classifier_fallback: bool = False) -> bool:
    """Top-1 or Top-2 match against MobileCLIP's own ground truth (Ground Truth A).

    When neither top-1 nor top-2 matches Ground Truth A, also accept a match against Ground Truth B
    (classifier_gt). 
    """
    if (
        record["mobileclip_top1"] == record["mobileclip_gt"]
        or record["mobileclip_top2"] == record["mobileclip_gt"]
    ):
        return True

    if not classifier_fallback:
        return False

    classifier_gt = normalize_label(record["classifier_gt"])
    return (
        record["mobileclip_top1"] == classifier_gt
        or record["mobileclip_top2"] == classifier_gt
    )


def classifier_correct(record: dict) -> bool:
    """Match against the classifier's own ground truth (Ground Truth B)."""
    return record["classifier_pred"] == record["classifier_gt"]


def is_hybrid_correct(record: dict, classifier_fallback: bool = False) -> bool:
    return mobileclip_correct(record, classifier_fallback) or classifier_correct(record)


def correct_source(record: dict, classifier_fallback: bool = False) -> str:
    """Which model(s) got this image right: both, mobileclip_only,
    classifier_only, or neither."""
    mc, cl = mobileclip_correct(record, classifier_fallback), classifier_correct(record)
    if mc and cl:
        return "both"
    if mc:
        return "mobileclip_only"
    if cl:
        return "classifier_only"
    return "neither"


def compute_mobileclip_metrics(records: list, classifier_fallback: bool = False) -> dict:
    """Top-1 accuracy, Top-2 accuracy, average inference time."""
    n = len(records)
    top1_hits = sum(r["mobileclip_top1"] == r["mobileclip_gt"] for r in records)
    top2_hits = sum(mobileclip_correct(r, classifier_fallback) for r in records)

    metrics = {
        "top1_accuracy": top1_hits / n if n else 0.0,
        "top2_accuracy": top2_hits / n if n else 0.0,
        "num_images": n,
    }

    inference_times = [
        r["mobileclip_ms"] for r in records if r.get("mobileclip_ms") is not None
    ]
    if inference_times:
        metrics["average_inference_ms"] = float(np.mean(inference_times))

    return metrics


def compute_hybrid_metrics(records: list, classifier_fallback: bool = False) -> dict:
    """Hybrid accuracy (OR rule), hybrid balanced accuracy,
    average end-to-end pipeline time, YOLO detection rate.
    """
    n = len(records)
    hybrid_flags = [is_hybrid_correct(r, classifier_fallback) for r in records]
    sources = [correct_source(r, classifier_fallback) for r in records]

    by_class = defaultdict(list)
    for record, flag in zip(records, hybrid_flags):
        by_class[record["classifier_gt"]].append(flag)

    per_class_accuracy = [float(np.mean(flags)) for flags in by_class.values()]

    metrics = {
        "hybrid_accuracy": float(np.mean(hybrid_flags)) if n else 0.0,
        "hybrid_balanced_accuracy": (
            float(np.mean(per_class_accuracy)) if per_class_accuracy else 0.0
        ),
        "num_images": n,
        "correct_source_breakdown": {
            source: sources.count(source)
            for source in ("both", "mobileclip_only", "classifier_only", "neither")
        },
    }

    pipeline_times = [
        r["pipeline_ms"] for r in records if r.get("pipeline_ms") is not None
    ]
    if pipeline_times:
        metrics["average_pipeline_ms"] = float(np.mean(pipeline_times))

    yolo_flags = [
        r["yolo_detected"] for r in records if r.get("yolo_detected") is not None
    ]
    if yolo_flags:
        metrics["yolo_detection_rate"] = float(np.mean(yolo_flags))

    return metrics


def build_metrics_report(
    records: list,
    classifier_y_true=None,
    classifier_y_pred=None,
    classifier_fallback: bool = False,
) -> dict:
    report = {
        "mobileclip": compute_mobileclip_metrics(records, classifier_fallback),
        "hybrid": compute_hybrid_metrics(records, classifier_fallback),
    }

    if classifier_y_true is not None and classifier_y_pred is not None:
        report["classifier"] = compute_classification_metrics(
            classifier_y_true, classifier_y_pred
        )

    return report
