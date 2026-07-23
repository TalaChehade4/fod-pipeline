"""Stage 5 - hybrid evaluation: dual ground truth, OR-rule, and the three
metric groups.

Each image is evaluated against two ground truths, since MobileCLIP and the
classifier predict over different label spaces:
  - mobileclip_gt / mobileclip_top1 / mobileclip_top2 - MobileCLIP's own taxonomy
  - classifier_gt / classifier_pred                    - the classifier's taxonomy

Hybrid rule: an image is correctly recognized if EITHER model succeeds
against its own ground truth.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from fod_pipeline.classifier.evaluate import compute_classification_metrics
from fod_pipeline.core.labels import normalize_label


def mobileclip_correct(record: dict, classifier_fallback: bool = False) -> bool:
    """Top-1 or Top-2 match against MobileCLIP's own ground truth (Ground Truth A).

    classifier_fallback (experimental, off by default): when neither top-1 nor
    top-2 matches Ground Truth A, also accept a match against Ground Truth B
    (classifier_gt). Ground Truth A was recently split into finer classes
    that used to be joined (e.g. manufactured wood/wood, bullet/bullet
    casings); the classifier's own taxonomy still has some of these joined,
    so this recovers matches that MobileCLIP's fixed prompt vocabulary was
    never going to distinguish in the first place. Classes split on *both*
    sides (rubber chunks vs. tire chunks) stay unresolved by this fallback.
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
    """Section 5.1 - Top-1 accuracy, Top-2 accuracy, average inference time."""
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
    """Section 5.3 - hybrid accuracy (OR rule), hybrid balanced accuracy,
    average end-to-end pipeline time, YOLO detection rate.

    Hybrid balanced accuracy is macro-averaged over the classifier's own
    label taxonomy (classifier_gt) - the coarser, canonical category space
    both models are ultimately being judged against.
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
    """Assemble all three metric groups (5.1/5.2/5.3) into one report.

    classifier_y_true/y_pred are the full encoded-label arrays for the
    classifier's own evaluation (Section 5.2, precision/recall/F1 etc) - if
    omitted, that section is left out of the report.

    classifier_fallback (experimental, off by default) - see
    mobileclip_correct() - widens the Section 5.1/5.3 MobileCLIP matching
    with a second-chance comparison against Ground Truth B.
    """
    report = {
        "mobileclip": compute_mobileclip_metrics(records, classifier_fallback),
        "hybrid": compute_hybrid_metrics(records, classifier_fallback),
    }

    if classifier_y_true is not None and classifier_y_pred is not None:
        report["classifier"] = compute_classification_metrics(
            classifier_y_true, classifier_y_pred
        )

    return report
