import pytest

from fod_pipeline.hybrid.metrics import (
    build_metrics_report,
    compute_hybrid_metrics,
    compute_mobileclip_metrics,
    correct_source,
    is_hybrid_correct,
)


def _record(mobileclip_gt, top1, top2, classifier_gt, classifier_pred):
    return {
        "mobileclip_gt": mobileclip_gt,
        "mobileclip_top1": top1,
        "mobileclip_top2": top2,
        "classifier_gt": classifier_gt,
        "classifier_pred": classifier_pred,
    }


def test_hybrid_or_rule_matches_architecture_doc_examples():
    # Example 1 (doc section 4.2): MobileCLIP top1 correct, classifier correct -> Correct
    r1 = _record("Screw", "Screw", "Bolt", "Fastener", "Fastener")
    # Example 2: MobileCLIP top1 wrong but top2 correct, classifier correct -> Correct
    r2 = _record("Screw", "Bolt", "Screw", "Fastener", "Fastener")
    # Example 3: MobileCLIP both wrong, classifier wrong -> Incorrect
    r3 = _record("Screw", "Bolt", "Bolt", "Fastener", "Nut")

    assert is_hybrid_correct(r1) is True
    assert is_hybrid_correct(r2) is True
    assert is_hybrid_correct(r3) is False


def test_correct_source_breakdown():
    both = _record("Screw", "Screw", "Bolt", "Fastener", "Fastener")
    mobileclip_only = _record("Screw", "Screw", "Bolt", "Fastener", "Nut")
    classifier_only = _record("Screw", "Bolt", "Nut", "Fastener", "Fastener")
    neither = _record("Screw", "Bolt", "Nut", "Fastener", "Nut")

    assert correct_source(both) == "both"
    assert correct_source(mobileclip_only) == "mobileclip_only"
    assert correct_source(classifier_only) == "classifier_only"
    assert correct_source(neither) == "neither"


def test_compute_mobileclip_metrics_top1_top2_accuracy():
    records = [
        _record("Screw", "Screw", "Bolt", "Fastener", "Fastener"),  # top1 correct
        _record("Screw", "Bolt", "Screw", "Fastener", "Fastener"),  # top2 only
        _record("Screw", "Bolt", "Nut", "Fastener", "Fastener"),  # neither
    ]

    metrics = compute_mobileclip_metrics(records)

    assert metrics["top1_accuracy"] == pytest.approx(1 / 3)
    assert metrics["top2_accuracy"] == pytest.approx(2 / 3)


def test_compute_hybrid_metrics_accuracy_and_balanced_accuracy():
    records = [
        _record("Screw", "Screw", "Bolt", "Fastener", "Fastener"),  # correct, Fastener
        _record("Screw", "Bolt", "Nut", "Fastener", "Nut"),  # incorrect, Fastener
        _record("Nail", "Nail", "Bolt", "Fastener", "Fastener"),  # correct, Fastener
        _record("Wire", "Wire", "Bolt", "Wire", "Wire"),  # correct, Wire
    ]

    metrics = compute_hybrid_metrics(records)

    assert metrics["hybrid_accuracy"] == pytest.approx(3 / 4)
    # Fastener class: 2/3 correct, Wire class: 1/1 correct -> macro average
    assert metrics["hybrid_balanced_accuracy"] == pytest.approx((2 / 3 + 1) / 2)
    assert metrics["correct_source_breakdown"]["neither"] == 1


def test_build_metrics_report_includes_classifier_section_when_arrays_given():
    records = [_record("Screw", "Screw", "Bolt", "Fastener", "Fastener")]

    report = build_metrics_report(records, classifier_y_true=[0], classifier_y_pred=[0])

    assert "classifier" in report
    assert report["classifier"]["accuracy"] == 1.0


def test_build_metrics_report_omits_classifier_section_when_arrays_missing():
    records = [_record("Screw", "Screw", "Bolt", "Fastener", "Fastener")]

    report = build_metrics_report(records)

    assert "classifier" not in report
