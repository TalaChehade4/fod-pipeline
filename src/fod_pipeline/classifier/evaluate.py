"""Stage 4 classifier evaluation, shared by post-training validation and
standalone test-set evaluation (Section 5.2 metrics).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        outputs = model(embeddings)
        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_predictions), np.array(all_labels)


@torch.no_grad()
def predict_with_latency(model, embeddings: np.ndarray, device, batch_size: int = 256):
    """Batched inference with per-image latency measurement, for the
    throughput/latency figures reported during standalone test evaluation.
    """
    model.eval()

    predictions = []
    latencies = []

    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    total = len(embeddings)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = embeddings[start:end].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model(batch)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        batch_latency_ms = (t1 - t0) * 1000
        latencies.extend([batch_latency_ms / len(batch)] * len(batch))

        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())

    return np.array(predictions), np.array(latencies)


def compute_classification_metrics(y_true, y_pred) -> dict:
    """Top-1 accuracy, balanced accuracy, precision/recall/F1 (macro + weighted)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "num_samples": int(len(y_true)),
    }


def latency_summary(latencies: np.ndarray) -> dict:
    return {
        "average_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "throughput_images_per_sec": float(1000 / np.mean(latencies)),
    }


def classification_report_dict(y_true, y_pred, class_names) -> dict:
    return classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )


def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def save_confusion_matrix_plot(cm, class_names, output_path, title="Confusion Matrix"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(18, 16))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=8)
    plt.yticks(range(len(class_names)), class_names, fontsize=8)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
