"""Stage 4 classifier training (MLP2 only).

Input: prepared embeddings (train.pt/val.pt/class_weights.pt/label_encoder.json,
produced by fod_pipeline.classifier.dataset). Output: trained model weights +
validation metrics.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch import nn

from fod_pipeline.classifier.dataset import (
    create_dataloaders,
    load_label_names,
    load_prepared_dataset,
)
from fod_pipeline.classifier.evaluate import (
    classification_report_dict,
    compute_classification_metrics,
    compute_confusion_matrix,
    get_predictions,
    save_confusion_matrix_plot,
)
from fod_pipeline.classifier.model import MLP2Classifier
from fod_pipeline.core.device import get_device

DEFAULT_PATIENCE = 10


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)

    parser.add_argument(
        "--train-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "PreparedData"),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "FinalModel"),
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default=None,
        help="Defaults to <train-dir>/label_encoder.json",
    )

    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for embeddings, labels in loader:
        embeddings, labels = embeddings.to(device), labels.to(device)

        outputs = model(embeddings)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for embeddings, labels in loader:
        embeddings, labels = embeddings.to(device), labels.to(device)

        outputs = model(embeddings)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def train_model(
    model, train_loader, val_loader, criterion, optimizer, device, epochs, model_dir, patience
):
    os.makedirs(model_dir, exist_ok=True)

    best_val_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2%} val_loss={val_loss:.4f} val_acc={val_acc:.2%}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
            print("Saved best model")
        else:
            patience_counter += 1
            print(f"No improvement {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    with open(os.path.join(model_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    return best_epoch, best_val_accuracy


def evaluate_and_save(model, val_loader, device, model_dir, class_names):
    predictions, labels = get_predictions(model, val_loader, device)

    metrics = compute_classification_metrics(labels, predictions)
    report = classification_report_dict(labels, predictions, class_names)
    cm = compute_confusion_matrix(labels, predictions)

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(model_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    save_confusion_matrix_plot(
        cm,
        class_names,
        os.path.join(model_dir, "confusion_matrix.png"),
        title="MLP2 Confusion Matrix",
    )

    return metrics


def main():
    args = parse_args()

    label_map_path = args.label_map or os.path.join(args.train_dir, "label_encoder.json")
    class_names = load_label_names(label_map_path)

    train_data, val_data, class_weights = load_prepared_dataset(args.train_dir)
    train_loader, val_loader = create_dataloaders(train_data, val_data, args.batch_size)

    device = get_device()
    print("Using device:", device)

    model = MLP2Classifier(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=0.1
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_epoch, best_accuracy = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        args.epochs,
        args.model_dir,
        args.patience,
    )

    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pth")))
    evaluate_and_save(model, val_loader, device, args.model_dir, class_names)

    print("Best epoch:", best_epoch)
    print("Best validation accuracy:", best_accuracy)


if __name__ == "__main__":
    main()
