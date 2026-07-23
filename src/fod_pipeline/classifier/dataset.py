"""Classifier data preparation: load embeddings, encode labels, split, class weights.

Default split is 90/10 train/validation
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

DEFAULT_VAL_SIZE = 0.1


class EmbeddingDataset(Dataset):
    def __init__(self, data: dict):
        self.embeddings = data["embeddings"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def load_embeddings(input_dir: str):
    """Read every embedding JSON file under input_dir - each file may hold
    one sample or a list of samples - into (X, y) arrays."""
    embeddings = []
    labels = []

    json_files = glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True)

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)

        samples = data if isinstance(data, list) else [data]

        for item in samples:
            embeddings.append(item["embedding"])
            labels.append(item["label"])

    X = np.array(embeddings, dtype=np.float32)
    y = np.array(labels)

    return X, y


def encode_labels(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def split_dataset(X, y, val_size: float = DEFAULT_VAL_SIZE, random_state: int = 42):
    return train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )


def compute_weights(y_train) -> torch.Tensor:
    weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(train_data: dict, val_data: dict, batch_size: int):
    train_loader = DataLoader(
        EmbeddingDataset(train_data), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        EmbeddingDataset(val_data), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


def save_prepared_dataset(output_dir, X_train, X_val, y_train, y_val, weights, encoder):
    os.makedirs(output_dir, exist_ok=True)

    train_data = {
        "embeddings": torch.tensor(X_train, dtype=torch.float32),
        "labels": torch.tensor(y_train, dtype=torch.long),
    }
    val_data = {
        "embeddings": torch.tensor(X_val, dtype=torch.float32),
        "labels": torch.tensor(y_val, dtype=torch.long),
    }

    torch.save(train_data, os.path.join(output_dir, "train.pt"))
    torch.save(val_data, os.path.join(output_dir, "val.pt"))
    torch.save(weights, os.path.join(output_dir, "class_weights.pt"))

    _write_json_dataset(os.path.join(output_dir, "train.json"), X_train, y_train)
    _write_json_dataset(os.path.join(output_dir, "val.json"), X_val, y_val)

    label_map = {int(i): label for i, label in enumerate(encoder.classes_)}
    with open(os.path.join(output_dir, "label_encoder.json"), "w") as f:
        json.dump(label_map, f, indent=4)

    return train_data, val_data


def _write_json_dataset(path, X, y):
    data = [
        {"embedding": embedding.tolist(), "label": int(label)}
        for embedding, label in zip(X, y)
    ]
    with open(path, "w") as f:
        json.dump(data, f)


def load_prepared_dataset(input_dir: str):
    train_data = torch.load(os.path.join(input_dir, "train.pt"))
    val_data = torch.load(os.path.join(input_dir, "val.pt"))
    class_weights = torch.load(os.path.join(input_dir, "class_weights.pt"))
    return train_data, val_data, class_weights


def load_label_names(label_encoder_path: str) -> list:
    with open(label_encoder_path, "r") as f:
        label_map = json.load(f)
    # JSON keys are strings; class index order must be preserved for metrics/plots.
    return [label_map[str(i)] for i in range(len(label_map))]
