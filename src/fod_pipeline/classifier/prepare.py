"""
Prepare embedding datasets for classifier training.

This script converts the embedding JSON files generated during the MobileCLIP
preprocessing stage into a training-ready dataset for the classifier.

The preparation pipeline performs the following steps:

    1. Load all embedding files.
    2. Encode string labels into integer class indices.
    3. Split the dataset into training and validation sets.
    4. Compute balanced class weights.
    5. Save the processed datasets and label encoder.

The resulting files can be used directly by the classifier training script.
"""
from __future__ import annotations

import argparse

from fod_pipeline.classifier.dataset import (
    DEFAULT_VAL_SIZE,
    compute_weights,
    encode_labels,
    load_embeddings,
    save_prepared_dataset,
    split_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="PreparedData")
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)

    return parser.parse_args()


def main():
    args = parse_args()

    X, y = load_embeddings(args.input_dir)
    print(f"Embedding shape: {X.shape}, classes: {len(set(y))}")

    y_encoded, encoder = encode_labels(y)

    X_train, X_val, y_train, y_val = split_dataset(
        X, y_encoded, val_size=args.val_size
    )
    print(f"Train samples: {len(X_train)}, validation samples: {len(X_val)}")

    weights = compute_weights(y_train)

    save_prepared_dataset(
        args.output_dir, X_train, X_val, y_train, y_val, weights, encoder
    )
    print("Saved processed dataset to", args.output_dir)


if __name__ == "__main__":
    main()
