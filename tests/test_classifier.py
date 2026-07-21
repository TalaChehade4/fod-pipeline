import numpy as np
import torch

from fod_pipeline.classifier.dataset import split_dataset
from fod_pipeline.classifier.model import MLP2Classifier


def test_mlp2classifier_output_shape():
    model = MLP2Classifier(input_dim=512, num_classes=10)
    x = torch.randn(4, 512)

    logits = model(x)

    assert logits.shape == (4, 10)


def test_split_dataset_defaults_to_90_10():
    X = np.arange(100).reshape(100, 1).astype("float32")
    y = np.array([i % 5 for i in range(100)])  # 5 balanced classes, stratify-friendly

    X_train, X_val, _y_train, y_val = split_dataset(X, y)

    assert len(X_val) == 10
    assert len(X_train) == 90
    assert len(y_val) == 10
