"""
Wrappers used when exporting models for mobile deployment.

This module adapts the original PyTorch models into a format that is more
convenient for Android and iOS applications. It exposes only the MobileCLIP
image encoder (excluding the text encoder) and wraps the classifier with a
Softmax layer so the exported model returns class probabilities instead of
raw logits.
"""
from __future__ import annotations

import torch
from torch import nn


class MobileClipImageEncoderExport(nn.Module):
    """Image encoder only, L2-normalized output.
    """

    def __init__(self, mobileclip_model: nn.Module):
        super().__init__()
        self.model = mobileclip_model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(image, normalize=True)


class ClassifierWithSoftmax(nn.Module):
    """Wraps the MLP classifier so mobile apps get class probabilities
    directly instead of raw logits.
    """

    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.classifier(embedding))
