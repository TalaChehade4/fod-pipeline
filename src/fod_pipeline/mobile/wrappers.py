"""
Thin nn.Module wrappers that pin down exactly what gets exported.

ONNX/CoreML conversion traces whatever `forward()` computes, so these
wrappers exist to guarantee the exported graph reproduces the same math
`fod_pipeline.pipeline.infer.HybridPipeline.predict()` runs today - no more,
no less.
"""
from __future__ import annotations

import torch
from torch import nn


class MobileClipImageEncoderExport(nn.Module):
    """Image tower only, L2-normalized output.

    Equivalent to ``fod_pipeline.core.embedding.encode_image(model, ...,
    normalize=True)``, which is what ``HybridPipeline.predict()`` feeds to
    both the MobileCLIP text-similarity step and the MLP classifier. The
    text tower is intentionally excluded - see the MobileCLIP export
    script for why.
    """

    def __init__(self, mobileclip_model: nn.Module):
        super().__init__()
        self.model = mobileclip_model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(image, normalize=True)


class ClassifierWithSoftmax(nn.Module):
    """Wraps the MLP classifier so mobile apps get class probabilities
    directly instead of raw logits.

    argmax(softmax(x)) == argmax(x), so this changes nothing about which
    class wins - it just saves every mobile app from having to re-implement
    softmax to show a confidence score.
    """

    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.classifier(embedding))
