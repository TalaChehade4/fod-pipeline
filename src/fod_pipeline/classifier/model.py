"""Stage 4 classifier architecture.

Only MLP2 ships to production. Linear and MLP1 were evaluated during
development (see the archived comparison in TestingClassifier/) but MLP2
outperformed both, so it is the only architecture trained and served here.
"""
from __future__ import annotations

from torch import nn


class MLP2Classifier(nn.Module):
    def __init__(self, input_dim: int = 512, num_classes: int = 59, dropout: float = 0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)
