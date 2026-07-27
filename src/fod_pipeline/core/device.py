"""
Device selection utilities.

This module provides a single helper function for determining the appropriate
PyTorch device for model execution.

The function automatically selects a CUDA-enabled GPU when one is available;
otherwise, it falls back to the CPU. By centralizing device selection in one
place, the pipeline maintains consistent behavior across all components,
including YOLO detection, MobileCLIP embedding, and classifier inference.
"""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
