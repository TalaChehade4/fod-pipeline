"""
YOLO detection and cropping utilities.

This module contains helper functions for loading the YOLOv5 object detection
model, extracting model weights from SageMaker model archives, and cropping
detected objects from images.

Responsibilities:
    - Load a trained YOLOv5 model for inference.
    - Extract ``best.pt`` from a SageMaker ``model.tar.gz`` archive.
    - Select the highest-confidence detection.
    - Expand the detected bounding box by a configurable margin.
    - Return the cropped object image and its bounding box coordinates.

The cropped image produced by this module is passed to the embedding stage,
where MobileCLIP generates a feature vector for classification.

Typical pipeline:

    Image
      ↓
    YOLOv5 Detection
      ↓
    Highest-confidence bounding box
      ↓
    Expanded crop (default: 20% padding)
      ↓
    Cropped image → MobileCLIP embedding

This module performs inference only; it does not train YOLO models or manage
datasets.
"""
from __future__ import annotations

import os
import tarfile

import torch
from PIL import Image

from fod_pipeline.core.device import get_device

DEFAULT_EXPANSION = 0.2


def load_yolo(model_path: str, device: torch.device | None = None, fp16: bool = False):
    """Load the customized YOLO model"""
    device = device or get_device()

    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=model_path,
        force_reload=False,
        trust_repo=True,
    )
    model.to(device)
    model.eval()

    if fp16 and device.type == "cuda":
        model.half()

    return model


def extract_yolo_weights(
    yolo_tar: str, extract_dir: str = "/opt/ml/processing/input/yolo"
) -> str:
    """Extract a SageMaker model.tar.gz and return the path to best.pt inside it."""
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(yolo_tar) as tar:
        tar.extractall(extract_dir)

    for root, _dirs, files in os.walk(extract_dir):
        if "best.pt" in files:
            return os.path.join(root, "best.pt")

    raise FileNotFoundError(f"best.pt not found inside {yolo_tar}")


def crop_yolo_detection(
    image: Image.Image, results, expansion: float = DEFAULT_EXPANSION
):
    """Crop the highest-confidence detection, expanded by `expansion` on each side.

    Returns (crop, bbox) where bbox is (xmin, ymin, xmax, ymax), or
    (image, None) unchanged if YOLO found nothing.
    """
    detections = results.pandas().xyxy[0]

    if len(detections) == 0:
        return image, None

    det = detections.sort_values("confidence", ascending=False).iloc[0]

    xmin, ymin, xmax, ymax = det["xmin"], det["ymin"], det["xmax"], det["ymax"]

    width = xmax - xmin
    height = ymax - ymin

    pad_x = width * expansion
    pad_y = height * expansion

    xmin = max(0, int(xmin - pad_x))
    ymin = max(0, int(ymin - pad_y))
    xmax = min(image.width, int(xmax + pad_x))
    ymax = min(image.height, int(ymax + pad_y))

    crop = image.crop((xmin, ymin, xmax, ymax))

    return crop, (xmin, ymin, xmax, ymax)
