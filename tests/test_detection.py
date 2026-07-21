from types import SimpleNamespace

import pandas as pd
from PIL import Image

from fod_pipeline.core.detection import crop_yolo_detection


def _fake_results(rows):
    df = pd.DataFrame(rows)
    return SimpleNamespace(pandas=lambda: SimpleNamespace(xyxy=[df]))


def test_crop_yolo_detection_expands_bbox_by_20_percent():
    image = Image.new("RGB", (200, 200))
    results = _fake_results(
        [{"xmin": 50, "ymin": 50, "xmax": 100, "ymax": 100, "confidence": 0.9}]
    )

    crop, bbox = crop_yolo_detection(image, results, expansion=0.2)

    # width/height = 50, pad = 10 on each side
    assert bbox == (40, 40, 110, 110)
    assert crop.size == (70, 70)


def test_crop_yolo_detection_clamps_to_image_bounds():
    image = Image.new("RGB", (100, 100))
    results = _fake_results(
        [{"xmin": 0, "ymin": 0, "xmax": 90, "ymax": 90, "confidence": 0.9}]
    )

    _crop, bbox = crop_yolo_detection(image, results, expansion=0.2)

    assert bbox == (0, 0, 100, 100)


def test_crop_yolo_detection_picks_highest_confidence_detection():
    image = Image.new("RGB", (200, 200))
    results = _fake_results(
        [
            {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10, "confidence": 0.1},
            {"xmin": 50, "ymin": 50, "xmax": 100, "ymax": 100, "confidence": 0.99},
        ]
    )

    _crop, bbox = crop_yolo_detection(image, results, expansion=0.0)

    assert bbox == (50, 50, 100, 100)


def test_crop_yolo_detection_no_detections_returns_original_image():
    image = Image.new("RGB", (100, 100))
    results = _fake_results([])

    crop, bbox = crop_yolo_detection(image, results)

    assert bbox is None
    assert crop is image
