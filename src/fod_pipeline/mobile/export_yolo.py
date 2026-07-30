"""
Export the trained YOLO detector to ONNX (Android, via ONNX Runtime
Mobile) and Core ML (iOS).

Unlike the MobileCLIP/classifier exports, this goes through
``ultralytics``'s own ``YOLO.export()`` API rather than the manual
ONNX/coremltools path in `convert_utils`: ultralytics already implements
and maintains first-class ONNX and Core ML exporters.

"""
from __future__ import annotations

import argparse
import os
import shutil


def export_yolo(weights_path: str, output_dir: str, formats: list, imgsz: int = 640) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLO export. Install it with: pip install ultralytics"
        ) from exc

    android_dir = os.path.join(output_dir, "android")
    ios_dir = os.path.join(output_dir, "ios")
    os.makedirs(android_dir, exist_ok=True)
    os.makedirs(ios_dir, exist_ok=True)

    outputs = {}

    if "onnx" in formats:
        model = YOLO(weights_path)
        exported_path = model.export(format="onnx", imgsz=imgsz, nms=True)
        dest = os.path.join(android_dir, "yolo_fod.onnx")
        shutil.copyfile(exported_path, dest)
        outputs["onnx"] = dest

    if "coreml" in formats:
        model = YOLO(weights_path)
        exported_path = model.export(format="coreml", imgsz=imgsz, nms=True)
        dest = os.path.join(ios_dir, "YoloFOD.mlpackage")
        if os.path.isdir(exported_path):
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(exported_path, dest)
        else:
            shutil.copyfile(exported_path, dest)
        outputs["coreml"] = dest

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--output-dir", default="mobile_models")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["onnx", "coreml"],
        choices=["onnx", "coreml"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = export_yolo(
        weights_path=args.weights,
        output_dir=args.output_dir,
        formats=args.formats,
        imgsz=args.imgsz,
    )
    for fmt, path in outputs.items():
        print(f"[{fmt}] {path}")


if __name__ == "__main__":
    main()
