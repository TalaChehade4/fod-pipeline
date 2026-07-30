"""
Export the trained YOLO detector to ONNX (Android, via ONNX Runtime
Mobile) and Core ML (iOS).

NMS note: YOLOv5's exporter only bakes NMS into the CoreML graph (via
``--nms``, which wraps the model before tracing); its ONNX export has no
NMS parameter at all. So the iOS CoreML model returns final detections,
but the Android ONNX model returns raw predictions - the app must run NMS
itself for Android.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def _yolov5_repo_dir(weights_path: str) -> str:
    import torch

    repo_dir = os.path.join(torch.hub.get_dir(), "ultralytics_yolov5_master")
    if not os.path.isdir(repo_dir):
        torch.hub.load(
            "ultralytics/yolov5", "custom", path=weights_path, autoshape=False, trust_repo=True
        )
    return repo_dir


def _run_yolov5_export(weights_path: str, formats: list, imgsz: int) -> dict:
    weights_path = os.path.abspath(weights_path)
    repo_dir = _yolov5_repo_dir(weights_path)

    cmd = [
        sys.executable,
        os.path.join(repo_dir, "export.py"),
        "--weights", weights_path,
        "--imgsz", str(imgsz),
        "--device", "cpu",
        "--include", *formats,
        "--nms",
    ]
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"YOLOv5 export failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
        )

    stem = os.path.splitext(weights_path)[0]
    exported = {}
    if "onnx" in formats:
        exported["onnx"] = f"{stem}.onnx"
    if "coreml" in formats:
        exported["coreml"] = f"{stem}.mlpackage"
    return exported


def export_yolo(weights_path: str, output_dir: str, formats: list, imgsz: int = 640) -> dict:
    android_dir = os.path.join(output_dir, "android")
    ios_dir = os.path.join(output_dir, "ios")
    os.makedirs(android_dir, exist_ok=True)
    os.makedirs(ios_dir, exist_ok=True)

    exported = _run_yolov5_export(weights_path, formats, imgsz)

    outputs = {}

    if "onnx" in exported:
        dest = os.path.join(android_dir, "yolo_fod.onnx")
        shutil.copyfile(exported["onnx"], dest)
        outputs["onnx"] = dest

    if "coreml" in exported:
        dest = os.path.join(ios_dir, "YoloFOD.mlpackage")
        src = exported["coreml"]
        if os.path.isdir(src):
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            shutil.copyfile(src, dest)
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
