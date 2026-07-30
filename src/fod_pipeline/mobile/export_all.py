"""
Export the full hybrid pipeline (YOLO + MobileCLIP + classifier) to
mobile formats in one step.

Produces:

    mobile_models/
      android/
        yolo_fod.onnx
        mobileclip_image_encoder.onnx
        mobileclip_text_bank.bin
        mobileclip_text_bank.json
        fod_classifier.onnx
        classifier_labels.json
      ios/
        YoloFOD.mlpackage
        MobileClipImageEncoder.mlpackage
        mobileclip_text_bank.bin
        mobileclip_text_bank.json
        FODClassifier.mlpackage
        classifier_labels.json

Usage:
    fod-export-mobile \\
        --yolo best.pt \\
        --mobileclip mobileclip_s0.pt \\
        --classifier-weights model.pth \\
        --label-encoder label_encoder.json \\
        --output-dir mobile_models
"""
from __future__ import annotations

import argparse

from fod_pipeline.mobile.export_classifier import export_classifier
from fod_pipeline.mobile.export_mobileclip import export_mobileclip
from fod_pipeline.mobile.export_yolo import export_yolo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo", required=True, help="Path to best.pt")
    parser.add_argument("--mobileclip", required=True, help="Path to mobileclip_s0.pt")
    parser.add_argument("--mobileclip-model-name", default="mobileclip_s0")
    parser.add_argument(
        "--prompts", default=None, help="Override the bundled category/prompt JSON"
    )
    parser.add_argument("--classifier-weights", required=True, help="Path to model.pth")
    parser.add_argument("--label-encoder", required=True, help="Path to label_encoder.json")
    parser.add_argument("--output-dir", default="mobile_models")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["onnx", "coreml"],
        choices=["onnx", "coreml"],
        help="onnx -> Android (ONNX Runtime Mobile), coreml -> iOS. Pass one to convert for "
        "a single platform.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("== YOLO detector ==")
    for fmt, path in export_yolo(
        args.yolo, args.output_dir, args.formats, imgsz=args.imgsz
    ).items():
        print(f"[{fmt}] {path}")

    print("== MobileCLIP image encoder + text bank ==")
    for fmt, path in export_mobileclip(
        args.mobileclip,
        args.output_dir,
        args.formats,
        model_name=args.mobileclip_model_name,
        prompts_path=args.prompts,
    ).items():
        print(f"[{fmt}] {path}")

    print("== MLP classifier ==")
    for fmt, path in export_classifier(
        args.classifier_weights,
        args.label_encoder,
        args.output_dir,
        args.formats,
    ).items():
        print(f"[{fmt}] {path}")

    print(f"\nDone. Mobile models written under: {args.output_dir}")


if __name__ == "__main__":
    main()
