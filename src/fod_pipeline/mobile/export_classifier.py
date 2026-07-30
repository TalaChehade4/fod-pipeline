"""
Export the trained MLP classifier to ONNX (Android, via ONNX Runtime
Mobile) and Core ML (iOS).

Input:  a 512-dim MobileCLIP image embedding (float32, shape (1, 512)).
Output: per-class probabilities (float32, shape (1, num_classes)) -
        softmax is baked into the export
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from fod_pipeline.classifier.dataset import load_label_names
from fod_pipeline.classifier.model import MLP2Classifier
from fod_pipeline.mobile.convert_utils import export_onnx, torch_to_coreml
from fod_pipeline.mobile.wrappers import ClassifierWithSoftmax

INPUT_NAME = "embedding"
OUTPUT_NAME = "class_probabilities"


def export_classifier(
    weights_path: str,
    label_encoder_path: str,
    output_dir: str,
    formats: list,
) -> dict:
    class_names = load_label_names(label_encoder_path)

    classifier = MLP2Classifier(num_classes=len(class_names))
    classifier.load_state_dict(torch.load(weights_path, map_location="cpu"))
    classifier.eval()

    export_model = ClassifierWithSoftmax(classifier)
    dummy_input = torch.randn(1, 512, dtype=torch.float32)

    android_dir = os.path.join(output_dir, "android")
    ios_dir = os.path.join(output_dir, "ios")
    os.makedirs(android_dir, exist_ok=True)
    os.makedirs(ios_dir, exist_ok=True)

    outputs = {}

    if "onnx" in formats:
        onnx_path = os.path.join(android_dir, "fod_classifier.onnx")
        export_onnx(
            export_model,
            dummy_input,
            onnx_path,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
        )
        outputs["onnx"] = onnx_path

    if "coreml" in formats:
        coreml_path = os.path.join(ios_dir, "FODClassifier.mlpackage")
        outputs["coreml"] = torch_to_coreml(
            export_model,
            dummy_input,
            coreml_path,
            input_name=INPUT_NAME,
            output_name=OUTPUT_NAME,
        )

    # class_names is required at inference time to turn a predicted index
    # back into a label - ship it alongside the models on both platforms.
    label_names_path_android = os.path.join(android_dir, "classifier_labels.json")
    label_names_path_ios = os.path.join(ios_dir, "classifier_labels.json")
    for path in (label_names_path_android, label_names_path_ios):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(class_names, f, indent=2)

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to model.pth")
    parser.add_argument("--label-encoder", required=True, help="Path to label_encoder.json")
    parser.add_argument("--output-dir", default="mobile_models")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["onnx", "coreml"],
        choices=["onnx", "coreml"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = export_classifier(
        weights_path=args.weights,
        label_encoder_path=args.label_encoder,
        output_dir=args.output_dir,
        formats=args.formats,
    )
    for fmt, path in outputs.items():
        print(f"[{fmt}] {path}")


if __name__ == "__main__":
    main()
