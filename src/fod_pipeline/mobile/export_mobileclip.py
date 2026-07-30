"""
Export MobileCLIP's image encoder to ONNX (Android, via ONNX Runtime
Mobile) and Core ML (iOS), and precompute its text-prompt embedding bank
for on-device zero-shot matching.

"""
from __future__ import annotations

import argparse
import os

import torch

from fod_pipeline.core.embedding import load_mobileclip
from fod_pipeline.core.prompts import load_prompt_config
from fod_pipeline.mobile.convert_utils import export_onnx, torch_to_coreml
from fod_pipeline.mobile.text_bank import export_text_bank
from fod_pipeline.mobile.wrappers import MobileClipImageEncoderExport

INPUT_NAME = "image"
OUTPUT_NAME = "image_embedding"
IMAGE_SIZE = 256


def export_mobileclip(
    weights_path: str,
    output_dir: str,
    formats: list,
    model_name: str = "mobileclip_s0",
    prompts_path: str | None = None,
) -> dict:
    device = torch.device("cpu")  # export runs once, offline - CPU is enough
    model, _preprocess, tokenizer, device = load_mobileclip(
        weights_path, model_name=model_name, device=device
    )

    export_model = MobileClipImageEncoderExport(model)
    dummy_input = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)

    android_dir = os.path.join(output_dir, "android")
    ios_dir = os.path.join(output_dir, "ios")
    os.makedirs(android_dir, exist_ok=True)
    os.makedirs(ios_dir, exist_ok=True)

    outputs = {}

    if "onnx" in formats:
        onnx_path = os.path.join(android_dir, "mobileclip_image_encoder.onnx")
        export_onnx(
            export_model,
            dummy_input,
            onnx_path,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
        )
        outputs["onnx"] = onnx_path

    if "coreml" in formats:
        coreml_path = os.path.join(ios_dir, "MobileClipImageEncoder.mlpackage")
        outputs["coreml"] = torch_to_coreml(
            export_model,
            dummy_input,
            coreml_path,
            input_name=INPUT_NAME,
            output_name=OUTPUT_NAME,
        )

    prompt_config = load_prompt_config(prompts_path)
    for platform_dir in (android_dir, ios_dir):
        export_text_bank(
            model,
            tokenizer,
            prompt_config["categories"],
            prompt_config["templates"],
            device,
            platform_dir,
        )
    outputs["text_bank"] = [
        os.path.join(android_dir, "mobileclip_text_bank.bin"),
        os.path.join(ios_dir, "mobileclip_text_bank.bin"),
    ]

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to mobileclip_s0.pt")
    parser.add_argument("--model-name", default="mobileclip_s0")
    parser.add_argument(
        "--prompts", default=None, help="Override the bundled category/prompt JSON"
    )
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
    outputs = export_mobileclip(
        weights_path=args.weights,
        output_dir=args.output_dir,
        formats=args.formats,
        model_name=args.model_name,
        prompts_path=args.prompts,
    )
    for fmt, path in outputs.items():
        print(f"[{fmt}] {path}")


if __name__ == "__main__":
    main()
