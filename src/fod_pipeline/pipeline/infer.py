"""
Hybrid FOD inference pipeline combining object detection,
vision-language classification, and learned feature classification.

Pipeline flow:
    1. YOLO detects the FOD object and extracts a cropped region.
    2. MobileCLIP converts the crop into a visual embedding and performs
       zero-shot classification using text prompts.
    3. The same MobileCLIP image embedding is passed to a trained MLP
       classifier for supervised prediction.
    4. The final prediction exposes both MobileCLIP candidates and the
       classifier output for hybrid evaluation.

The pipeline loads all models once during initialization and provides
a lightweight `predict()` method for per-image inference.

Supported features:
    - GPU/CPU execution
    - FP16 inference on CUDA devices
    - configurable MobileCLIP prompts
    - synonym mapping between MobileCLIP vocabulary and dataset labels
    - latency measurement for each pipeline stage
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import torch
from PIL import Image

from fod_pipeline.classifier.dataset import load_label_names
from fod_pipeline.classifier.model import MLP2Classifier
from fod_pipeline.core.detection import DEFAULT_EXPANSION, crop_yolo_detection, load_yolo
from fod_pipeline.core.device import get_device
from fod_pipeline.core.embedding import (
    build_text_features,
    encode_image,
    load_mobileclip,
    score_text_prompts,
    topk_predictions,
)
from fod_pipeline.core.labels import canonical_label, load_synonym_mapping
from fod_pipeline.core.prompts import load_prompt_config


@dataclass
class HybridPrediction:
    mobileclip_top1: str
    mobileclip_top2: str
    classifier_prediction: str
    yolo_detected: bool
    mobileclip_ms: float
    classifier_ms: float
    pipeline_ms: float

    @property
    def candidates(self) -> list:
        """The 3-candidate hybrid prediction set (Stage 5)."""
        return [self.mobileclip_top1, self.mobileclip_top2, self.classifier_prediction]


class HybridPipeline:
    """Loads YOLO, MobileCLIP, and the classifier once; call .predict(image) per image."""

    def __init__(
        self,
        yolo_model,
        mobileclip_model,
        mobileclip_preprocess,
        mobileclip_text_features,
        mobileclip_categories,
        classifier_model,
        classifier_class_names,
        device,
        crop_expansion: float = DEFAULT_EXPANSION,
        mobileclip_synonym_mapping: dict | None = None,
        fp16: bool = False,
    ):
        self.yolo_model = yolo_model
        self.mobileclip_model = mobileclip_model
        self.mobileclip_preprocess = mobileclip_preprocess
        self.mobileclip_text_features = mobileclip_text_features
        self.mobileclip_categories = mobileclip_categories
        self.classifier_model = classifier_model
        self.classifier_class_names = classifier_class_names
        self.device = device
        self.crop_expansion = crop_expansion
        self.fp16 = fp16
        # MobileCLIP's own category vocabulary (e.g. "Bolt") often differs
        # from the dataset's ground-truth vocabulary (e.g. "Bolts") - this
        # maps predictions into the ground-truth vocabulary so comparisons
        # against Ground Truth A are meaningful. Empty mapping still
        # normalizes case/whitespace via canonical_label.
        self.mobileclip_synonym_mapping = mobileclip_synonym_mapping or {}

    def predict(self, image: Image.Image) -> HybridPrediction:
        pipeline_start = time.perf_counter()

        yolo_results = self.yolo_model(image)
        crop, bbox = crop_yolo_detection(image, yolo_results, expansion=self.crop_expansion)
        yolo_detected = bbox is not None

        mobileclip_start = time.perf_counter()
        image_features = encode_image(
            self.mobileclip_model,
            self.mobileclip_preprocess,
            self.device,
            crop,
            fp16=self.fp16,
            normalize=True,
        )
        similarity = score_text_prompts(image_features, self.mobileclip_text_features)
        (top1, top2), _scores = topk_predictions(
            similarity, self.mobileclip_categories, k=2
        )
        top1 = canonical_label(top1, self.mobileclip_synonym_mapping)
        top2 = canonical_label(top2, self.mobileclip_synonym_mapping)
        mobileclip_ms = (time.perf_counter() - mobileclip_start) * 1000

        classifier_start = time.perf_counter()
        with torch.no_grad():
            logits = self.classifier_model(image_features.float())
            predicted_idx = torch.argmax(logits, dim=1).item()
        classifier_prediction = self.classifier_class_names[predicted_idx]
        classifier_ms = (time.perf_counter() - classifier_start) * 1000

        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000

        return HybridPrediction(
            mobileclip_top1=top1,
            mobileclip_top2=top2,
            classifier_prediction=classifier_prediction,
            yolo_detected=yolo_detected,
            mobileclip_ms=mobileclip_ms,
            classifier_ms=classifier_ms,
            pipeline_ms=pipeline_ms,
        )


def build_pipeline(
    yolo_path: str,
    mobileclip_path: str,
    classifier_weights_path: str,
    label_encoder_path: str,
    mobileclip_model_name: str = "mobileclip_s0",
    prompts_path: str | None = None,
    mobileclip_mapping_path: str | None = None,
    fp16: bool = False,
) -> HybridPipeline:
    device = get_device()

    yolo_model = load_yolo(yolo_path, device=device, fp16=fp16)
    mobileclip_model, preprocess, tokenizer, device = load_mobileclip(
        mobileclip_path, model_name=mobileclip_model_name, device=device, fp16=fp16
    )

    prompt_config = load_prompt_config(prompts_path)
    categories = prompt_config["categories"]
    templates = prompt_config["templates"]
    text_features = build_text_features(
        mobileclip_model, tokenizer, categories, templates, device, fp16=fp16
    )

    class_names = load_label_names(label_encoder_path)
    classifier_model = MLP2Classifier(num_classes=len(class_names)).to(device)
    classifier_model.load_state_dict(
        torch.load(classifier_weights_path, map_location=device)
    )
    classifier_model.eval()

    mobileclip_synonym_mapping = (
        load_synonym_mapping(mobileclip_mapping_path) if mobileclip_mapping_path else {}
    )

    return HybridPipeline(
        yolo_model=yolo_model,
        mobileclip_model=mobileclip_model,
        mobileclip_preprocess=preprocess,
        mobileclip_text_features=text_features,
        mobileclip_categories=categories,
        classifier_model=classifier_model,
        classifier_class_names=class_names,
        device=device,
        mobileclip_synonym_mapping=mobileclip_synonym_mapping,
        fp16=fp16,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("image", type=str, help="Path to a local image file")
    parser.add_argument("--yolo", type=str, required=True)
    parser.add_argument("--mobileclip", type=str, required=True)
    parser.add_argument("--mobileclip-model-name", type=str, default="mobileclip_s0")
    parser.add_argument(
        "--prompts", type=str, default=None, help="Override the bundled category/prompt JSON"
    )
    parser.add_argument(
        "--mobileclip-mapping",
        type=str,
        default=None,
        help="category_to_objects synonym map (e.g. mobileclip_category_mapping_new.json) "
        "translating MobileCLIP's category vocabulary into the dataset's ground-truth vocabulary",
    )
    parser.add_argument("--classifier-weights", type=str, required=True)
    parser.add_argument("--label-encoder", type=str, required=True)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run YOLO/MobileCLIP in fp16 on GPU for faster inference (no effect on CPU)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = build_pipeline(
        yolo_path=args.yolo,
        mobileclip_path=args.mobileclip,
        classifier_weights_path=args.classifier_weights,
        label_encoder_path=args.label_encoder,
        mobileclip_model_name=args.mobileclip_model_name,
        prompts_path=args.prompts,
        mobileclip_mapping_path=args.mobileclip_mapping,
        fp16=args.fp16,
    )

    image = Image.open(args.image).convert("RGB")
    prediction = pipeline.predict(image)

    print(
        json.dumps(
            {
                "mobileclip_top1": prediction.mobileclip_top1,
                "mobileclip_top2": prediction.mobileclip_top2,
                "classifier_prediction": prediction.classifier_prediction,
                "hybrid_candidates": prediction.candidates,
                "yolo_detected": prediction.yolo_detected,
                "mobileclip_ms": prediction.mobileclip_ms,
                "classifier_ms": prediction.classifier_ms,
                "pipeline_ms": prediction.pipeline_ms,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
