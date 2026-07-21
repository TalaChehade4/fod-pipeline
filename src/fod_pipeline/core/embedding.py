"""Stage 2/3 - MobileCLIP feature extraction and text-prompt similarity.

MobileCLIP is run once per image: the resulting 512-D embedding is reused
both for Top-K similarity scoring against class prompts (Stage 3) and as
the classifier's input features (Stage 4).
"""
from __future__ import annotations

from typing import Sequence

import torch
import mobileclip

from fod_pipeline.core.device import get_device


def load_mobileclip(
    weights_path: str,
    model_name: str = "mobileclip_s0",
    device: torch.device | None = None,
    fp16: bool = False,
):
    """Load the MobileCLIP model, its image preprocessing transform, and tokenizer.

    fp16 halves the model for faster GPU inference. LayerNorm modules are
    kept in float32 - MobileCLIP's text-encoder LayerNorm needs float32
    internally, and F.layer_norm otherwise raises "expected scalar type
    Float but found Half" mid-forward-pass.
    """
    device = device or get_device()

    model, _, preprocess = mobileclip.create_model_and_transforms(
        model_name,
        pretrained=weights_path,
        device=device,
    )
    model.eval()

    if fp16 and device.type == "cuda":
        model = model.half()
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.float()

    tokenizer = mobileclip.get_tokenizer(model_name)

    return model, preprocess, tokenizer, device


@torch.inference_mode()
def encode_image(
    model, preprocess, device, image, fp16: bool = False, normalize: bool = False
) -> torch.Tensor:
    """Encode a single PIL image to its MobileCLIP feature vector, shape (1, 512)."""
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    if fp16 and device.type == "cuda":
        image_tensor = image_tensor.half()

    features = model.encode_image(image_tensor)

    if normalize:
        features = features / features.norm(dim=-1, keepdim=True)

    return features


def embedding_to_list(features: torch.Tensor) -> list:
    """Convert a (1, D) or (D,) feature tensor to a JSON-serializable list."""
    if features.dim() > 1:
        features = features.squeeze(0)
    return features.detach().cpu().float().numpy().tolist()


def build_text_features(
    model,
    tokenizer,
    categories: Sequence[str],
    templates: Sequence[str],
    device: torch.device,
    fp16: bool = False,
) -> torch.Tensor:
    """Encode every (category, template) prompt pair.

    Returns shape (num_categories, num_templates, D), L2-normalized per prompt.
    """
    text_features = []

    with torch.inference_mode():
        for category in categories:
            prompts = [template.format(category.lower()) for template in templates]
            tokens = tokenizer(prompts).to(device)
            embeddings = model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            text_features.append(embeddings)

    text_features = torch.stack(text_features)

    if fp16 and device.type == "cuda":
        text_features = text_features.half()

    return text_features


def score_text_prompts(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Similarity of one image against each category's best-matching prompt.

    text_features: (num_categories, num_templates, D). Only the single
    best-matching template per category is kept - we want the best fit
    per category, not an average across phrasings.
    """
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    similarity = []
    for class_prompts in text_features:
        sims = image_features @ class_prompts.T
        similarity.append(sims.max(dim=-1).values)

    similarity = torch.stack(similarity, dim=1)

    if logit_scale is not None:
        similarity = similarity * logit_scale.exp()

    return similarity


def topk_predictions(similarity: torch.Tensor, categories: Sequence[str], k: int = 2):
    """Top-k category names + scores for a single image's similarity row.

    Use k=2 for the production hybrid prediction set (Stage 3), or a
    larger k during evaluation to report the fuller accuracy breakdown.
    """
    scores, indices = torch.topk(similarity, k=k, dim=1)

    indices = indices[0].cpu().tolist()
    scores = scores[0].float().cpu().tolist()

    predictions = [categories[i] for i in indices]

    return predictions, scores
