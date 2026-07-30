"""
Precomputed MobileCLIP text-embedding bank for on-device zero-shot matching.

MobileCLIP classifies an image by comparing its embedding against text
embeddings of fixed prompts. Since the category/template list is fixed at build time, the
text tower never needs to run on the phone: we run it once here and ship
the resulting embeddings as a flat binary blob the app loads at startup.
"""
from __future__ import annotations

import json
import os

import torch

from fod_pipeline.core.embedding import build_text_features


def export_text_bank(
    mobileclip_model,
    tokenizer,
    categories: list,
    templates: list,
    device: torch.device,
    output_dir: str,
    basename: str = "mobileclip_text_bank",
) -> dict:
    """Compute and save the (categories x templates x dim) text-embedding bank.

    Writes ``{basename}.bin`` (raw float32 tensor) and
    ``{basename}.json`` (categories, templates, tensor shape - needed to
    interpret the raw floats on-device).

    Returns the metadata dict that was written to the .json file.
    """
    text_features = build_text_features(
        mobileclip_model, tokenizer, categories, templates, device, fp16=False
    )  # shape: (num_categories, num_templates, dim)

    os.makedirs(output_dir, exist_ok=True)

    bin_path = os.path.join(output_dir, f"{basename}.bin")
    text_features.detach().cpu().float().numpy().tofile(bin_path)

    metadata = {
        "categories": list(categories),
        "templates": list(templates),
        "shape": list(text_features.shape),
        "dtype": "float32",
        "layout": "row-major, shape = [num_categories, num_templates, embedding_dim]",
    }
    json_path = os.path.join(output_dir, f"{basename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata
