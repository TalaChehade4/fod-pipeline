import json
import os

import numpy as np

from fod_pipeline.mobile.text_bank import export_text_bank


def test_export_text_bank_writes_matching_bin_and_metadata(tmp_path):
    import mobileclip

    import torch

    model, _, _preprocess = mobileclip.create_model_and_transforms(
        "mobileclip_s0", pretrained=None
    )
    # MobileCLIP zero-initializes its final text projection, so an
    # untrained model emits an all-zero (unnormalizable) text embedding.
    # Perturb any all-zero parameter so this test can exercise real
    # normalization without needing to download trained weights.
    for p in model.parameters():
        if torch.count_nonzero(p) == 0:
            torch.nn.init.normal_(p, std=0.02)
    model.eval()
    tokenizer = mobileclip.get_tokenizer("mobileclip_s0")

    categories = ["Allen Key", "Bolt"]
    templates = ["a photo of a {}", "a {} on a runway"]

    metadata = export_text_bank(
        model, tokenizer, categories, templates, "cpu", str(tmp_path)
    )

    assert metadata["categories"] == categories
    assert metadata["templates"] == templates
    assert metadata["shape"] == [len(categories), len(templates), 512]

    bin_path = tmp_path / "mobileclip_text_bank.bin"
    json_path = tmp_path / "mobileclip_text_bank.json"
    assert bin_path.exists()
    assert json_path.exists()

    with open(json_path) as f:
        saved_metadata = json.load(f)
    assert saved_metadata == metadata

    raw = np.fromfile(bin_path, dtype=np.float32)
    assert raw.size == len(categories) * len(templates) * 512

    # Each (category, template) row should be L2-normalized, matching
    # build_text_features' own normalization.
    rows = raw.reshape(len(categories), len(templates), 512)
    norms = np.linalg.norm(rows, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)
