"""MobileCLIP category list + prompt templates for Stage 2/3 text scoring."""
from __future__ import annotations

import json
from importlib import resources


def load_prompt_config(path: str | None = None) -> dict:
    """Load {"categories": [...], "templates": [...]}.

    Defaults to the bundled project prompt set
    (fod_pipeline/data/mobileclip_prompts.json). Pass a path to override
    with a project-specific category list.
    """
    if path is not None:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    resource = resources.files("fod_pipeline.data").joinpath("mobileclip_prompts.json")
    with resource.open(encoding="utf-8") as f:
        return json.load(f)
