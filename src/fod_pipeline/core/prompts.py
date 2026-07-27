"""
Prompt configuration utilities.

This module provides a helper function for loading the category names and text
prompt templates used by MobileCLIP during zero-shot classification.

By default, the prompt configuration is loaded from the project's bundled
configuration file (``fod_pipeline/data/mobileclip_prompts.json``). A custom
configuration file may also be supplied to override the default categories or
prompt templates without modifying the source code.

Responsibilities:
    - Load the default MobileCLIP prompt configuration.
    - Support custom prompt configuration files.
    - Return a dictionary containing categories and templates.

Expected configuration format:

    {
        "categories": [...],
        "templates": [...]
    }
"""
from __future__ import annotations

import json
from importlib import resources


def load_prompt_config(path: str | None = None) -> dict:
    if path is not None:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    resource = resources.files("fod_pipeline.data").joinpath("mobileclip_prompts.json")
    with resource.open(encoding="utf-8") as f:
        return json.load(f)
