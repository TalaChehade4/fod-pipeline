"""Shared path resolution for SageMaker launch scripts.

SageMaker's ProcessingInput/Estimator `dependencies=[...]` mechanism copies
each listed path into the job container alongside the entry script. Bundling
the fod_pipeline package directory this way (rather than requiring it to be
pip-installed in a custom container image) lets every launch script use the
stock SageMaker PyTorch images.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
FOD_PIPELINE_PACKAGE = SRC_DIR / "fod_pipeline"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"


def package_dependencies() -> list:
    """Dependencies list bundling this repo's package + pinned requirements."""
    return [str(FOD_PIPELINE_PACKAGE), str(REQUIREMENTS_PATH)]
