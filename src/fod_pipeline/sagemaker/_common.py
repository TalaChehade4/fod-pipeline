"""Helper functions for preparing project files for SageMaker jobs.

This module finds the important project paths that SageMaker needs when
running a training or processing job.

It provides the path to the `fod_pipeline` source code and the
`requirements.txt` file so SageMaker can copy the code and install the
required libraries inside the job environment.

By using these paths, SageMaker jobs can use the existing project code
without needing a custom Docker image or manual file copying.
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
