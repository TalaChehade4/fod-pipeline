"""Launch the Stage 4 dataset-prep job: embeddings -> train/val split,
class weights, label encoding.

Replaces Preprocessing2/launch_job2.py.
"""
from __future__ import annotations

import argparse

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch import PyTorchProcessor

from fod_pipeline.config import get_config, require_sagemaker_config
from fod_pipeline.sagemaker._common import FOD_PIPELINE_PACKAGE, package_dependencies


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embeddings-uri", type=str, required=True)
    parser.add_argument("--output-uri", type=str, required=True)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--job-name", type=str, default="fod-dataset-prep")

    return parser.parse_args()


def main():
    args = parse_args()

    config = get_config()
    require_sagemaker_config(config)

    processor = PyTorchProcessor(
        framework_version="2.1",
        py_version="py310",
        role=config.sagemaker_role_arn,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        volume_size_in_gb=50,
        max_runtime_in_seconds=3600,
        base_job_name=args.job_name,
        sagemaker_session=sagemaker.Session(),
    )

    processor.run(
        code=str(FOD_PIPELINE_PACKAGE / "classifier" / "prepare.py"),
        dependencies=package_dependencies(),
        arguments=[
            "--input-dir", "/opt/ml/processing/input",
            "--output-dir", "/opt/ml/processing/output",
            "--val-size", str(args.val_size),
        ],
        inputs=[
            ProcessingInput(source=args.embeddings_uri, destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=args.output_uri),
        ],
        wait=True,
        logs=True,
    )


if __name__ == "__main__":
    main()
