"""Launches the SageMaker Training job that trains the MLP classifier.

This script configures a SageMaker PyTorch estimator, trains the classifier
using the prepared embedding dataset stored in S3, saves the resulting model
artifact, and copies it to a fixed S3 location for easy use by later stages
of the pipeline. Training settings can be customized through command-line
arguments, while default paths are loaded from config.py.
"""
from __future__ import annotations

import argparse

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

from fod_pipeline.config import get_config, require_sagemaker_config
from fod_pipeline.core.s3_io import copy_within_s3
from fod_pipeline.sagemaker._common import FOD_PIPELINE_PACKAGE, package_dependencies


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--job-name", type=str, default="fod-classifier-mlp2")

    return parser.parse_args()


def main():
    args = parse_args()

    config = get_config()
    require_sagemaker_config(config)

    estimator = PyTorch(
        entry_point=str(FOD_PIPELINE_PACKAGE / "classifier" / "train.py"),
        dependencies=package_dependencies(),
        role=config.sagemaker_role_arn,
        framework_version="2.1",
        py_version="py310",
        instance_type=config.training_instance_type,
        instance_count=1,
        volume_size=50,
        max_run=3600,
        hyperparameters={
            "epochs": args.epochs,
            "batch-size": args.batch_size,
            "learning-rate": args.learning_rate,
            "weight-decay": args.weight_decay,
        },
        output_path=config.s3.classifier_results,
        base_job_name=args.job_name,
        sagemaker_session=sagemaker.Session(),
    )

    estimator.fit(
        inputs={"train": TrainingInput(s3_data=config.s3.classifier_data)},
        wait=True,
        logs=True,
    )

    fixed_model_uri = config.s3.classifier_models + "model.tar.gz"
    copy_within_s3(estimator.model_data, fixed_model_uri)
    print(f"Model artifact: {estimator.model_data}")
    print(f"Copied to fixed path: {fixed_model_uri}")


if __name__ == "__main__":
    main()
