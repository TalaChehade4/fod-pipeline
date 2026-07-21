"""Launch the Stage 4 classifier training job (MLP2 only).

Replaces TrainingClassifier/launch_training.py. The old --architecture
switch (linear/mlp1/mlp2) is gone - MLP2 is the only architecture that
ships to production.
"""
from __future__ import annotations

import argparse

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

from fod_pipeline.config import get_config, require_sagemaker_config
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


if __name__ == "__main__":
    main()
