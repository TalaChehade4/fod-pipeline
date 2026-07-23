"""Central configuration, loaded from environment variables (see .env.example).

No real AWS account IDs, ARNs, or bucket names should ever be hardcoded in
this repository - everything environment-specific goes through this module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Copy .env.example to .env and fill in your values."
        )
    return value


def _optional(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class S3Paths:
    """S3 locations for pipeline artifacts, all rooted under one bucket/prefix."""

    bucket: str
    project_prefix: str

    def uri(self, *parts: str) -> str:
        path = "/".join([self.project_prefix, *parts])
        return f"s3://{self.bucket}/{path}/"

    def manifest(self, split: str) -> str:
        return self.uri("manifests") + f"{split}_manifest.json"

    def label_map(self, split: str) -> str:
        """Classifier-taxonomy ground truth for a split."""
        return self.uri("manifests") + f"{split}_label_map.json"

    def mobileclip_label_map(self, split: str) -> str:
        """MobileCLIP-taxonomy ground truth (Ground Truth A) for a split."""
        return self.uri("manifests") + f"{split}_mobileclip_label_map.json"

    def database_csv(self, split: str) -> str:
        """The id->name database CSV (trainingdata_old.csv/testingdata_old.csv)."""
        filename = "trainingdata_old.csv" if split == "train" else "testingdata_old.csv"
        return self.uri("manifests") + filename

    @property
    def join_config(self) -> str:
        """Curated object_id->label overrides, shared across splits."""
        return self.uri("manifests") + "join_config.json"

    @property
    def train_embeddings(self) -> str:
        return self.uri("embeddings", "train")

    @property
    def test_embeddings(self) -> str:
        return self.uri("embeddings", "test")

    @property
    def classifier_data(self) -> str:
        return self.uri("classifier-data")

    @property
    def classifier_models(self) -> str:
        return self.uri("classifier-models")

    @property
    def classifier_results(self) -> str:
        return self.uri("classifier-results")

    @property
    def hybrid_results(self) -> str:
        return self.uri("hybrid-results")

    @property
    def yolo_weights_tar(self) -> str:
        return self.uri("weights", "yolo") + "model.tar.gz"

    @property
    def mobileclip_weights(self) -> str:
        return self.uri("weights", "mobileclip") + "mobileclip_s0.pt"


@dataclass(frozen=True)
class Config:
    aws_region: str
    sagemaker_role_arn: str
    s3: S3Paths

    yolo_weights_path: str
    mobileclip_weights_path: str
    mobileclip_model_name: str

    embedding_instance_type: str
    training_instance_type: str
    testing_instance_type: str


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load and cache configuration from the environment. Raises if a
    required SageMaker-related variable is missing - local-only usage
    (inference, training on already-prepared data) does not need those.
    """
    return Config(
        aws_region=_optional("AWS_REGION", "us-east-1"),
        sagemaker_role_arn=os.environ.get("SAGEMAKER_ROLE_ARN", ""),
        s3=S3Paths(
            bucket=os.environ.get("S3_BUCKET", ""),
            project_prefix=_optional("S3_PROJECT_PREFIX", "fod-pipeline"),
        ),
        yolo_weights_path=_optional("YOLO_WEIGHTS_PATH", "weights/yolo/best.pt"),
        mobileclip_weights_path=_optional(
            "MOBILECLIP_WEIGHTS_PATH", "weights/mobileclip/mobileclip_s0.pt"
        ),
        mobileclip_model_name=_optional("MOBILECLIP_MODEL_NAME", "mobileclip_s0"),
        embedding_instance_type=_optional(
            "SAGEMAKER_EMBEDDING_INSTANCE_TYPE", "ml.g5.xlarge"
        ),
        training_instance_type=_optional(
            "SAGEMAKER_TRAINING_INSTANCE_TYPE", "ml.g5.xlarge"
        ),
        testing_instance_type=_optional(
            "SAGEMAKER_TESTING_INSTANCE_TYPE", "ml.g5.xlarge"
        ),
    )


def require_s3_bucket(config: Config) -> None:
    """Call before computing any S3Paths-derived default - fails fast with a
    clear message instead of producing a URI with an empty bucket.
    """
    if not config.s3.bucket:
        raise RuntimeError(
            "S3_BUCKET is not set. Copy .env.example to .env and fill it in."
        )


def require_sagemaker_config(config: Config) -> None:
    """Call at the top of any SageMaker launch script - fails fast with a
    clear message instead of SageMaker rejecting an empty role ARN later.
    """
    if not config.sagemaker_role_arn:
        raise RuntimeError(
            "SAGEMAKER_ROLE_ARN is not set. Copy .env.example to .env and fill it in."
        )
    require_s3_bucket(config)
