"""Launches the SageMaker Processing job that generates MobileCLIP embeddings.

This script configures a PyTorchProcessor, downloads the required inputs
(manifest, label map, YOLO weights, and MobileCLIP weights) from S3,
runs preprocess.py inside the SageMaker container, and uploads the
generated embedding files back to S3. Default S3 locations are taken
from config.py, but can be overridden through command-line arguments.
"""
from __future__ import annotations

import argparse
import os

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch import PyTorchProcessor

from fod_pipeline.config import get_config, require_sagemaker_config
from fod_pipeline.sagemaker._common import FOD_PIPELINE_PACKAGE, package_dependencies


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest-uri",
        type=str,
        default=None,
        help="Defaults to manifests/<split>_manifest.json under S3_PROJECT_PREFIX if omitted.",
    )
    parser.add_argument(
        "--label-map-uri",
        type=str,
        default=None,
        help="Defaults to manifests/<split>_label_map.json under S3_PROJECT_PREFIX if omitted.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Which manifest this is - picks the default --manifest-uri, --label-map-uri, "
        "and --output-uri (under S3_PROJECT_PREFIX) when those flags are not given "
        "explicitly.",
    )
    parser.add_argument(
        "--yolo-weights-uri",
        type=str,
        default=None,
        help="S3 model.tar.gz. Defaults to the weights/yolo/model.tar.gz path under "
        "S3_PROJECT_PREFIX if omitted.",
    )
    parser.add_argument(
        "--mobileclip-weights-uri",
        type=str,
        default=None,
        help="Defaults to the weights/mobileclip/mobileclip_s0.pt path under "
        "S3_PROJECT_PREFIX if omitted.",
    )
    parser.add_argument(
        "--output-uri",
        type=str,
        default=None,
        help="Defaults to embeddings/<split> under S3_PROJECT_PREFIX if omitted.",
    )
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--job-name", type=str, default="fod-embedding-extraction")

    return parser.parse_args()


def main():
    args = parse_args()

    config = get_config()
    require_sagemaker_config(config)

    manifest_uri = args.manifest_uri or config.s3.manifest(args.split)
    label_map_uri = args.label_map_uri or config.s3.label_map(args.split)
    yolo_weights_uri = args.yolo_weights_uri or config.s3.yolo_weights_tar
    mobileclip_weights_uri = args.mobileclip_weights_uri or config.s3.mobileclip_weights
    output_uri = args.output_uri or (
        config.s3.train_embeddings if args.split == "train" else config.s3.test_embeddings
    )

    processor = PyTorchProcessor(
        framework_version="2.8",
        py_version="py312",
        role=config.sagemaker_role_arn,
        instance_type=config.embedding_instance_type,
        instance_count=1,
        base_job_name=args.job_name,
        sagemaker_session=sagemaker.Session(),
    )

    # ProcessingInput preserves each S3 object's own filename inside its
    # destination directory - it does not rename to a fixed name - so the
    # container-side argument paths must be built from the real basenames,
    # not assumed generic names like "manifest.json".
    manifest_filename = os.path.basename(manifest_uri)
    label_map_filename = os.path.basename(label_map_uri)
    yolo_tar_filename = os.path.basename(yolo_weights_uri)
    mobileclip_filename = os.path.basename(mobileclip_weights_uri)

    processor.run(
        code=str(FOD_PIPELINE_PACKAGE / "pipeline" / "preprocess.py"),
        dependencies=package_dependencies(),
        arguments=[
            "--manifest", f"/opt/ml/processing/input/manifest/{manifest_filename}",
            "--label-map", f"/opt/ml/processing/input/labels/{label_map_filename}",
            "--yolo-tar", f"/opt/ml/processing/input/yolo/{yolo_tar_filename}",
            "--mobileclip", f"/opt/ml/processing/input/mobileclip/{mobileclip_filename}",
            "--output-dir", "/opt/ml/processing/output",
            "--max-images", str(args.max_images),
        ],
        inputs=[
            ProcessingInput(
                source=manifest_uri, destination="/opt/ml/processing/input/manifest"
            ),
            ProcessingInput(
                source=label_map_uri, destination="/opt/ml/processing/input/labels"
            ),
            ProcessingInput(
                source=yolo_weights_uri, destination="/opt/ml/processing/input/yolo"
            ),
            ProcessingInput(
                source=mobileclip_weights_uri,
                destination="/opt/ml/processing/input/mobileclip",
            ),
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=output_uri),
        ],
    )


if __name__ == "__main__":
    main()
