"""Launches the SageMaker Processing job that evaluates the complete hybrid pipeline.

This script configures a PyTorchProcessor, downloads all required inputs
(test manifest, label maps, model weights, and configuration files) from S3,
runs evaluate.py inside the SageMaker container, and uploads the generated
predictions and evaluation metrics back to S3. Default S3 locations are taken
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
        help="Defaults to manifests/test_manifest.json under S3_PROJECT_PREFIX if omitted.",
    )
    parser.add_argument(
        "--mobileclip-label-map-uri",
        type=str,
        default=None,
        help="Ground Truth A. Defaults to manifests/test_mobileclip_label_map.json under "
        "S3_PROJECT_PREFIX if omitted.",
    )
    parser.add_argument(
        "--classifier-label-map-uri",
        type=str,
        default=None,
        help="Ground Truth B. Defaults to manifests/test_label_map.json under "
        "S3_PROJECT_PREFIX if omitted.",
    )
    parser.add_argument(
        "--yolo-weights-uri",
        type=str,
        default=None,
        help="Defaults to the weights/yolo/model.tar.gz path under S3_PROJECT_PREFIX if omitted.",
    )
    parser.add_argument(
        "--mobileclip-weights-uri",
        type=str,
        default=None,
        help="Defaults to the weights/mobileclip/mobileclip_s0.pt path under "
        "S3_PROJECT_PREFIX if omitted.",
    )
    parser.add_argument(
        "--mobileclip-mapping-uri",
        type=str,
        default=None,
        help="category_to_objects synonym map - required for meaningful Stage 3 accuracy, "
        "since MobileCLIP's category vocabulary differs from the dataset's ground-truth "
        "vocabulary. Defaults to manifests/mobileclip_category_mapping.json under "
        "S3_PROJECT_PREFIX (upload it there with fod-upload --kind mobileclip-mapping) "
        "if omitted.",
    )
    parser.add_argument(
        "--classifier-weights-uri",
        type=str,
        default=None,
        help="Defaults to the classifier-models/model.tar.gz path under S3_PROJECT_PREFIX "
        "(where fod-sm-train copies its output) if omitted.",
    )
    parser.add_argument(
        "--label-encoder-uri",
        type=str,
        default=None,
        help="Defaults to the classifier-data/label_encoder.json path under S3_PROJECT_PREFIX "
        "(where fod-sm-prepare writes it) if omitted.",
    )
    parser.add_argument(
        "--output-uri",
        type=str,
        default=None,
        help="Defaults to hybrid-results under S3_PROJECT_PREFIX if omitted.",
    )
    parser.add_argument("--job-name", type=str, default="fod-hybrid-evaluation")
    parser.add_argument(
        "--fp16", action="store_true", help="Run YOLO/MobileCLIP in fp16 on the GPU instance"
    )
    parser.add_argument(
        "--classifier-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When MobileCLIP's top-1/top-2 miss Ground Truth A, also accept a "
        "match against Ground Truth B (classifier_gt) before calling it wrong. "
        "On by default - pass --no-classifier-fallback for strict "
        "Ground-Truth-A-only matching.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = get_config()
    require_sagemaker_config(config)

    manifest_uri = args.manifest_uri or config.s3.manifest("test")
    mobileclip_label_map_uri = (
        args.mobileclip_label_map_uri or config.s3.mobileclip_label_map("test")
    )
    classifier_label_map_uri = (
        args.classifier_label_map_uri or config.s3.label_map("test")
    )
    yolo_weights_uri = args.yolo_weights_uri or config.s3.yolo_weights_tar
    mobileclip_weights_uri = args.mobileclip_weights_uri or config.s3.mobileclip_weights
    classifier_weights_uri = args.classifier_weights_uri or config.s3.classifier_weights_tar
    label_encoder_uri = args.label_encoder_uri or config.s3.label_encoder
    mobileclip_mapping_uri = args.mobileclip_mapping_uri or config.s3.mobileclip_mapping
    output_uri = args.output_uri or config.s3.hybrid_results

    processor = PyTorchProcessor(
        framework_version="2.8",
        py_version="py312",
        role=config.sagemaker_role_arn,
        instance_type=config.testing_instance_type,
        instance_count=1,
        base_job_name=args.job_name,
        sagemaker_session=sagemaker.Session(),
    )

    # ProcessingInput preserves each S3 object's own filename inside its
    # destination directory - it does not rename to a fixed name - so the
    # container-side argument paths must be built from the real basenames,
    # not assumed generic names like "manifest.json".
    manifest_filename = os.path.basename(manifest_uri)
    mobileclip_label_map_filename = os.path.basename(mobileclip_label_map_uri)
    classifier_label_map_filename = os.path.basename(classifier_label_map_uri)
    yolo_tar_filename = os.path.basename(yolo_weights_uri)
    mobileclip_filename = os.path.basename(mobileclip_weights_uri)
    classifier_weights_filename = os.path.basename(classifier_weights_uri)
    label_encoder_filename = os.path.basename(label_encoder_uri)
    mobileclip_mapping_filename = os.path.basename(mobileclip_mapping_uri)

    arguments = [
        "--manifest", f"/opt/ml/processing/input/manifest/{manifest_filename}",
        "--mobileclip-label-map",
        f"/opt/ml/processing/input/mobileclip_labels/{mobileclip_label_map_filename}",
        "--classifier-label-map",
        f"/opt/ml/processing/input/classifier_labels/{classifier_label_map_filename}",
        "--yolo-tar", f"/opt/ml/processing/input/yolo/{yolo_tar_filename}",
        "--mobileclip", f"/opt/ml/processing/input/mobileclip/{mobileclip_filename}",
        "--classifier-weights-tar",
        f"/opt/ml/processing/input/classifier/{classifier_weights_filename}",
        "--label-encoder", f"/opt/ml/processing/input/label_encoder/{label_encoder_filename}",
        "--mobileclip-mapping",
        f"/opt/ml/processing/input/mapping/{mobileclip_mapping_filename}",
        "--output-dir", "/opt/ml/processing/output",
    ]

    if args.fp16:
        arguments.append("--fp16")

    # Forwarded explicitly (not just when true) since evaluate.py's own
    # --classifier-fallback also defaults to True - an unset flag here would
    # silently use that default instead of respecting --no-classifier-fallback.
    arguments.append(
        "--classifier-fallback" if args.classifier_fallback else "--no-classifier-fallback"
    )

    inputs = [
        ProcessingInput(
            source=manifest_uri, destination="/opt/ml/processing/input/manifest"
        ),
        ProcessingInput(
            source=mobileclip_label_map_uri,
            destination="/opt/ml/processing/input/mobileclip_labels",
        ),
        ProcessingInput(
            source=classifier_label_map_uri,
            destination="/opt/ml/processing/input/classifier_labels",
        ),
        ProcessingInput(
            source=yolo_weights_uri, destination="/opt/ml/processing/input/yolo"
        ),
        ProcessingInput(
            source=mobileclip_weights_uri,
            destination="/opt/ml/processing/input/mobileclip",
        ),
        ProcessingInput(
            source=classifier_weights_uri,
            destination="/opt/ml/processing/input/classifier",
        ),
        ProcessingInput(
            source=label_encoder_uri,
            destination="/opt/ml/processing/input/label_encoder",
        ),
        ProcessingInput(
            source=mobileclip_mapping_uri,
            destination="/opt/ml/processing/input/mapping",
        ),
    ]

    processor.run(
        code=str(FOD_PIPELINE_PACKAGE / "pipeline" / "evaluate.py"),
        dependencies=package_dependencies(),
        arguments=arguments,
        inputs=inputs,
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=output_uri),
        ],
    )


if __name__ == "__main__":
    main()
