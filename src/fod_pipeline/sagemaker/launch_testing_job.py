"""Launch the Stage 4/5 evaluation job: hybrid pipeline over a labeled test
manifest, producing the full metrics report (Sections 4 and 5).

Replaces three separate jobs - TestingClassifier/Launch_Job.py,
MobileCLIP_Alone/Launch_Job.py, and the manual CSV-combination step in
resultsOfHybrid/ - with one, since fod_pipeline.pipeline.evaluate already
runs both models per image and compares against both ground truths itself.
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

    parser.add_argument("--manifest-uri", type=str, required=True)
    parser.add_argument(
        "--mobileclip-label-map-uri", type=str, required=True, help="Ground Truth A"
    )
    parser.add_argument(
        "--classifier-label-map-uri", type=str, required=True, help="Ground Truth B"
    )
    parser.add_argument("--yolo-weights-uri", type=str, required=True)
    parser.add_argument("--mobileclip-weights-uri", type=str, required=True)
    parser.add_argument(
        "--mobileclip-mapping-uri",
        type=str,
        default=None,
        help="category_to_objects synonym map (e.g. mobileclip_category_mapping_new.json) - "
        "required for meaningful Stage 3 accuracy, since MobileCLIP's category vocabulary "
        "differs from the dataset's ground-truth vocabulary",
    )
    parser.add_argument("--classifier-weights-uri", type=str, required=True)
    parser.add_argument("--label-encoder-uri", type=str, required=True)
    parser.add_argument("--output-uri", type=str, required=True)
    parser.add_argument("--job-name", type=str, default="fod-hybrid-evaluation")
    parser.add_argument(
        "--fp16", action="store_true", help="Run YOLO/MobileCLIP in fp16 on the GPU instance"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = get_config()
    require_sagemaker_config(config)

    processor = PyTorchProcessor(
        framework_version="2.8",
        py_version="py311",
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
    manifest_filename = os.path.basename(args.manifest_uri)
    mobileclip_label_map_filename = os.path.basename(args.mobileclip_label_map_uri)
    classifier_label_map_filename = os.path.basename(args.classifier_label_map_uri)
    yolo_tar_filename = os.path.basename(args.yolo_weights_uri)
    mobileclip_filename = os.path.basename(args.mobileclip_weights_uri)
    classifier_weights_filename = os.path.basename(args.classifier_weights_uri)
    label_encoder_filename = os.path.basename(args.label_encoder_uri)

    arguments = [
        "--manifest", f"/opt/ml/processing/input/manifest/{manifest_filename}",
        "--mobileclip-label-map",
        f"/opt/ml/processing/input/mobileclip_labels/{mobileclip_label_map_filename}",
        "--classifier-label-map",
        f"/opt/ml/processing/input/classifier_labels/{classifier_label_map_filename}",
        "--yolo-tar", f"/opt/ml/processing/input/yolo/{yolo_tar_filename}",
        "--mobileclip", f"/opt/ml/processing/input/mobileclip/{mobileclip_filename}",
        "--classifier-weights",
        f"/opt/ml/processing/input/classifier/{classifier_weights_filename}",
        "--label-encoder", f"/opt/ml/processing/input/classifier/{label_encoder_filename}",
        "--output-dir", "/opt/ml/processing/output",
    ]

    if args.fp16:
        arguments.append("--fp16")

    inputs = [
        ProcessingInput(
            source=args.manifest_uri, destination="/opt/ml/processing/input/manifest"
        ),
        ProcessingInput(
            source=args.mobileclip_label_map_uri,
            destination="/opt/ml/processing/input/mobileclip_labels",
        ),
        ProcessingInput(
            source=args.classifier_label_map_uri,
            destination="/opt/ml/processing/input/classifier_labels",
        ),
        ProcessingInput(
            source=args.yolo_weights_uri, destination="/opt/ml/processing/input/yolo"
        ),
        ProcessingInput(
            source=args.mobileclip_weights_uri,
            destination="/opt/ml/processing/input/mobileclip",
        ),
        ProcessingInput(
            source=args.classifier_weights_uri,
            destination="/opt/ml/processing/input/classifier",
        ),
        ProcessingInput(
            source=args.label_encoder_uri,
            destination="/opt/ml/processing/input/classifier",
        ),
    ]

    if args.mobileclip_mapping_uri:
        mapping_filename = os.path.basename(args.mobileclip_mapping_uri)
        arguments += [
            "--mobileclip-mapping", f"/opt/ml/processing/input/mapping/{mapping_filename}"
        ]
        inputs.append(
            ProcessingInput(
                source=args.mobileclip_mapping_uri,
                destination="/opt/ml/processing/input/mapping",
            )
        )

    processor.run(
        code=str(FOD_PIPELINE_PACKAGE / "pipeline" / "evaluate.py"),
        dependencies=package_dependencies(),
        arguments=arguments,
        inputs=inputs,
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=args.output_uri),
        ],
    )


if __name__ == "__main__":
    main()
