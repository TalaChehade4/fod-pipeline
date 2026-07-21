"""Launch the Stage 4/5 evaluation job: hybrid pipeline over a labeled test
manifest, producing the full metrics report (Sections 4 and 5).

Replaces three separate jobs - TestingClassifier/Launch_Job.py,
MobileCLIP_Alone/Launch_Job.py, and the manual CSV-combination step in
resultsOfHybrid/ - with one, since fod_pipeline.pipeline.evaluate already
runs both models per image and compares against both ground truths itself.
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

    parser.add_argument("--manifest-uri", type=str, required=True)
    parser.add_argument(
        "--mobileclip-label-map-uri", type=str, required=True, help="Ground Truth A"
    )
    parser.add_argument(
        "--classifier-label-map-uri", type=str, required=True, help="Ground Truth B"
    )
    parser.add_argument("--yolo-weights-uri", type=str, required=True)
    parser.add_argument("--mobileclip-weights-uri", type=str, required=True)
    parser.add_argument("--classifier-weights-uri", type=str, required=True)
    parser.add_argument("--label-encoder-uri", type=str, required=True)
    parser.add_argument("--output-uri", type=str, required=True)
    parser.add_argument("--job-name", type=str, default="fod-hybrid-evaluation")

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

    processor.run(
        code=str(FOD_PIPELINE_PACKAGE / "pipeline" / "evaluate.py"),
        dependencies=package_dependencies(),
        arguments=[
            "--manifest", "/opt/ml/processing/input/manifest/manifest.json",
            "--mobileclip-label-map", "/opt/ml/processing/input/mobileclip_labels/label_map.json",
            "--classifier-label-map", "/opt/ml/processing/input/classifier_labels/label_map.json",
            "--yolo-tar", "/opt/ml/processing/input/yolo/model.tar.gz",
            "--mobileclip", "/opt/ml/processing/input/mobileclip/mobileclip_s0.pt",
            "--classifier-weights", "/opt/ml/processing/input/classifier/model.pth",
            "--label-encoder", "/opt/ml/processing/input/classifier/label_encoder.json",
            "--output-dir", "/opt/ml/processing/output",
        ],
        inputs=[
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
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=args.output_uri),
        ],
    )


if __name__ == "__main__":
    main()
