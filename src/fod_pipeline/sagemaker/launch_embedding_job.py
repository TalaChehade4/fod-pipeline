"""Launch the Stage 1+2 embedding-extraction job (YOLO crop + MobileCLIP embed).

Used for both training-data and test-data extraction - point
--manifest-uri/--label-map-uri/--output-uri at whichever dataset you're
preparing. Replaces Preprocessing1/launch_job.py and
PreparingDataForTesting/Launch_Job_PreTesting.py, which ran identical code
against different S3 paths.
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
    parser.add_argument("--label-map-uri", type=str, required=True)
    parser.add_argument("--yolo-weights-uri", type=str, required=True, help="S3 model.tar.gz")
    parser.add_argument("--mobileclip-weights-uri", type=str, required=True)
    parser.add_argument("--output-uri", type=str, required=True)
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--job-name", type=str, default="fod-embedding-extraction")

    return parser.parse_args()


def main():
    args = parse_args()

    config = get_config()
    require_sagemaker_config(config)

    processor = PyTorchProcessor(
        framework_version="2.8",
        py_version="py311",
        role=config.sagemaker_role_arn,
        instance_type=config.embedding_instance_type,
        instance_count=1,
        base_job_name=args.job_name,
        sagemaker_session=sagemaker.Session(),
    )

    processor.run(
        code=str(FOD_PIPELINE_PACKAGE / "pipeline" / "preprocess.py"),
        dependencies=package_dependencies(),
        arguments=[
            "--manifest", "/opt/ml/processing/input/manifest/manifest.json",
            "--label-map", "/opt/ml/processing/input/labels/label_map.json",
            "--yolo-tar", "/opt/ml/processing/input/yolo/model.tar.gz",
            "--mobileclip", "/opt/ml/processing/input/mobileclip/mobileclip_s0.pt",
            "--output-dir", "/opt/ml/processing/output",
            "--max-images", str(args.max_images),
        ],
        inputs=[
            ProcessingInput(
                source=args.manifest_uri, destination="/opt/ml/processing/input/manifest"
            ),
            ProcessingInput(
                source=args.label_map_uri, destination="/opt/ml/processing/input/labels"
            ),
            ProcessingInput(
                source=args.yolo_weights_uri, destination="/opt/ml/processing/input/yolo"
            ),
            ProcessingInput(
                source=args.mobileclip_weights_uri,
                destination="/opt/ml/processing/input/mobileclip",
            ),
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=args.output_uri),
        ],
    )


if __name__ == "__main__":
    main()
