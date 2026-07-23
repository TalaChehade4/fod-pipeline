import pytest

from fod_pipeline.config import S3Paths
from fod_pipeline.pipeline.upload_artifact import resolve_destination


class _FakeConfig:
    def __init__(self, s3):
        self.s3 = s3


def _config():
    return _FakeConfig(S3Paths(bucket="my-bucket", project_prefix="fod-pipeline"))


def test_resolve_destination_yolo_weights():
    assert (
        resolve_destination(_config(), "yolo-weights", split=None)
        == "s3://my-bucket/fod-pipeline/weights/yolo/model.tar.gz"
    )


def test_resolve_destination_mobileclip_weights():
    assert (
        resolve_destination(_config(), "mobileclip-weights", split=None)
        == "s3://my-bucket/fod-pipeline/weights/mobileclip/mobileclip_s0.pt"
    )


def test_resolve_destination_manifest_requires_split():
    with pytest.raises(ValueError):
        resolve_destination(_config(), "manifest", split=None)


def test_resolve_destination_manifest_with_split():
    assert (
        resolve_destination(_config(), "manifest", split="train")
        == "s3://my-bucket/fod-pipeline/manifests/train_manifest.json"
    )


def test_resolve_destination_label_map_and_mobileclip_label_map():
    assert (
        resolve_destination(_config(), "label-map", split="test")
        == "s3://my-bucket/fod-pipeline/manifests/test_label_map.json"
    )
    assert (
        resolve_destination(_config(), "mobileclip-label-map", split="test")
        == "s3://my-bucket/fod-pipeline/manifests/test_mobileclip_label_map.json"
    )


def test_resolve_destination_database_csv_requires_split():
    with pytest.raises(ValueError):
        resolve_destination(_config(), "database-csv", split=None)


def test_resolve_destination_database_csv_with_split():
    assert (
        resolve_destination(_config(), "database-csv", split="train")
        == "s3://my-bucket/fod-pipeline/manifests/trainingdata_old.csv"
    )
    assert (
        resolve_destination(_config(), "database-csv", split="test")
        == "s3://my-bucket/fod-pipeline/manifests/testingdata_old.csv"
    )


def test_resolve_destination_join_config_no_split_needed():
    assert (
        resolve_destination(_config(), "join-config", split=None)
        == "s3://my-bucket/fod-pipeline/manifests/join_config.json"
    )
