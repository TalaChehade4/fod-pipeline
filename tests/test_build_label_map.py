import json
import shutil
import sys

import pytest

from fod_pipeline.config import S3Paths
from fod_pipeline.pipeline.build_label_map import (
    build_label_map,
    main,
    resolve_csv,
    resolve_id_column,
    resolve_join_config,
    resolve_manifest,
    resolve_output,
)


def _write_manifest(tmp_path, object_ids):
    manifest = [{"prefix": "s3://bucket/path/"}] + [
        f"FRS0131/FRS0131_{oid}_IM0001.png" for oid in object_ids
    ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _write_csv(tmp_path, rows, id_column="trainingID"):
    path = tmp_path / "database.csv"
    lines = [f"{id_column},name"] + [f"{oid},{name}" for oid, name in rows]
    path.write_text("\n".join(lines))
    return path


def test_classifier_ground_truth_prefers_join_config_over_csv(tmp_path):
    manifest_path = _write_manifest(tmp_path, ["A1-B", "A2-B"])
    csv_path = _write_csv(tmp_path, [("A1-B", "Should not be used"), ("A2-B", "Bolt")])
    join_config_path = tmp_path / "join_config.json"
    join_config_path.write_text(json.dumps({"A1-B": "Screw"}))

    resolved, unmapped = build_label_map(
        manifest_path=manifest_path,
        csv_path=csv_path,
        id_column="trainingID",
        join_config_path=join_config_path,
    )

    assert resolved == {"A1-B": "Screw", "A2-B": "Bolt"}
    assert unmapped == []


def test_mobileclip_ground_truth_resolves_from_csv_alone(tmp_path):
    manifest_path = _write_manifest(tmp_path, ["A1-B"])
    csv_path = _write_csv(tmp_path, [("A1-B", "Screw")], id_column="testingID")

    resolved, unmapped = build_label_map(
        manifest_path=manifest_path, csv_path=csv_path, id_column="testingID"
    )

    assert resolved == {"A1-B": "Screw"}
    assert unmapped == []


def test_unmapped_object_ids_are_reported(tmp_path):
    manifest_path = _write_manifest(tmp_path, ["A1-B", "A2-B"])
    csv_path = _write_csv(tmp_path, [("A1-B", "Screw")])

    resolved, unmapped = build_label_map(
        manifest_path=manifest_path, csv_path=csv_path, id_column="trainingID"
    )

    assert resolved == {"A1-B": "Screw"}
    assert unmapped == ["A2-B"]


class _FakeS3Client:
    """Treats a local directory as the "bucket" - download/upload are just
    copies to/from it, keyed by S3 key basename."""

    def __init__(self, bucket_dir):
        self.bucket_dir = bucket_dir

    def download_file(self, bucket, key, local_path):
        shutil.copy(self.bucket_dir / key, local_path)

    def upload_file(self, local_path, bucket, key):
        shutil.copy(local_path, self.bucket_dir / key)


def test_main_reads_and_writes_s3_uris(tmp_path, monkeypatch):
    bucket_dir = tmp_path / "fake-bucket"
    bucket_dir.mkdir()

    manifest_path = _write_manifest(tmp_path, ["A1-B"])
    csv_path = _write_csv(tmp_path, [("A1-B", "Screw")])
    shutil.copy(manifest_path, bucket_dir / "train_manifest.json")
    shutil.copy(csv_path, bucket_dir / "trainingdata_old.csv")

    monkeypatch.setattr(
        "fod_pipeline.pipeline.build_label_map.download_from_s3",
        lambda s3_uri, local_path, s3_client=None: _FakeS3Client(bucket_dir).download_file(
            None, s3_uri.split("/")[-1], local_path
        ),
    )
    monkeypatch.setattr(
        "fod_pipeline.pipeline.build_label_map.upload_to_s3",
        lambda local_path, s3_uri, s3_client=None: _FakeS3Client(bucket_dir).upload_file(
            local_path, None, s3_uri.split("/")[-1]
        ),
    )

    argv = [
        "fod-build-label-map",
        "--manifest", "s3://fake-bucket/manifests/train_manifest.json",
        "--csv", "s3://fake-bucket/trainingdata_old.csv",
        "--id-column", "trainingID",
        "--output", "s3://fake-bucket/manifests/train_label_map.json",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    written = json.loads((bucket_dir / "train_label_map.json").read_text())
    assert written == {"A1-B": "Screw"}


def test_build_label_map_requires_id_column_with_csv(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["fod-build-label-map", "--manifest", "m.json", "--csv", "c.csv", "--output", "o.json"],
    )

    with pytest.raises(SystemExit):
        main()


def test_requires_manifest_or_split(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fod-build-label-map", "--output", "o.json"])

    with pytest.raises(SystemExit):
        main()


def test_requires_output_or_split_and_ground_truth(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fod-build-label-map", "--manifest", "m.json"])

    with pytest.raises(SystemExit):
        main()


class _FakeConfig:
    def __init__(self, s3):
        self.s3 = s3


def _config():
    return _FakeConfig(S3Paths(bucket="my-bucket", project_prefix="fod-pipeline"))


def test_resolve_manifest_prefers_explicit_over_split():
    class _Args:
        manifest = "explicit.json"
        split = "train"

    assert resolve_manifest(_Args(), _config()) == "explicit.json"


def test_resolve_manifest_defaults_from_split():
    class _Args:
        manifest = None
        split = "train"

    assert (
        resolve_manifest(_Args(), _config())
        == "s3://my-bucket/fod-pipeline/manifests/train_manifest.json"
    )


def test_resolve_output_defaults_from_split_and_ground_truth():
    class _Args:
        output = None
        split = "test"
        ground_truth = "classifier"

    assert (
        resolve_output(_Args(), _config())
        == "s3://my-bucket/fod-pipeline/manifests/test_label_map.json"
    )

    class _MobileclipArgs:
        output = None
        split = "test"
        ground_truth = "mobileclip"

    assert (
        resolve_output(_MobileclipArgs(), _config())
        == "s3://my-bucket/fod-pipeline/manifests/test_mobileclip_label_map.json"
    )


def test_resolve_csv_prefers_explicit_over_split():
    class _Args:
        csv = "explicit.csv"
        split = "train"

    assert resolve_csv(_Args(), _config()) == "explicit.csv"


def test_resolve_csv_defaults_from_split():
    class _TrainArgs:
        csv = None
        split = "train"

    assert (
        resolve_csv(_TrainArgs(), _config())
        == "s3://my-bucket/fod-pipeline/manifests/trainingdata_old.csv"
    )

    class _TestArgs:
        csv = None
        split = "test"

    assert (
        resolve_csv(_TestArgs(), _config())
        == "s3://my-bucket/fod-pipeline/manifests/testingdata_old.csv"
    )


def test_resolve_csv_none_without_csv_or_split():
    class _Args:
        csv = None
        split = None

    assert resolve_csv(_Args(), _config()) is None


def test_resolve_id_column_defaults_from_split():
    class _TrainArgs:
        id_column = None
        split = "train"

    assert resolve_id_column(_TrainArgs()) == "trainingID"

    class _TestArgs:
        id_column = None
        split = "test"

    assert resolve_id_column(_TestArgs()) == "testingID"


def test_resolve_id_column_prefers_explicit():
    class _Args:
        id_column = "customID"
        split = "train"

    assert resolve_id_column(_Args()) == "customID"


def test_resolve_join_config_only_defaults_for_classifier():
    class _ClassifierArgs:
        join_config = None
        split = "train"
        ground_truth = "classifier"

    assert (
        resolve_join_config(_ClassifierArgs(), _config())
        == "s3://my-bucket/fod-pipeline/manifests/join_config.json"
    )

    class _MobileclipArgs:
        join_config = None
        split = "test"
        ground_truth = "mobileclip"

    assert resolve_join_config(_MobileclipArgs(), _config()) is None


def test_main_fully_defaults_from_split_and_ground_truth_alone(tmp_path, monkeypatch):
    """No --manifest/--csv/--id-column/--join-config/--output at all - just
    --split + --ground-truth, matching the fully S3-only workflow."""
    bucket_dir = tmp_path / "fake-bucket"
    bucket_dir.mkdir()

    manifest_path = _write_manifest(tmp_path, ["A1-B", "A2-B"])
    csv_path = _write_csv(tmp_path, [("A2-B", "Bolt")], id_column="trainingID")
    join_config_path = tmp_path / "join_config.json"
    join_config_path.write_text(json.dumps({"A1-B": "Screw"}))
    shutil.copy(manifest_path, bucket_dir / "train_manifest.json")
    shutil.copy(csv_path, bucket_dir / "trainingdata_old.csv")
    shutil.copy(join_config_path, bucket_dir / "join_config.json")

    monkeypatch.setattr(
        "fod_pipeline.pipeline.build_label_map.download_from_s3",
        lambda s3_uri, local_path, s3_client=None: _FakeS3Client(bucket_dir).download_file(
            None, s3_uri.split("/")[-1], local_path
        ),
    )
    monkeypatch.setattr(
        "fod_pipeline.pipeline.build_label_map.upload_to_s3",
        lambda local_path, s3_uri, s3_client=None: _FakeS3Client(bucket_dir).upload_file(
            local_path, None, s3_uri.split("/")[-1]
        ),
    )
    monkeypatch.setattr("fod_pipeline.pipeline.build_label_map.get_config", lambda: _config())
    monkeypatch.setattr(
        "fod_pipeline.pipeline.build_label_map.require_s3_bucket", lambda config: None
    )

    argv = ["fod-build-label-map", "--split", "train", "--ground-truth", "classifier"]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    written = json.loads((bucket_dir / "train_label_map.json").read_text())
    assert written == {"A1-B": "Screw", "A2-B": "Bolt"}


def test_main_uses_split_and_ground_truth_defaults(tmp_path, monkeypatch):
    bucket_dir = tmp_path / "fake-bucket"
    bucket_dir.mkdir()

    manifest_path = _write_manifest(tmp_path, ["A1-B"])
    csv_path = _write_csv(tmp_path, [("A1-B", "Screw")], id_column="testingID")
    shutil.copy(manifest_path, bucket_dir / "test_manifest.json")
    shutil.copy(csv_path, bucket_dir / "testingdata_old.csv")

    monkeypatch.setattr(
        "fod_pipeline.pipeline.build_label_map.download_from_s3",
        lambda s3_uri, local_path, s3_client=None: _FakeS3Client(bucket_dir).download_file(
            None, s3_uri.split("/")[-1], local_path
        ),
    )
    monkeypatch.setattr(
        "fod_pipeline.pipeline.build_label_map.upload_to_s3",
        lambda local_path, s3_uri, s3_client=None: _FakeS3Client(bucket_dir).upload_file(
            local_path, None, s3_uri.split("/")[-1]
        ),
    )
    monkeypatch.setattr(
        "fod_pipeline.pipeline.build_label_map.get_config",
        lambda: _config(),
    )
    monkeypatch.setattr(
        "fod_pipeline.pipeline.build_label_map.require_s3_bucket", lambda config: None
    )

    argv = [
        "fod-build-label-map",
        "--csv", "s3://fake-bucket/testingdata_old.csv",
        "--id-column", "testingID",
        "--split", "test",
        "--ground-truth", "mobileclip",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    written = json.loads((bucket_dir / "test_mobileclip_label_map.json").read_text())
    assert written == {"A1-B": "Screw"}
