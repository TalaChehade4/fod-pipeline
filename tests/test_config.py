from fod_pipeline.config import S3Paths


def test_manifest_and_label_map_defaults_are_split_scoped():
    s3 = S3Paths(bucket="my-bucket", project_prefix="fod-pipeline")

    assert s3.manifest("train") == "s3://my-bucket/fod-pipeline/manifests/train_manifest.json"
    assert s3.manifest("test") == "s3://my-bucket/fod-pipeline/manifests/test_manifest.json"

    assert s3.label_map("train") == "s3://my-bucket/fod-pipeline/manifests/train_label_map.json"
    assert (
        s3.mobileclip_label_map("test")
        == "s3://my-bucket/fod-pipeline/manifests/test_mobileclip_label_map.json"
    )


def test_database_csv_and_join_config_defaults():
    s3 = S3Paths(bucket="my-bucket", project_prefix="fod-pipeline")

    assert (
        s3.database_csv("train")
        == "s3://my-bucket/fod-pipeline/manifests/trainingdata_old.csv"
    )
    assert (
        s3.database_csv("test")
        == "s3://my-bucket/fod-pipeline/manifests/testingdata_old.csv"
    )
    assert s3.join_config == "s3://my-bucket/fod-pipeline/manifests/join_config.json"
