import json

import torch
from PIL import Image

import fod_pipeline.pipeline.preprocess as preprocess


def _write_manifest(tmp_path, filenames):
    manifest = [{"prefix": "s3://bucket/"}] + filenames
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return str(path)


def _write_label_map(tmp_path, mapping):
    path = tmp_path / "label_map.json"
    path.write_text(json.dumps(mapping))
    return str(path)


def _patch_pipeline(monkeypatch, encode_calls):
    monkeypatch.setattr(
        preprocess, "load_image_from_s3", lambda uri: Image.new("RGB", (10, 10))
    )
    monkeypatch.setattr(
        preprocess,
        "crop_yolo_detection",
        lambda image, results, expansion=0.2: (image, (0, 0, 10, 10)),
    )

    def fake_encode_images(model, preprocess_fn, device, images, fp16=False, normalize=False):
        encode_calls.append(len(images))
        return torch.zeros(len(images), 512)

    monkeypatch.setattr(preprocess, "encode_images", fake_encode_images)


def test_process_manifest_batches_embedding_calls(tmp_path, monkeypatch):
    filenames = [f"BATCH1/BATCH1_OBJ{i}_IM0001.png" for i in range(5)]
    manifest_path = _write_manifest(tmp_path, filenames)
    label_map_path = _write_label_map(tmp_path, {f"OBJ{i}": "Screw" for i in range(5)})

    encode_calls = []
    _patch_pipeline(monkeypatch, encode_calls)

    batch_results, failures = preprocess.process_manifest(
        manifest_path=manifest_path,
        label_map_path=label_map_path,
        yolo_model=lambda image: None,
        mobileclip_model=None,
        mobileclip_preprocess=None,
        device=torch.device("cpu"),
        embed_batch_size=2,
    )

    assert failures == []
    assert encode_calls == [2, 2, 1]  # 5 images, batch size 2 -> 2, 2, 1
    assert len(batch_results["BATCH1"]) == 5
    assert all("embedding" in record for record in batch_results["BATCH1"])
    assert all(len(record["embedding"]) == 512 for record in batch_results["BATCH1"])


def test_process_manifest_groups_by_batch_id(tmp_path, monkeypatch):
    filenames = [
        "BATCH1/BATCH1_OBJ1_IM0001.png",
        "BATCH2/BATCH2_OBJ2_IM0001.png",
    ]
    manifest_path = _write_manifest(tmp_path, filenames)
    label_map_path = _write_label_map(tmp_path, {"OBJ1": "Screw", "OBJ2": "Bolt"})

    encode_calls = []
    _patch_pipeline(monkeypatch, encode_calls)

    batch_results, failures = preprocess.process_manifest(
        manifest_path=manifest_path,
        label_map_path=label_map_path,
        yolo_model=lambda image: None,
        mobileclip_model=None,
        mobileclip_preprocess=None,
        device=torch.device("cpu"),
        embed_batch_size=64,
    )

    assert failures == []
    assert set(batch_results) == {"BATCH1", "BATCH2"}
    assert batch_results["BATCH1"][0]["label"] == "Screw"
    assert batch_results["BATCH2"][0]["label"] == "Bolt"


def test_process_manifest_skips_and_records_failures(tmp_path, monkeypatch):
    filenames = [
        "BATCH1/BATCH1_OBJ1_IM0001.png",
        "BATCH1/malformed.png",  # extract_object_id raises: too few "_"-separated parts
        "BATCH1/BATCH1_OBJ2_IM0002.png",
    ]
    manifest_path = _write_manifest(tmp_path, filenames)
    label_map_path = _write_label_map(tmp_path, {"OBJ1": "Screw", "OBJ2": "Bolt"})

    encode_calls = []
    _patch_pipeline(monkeypatch, encode_calls)

    batch_results, failures = preprocess.process_manifest(
        manifest_path=manifest_path,
        label_map_path=label_map_path,
        yolo_model=lambda image: None,
        mobileclip_model=None,
        mobileclip_preprocess=None,
        device=torch.device("cpu"),
        embed_batch_size=64,
    )

    assert len(failures) == 1
    assert failures[0]["image"] == "BATCH1/malformed.png"
    assert len(batch_results["BATCH1"]) == 2


def test_process_manifest_checkpoints_and_flushes_final_partial_batch(tmp_path, monkeypatch):
    filenames = [f"BATCH1/BATCH1_OBJ{i}_IM0001.png" for i in range(3)]
    manifest_path = _write_manifest(tmp_path, filenames)
    label_map_path = _write_label_map(tmp_path, {f"OBJ{i}": "Screw" for i in range(3)})

    encode_calls = []
    _patch_pipeline(monkeypatch, encode_calls)

    output_dir = tmp_path / "out"

    batch_results, failures = preprocess.process_manifest(
        manifest_path=manifest_path,
        label_map_path=label_map_path,
        yolo_model=lambda image: None,
        mobileclip_model=None,
        mobileclip_preprocess=None,
        device=torch.device("cpu"),
        embed_batch_size=10,  # never hit by batch-size alone; relies on final flush
        output_dir=str(output_dir),
    )

    assert failures == []
    assert encode_calls == [3]  # only flushed once, at end-of-manifest checkpoint
    saved = json.loads((output_dir / "BATCH1_embeddings.json").read_text())
    assert len(saved) == 3
