import json

import pytest

from fod_pipeline.core.labels import (
    canonical_label,
    group_by_label,
    load_label_map,
    load_synonym_mapping,
    map_object_ids_to_labels,
    normalize_label,
    reverse_label_map,
)


def test_map_object_ids_to_labels_prefers_join_config_over_csv_fallback():
    resolved, unmapped = map_object_ids_to_labels(
        object_ids=["A1", "A2", "A3"],
        join_config={"A1": "Screw"},
        csv_fallback={"A1": "Should not be used", "A2": "Bolt"},
    )

    assert resolved == {"A1": "Screw", "A2": "Bolt"}
    assert unmapped == ["A3"]


def test_group_by_label_and_reverse_label_map_round_trip():
    object_id_to_label = {"A1": "Screw", "A2": "Bolt", "A3": "Screw"}

    grouped = group_by_label(object_id_to_label)
    assert grouped == {"Bolt": ["A2"], "Screw": ["A1", "A3"]}

    assert reverse_label_map(grouped) == object_id_to_label


def test_reverse_label_map_raises_on_duplicate_object_id():
    with pytest.raises(ValueError):
        reverse_label_map({"Screw": ["A1"], "Bolt": ["A1"]})


def test_load_label_map_accepts_dict_shape(tmp_path):
    path = tmp_path / "label_map.json"
    path.write_text(json.dumps({"A1": "Screw"}))

    assert load_label_map(path) == {"A1": "Screw"}


def test_load_label_map_accepts_record_list_shape(tmp_path):
    path = tmp_path / "object_labels.json"
    path.write_text(json.dumps([{"object_id": "A1", "name": "Screw"}]))

    assert load_label_map(path) == {"A1": "Screw"}


def test_normalize_label():
    assert normalize_label(" Light_Bulb ") == "light bulb"


def test_canonical_label_maps_through_synonym_table():
    mapping = {"light bulb": "bulbs"}

    assert canonical_label("Light_Bulb", mapping) == "bulbs"
    assert canonical_label("unmapped label", mapping) == "unmapped label"


def test_load_synonym_mapping(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"category_to_objects": {"Bolt": ["Bolts"]}}))

    assert load_synonym_mapping(path) == {"bolt": "bolts"}
