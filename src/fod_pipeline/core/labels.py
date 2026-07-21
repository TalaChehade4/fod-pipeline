"""Object-ID -> label resolution: manifest parsing, join-config lookup, CSV fallback.

Consolidates what used to be three near-duplicate scripts
(build_objectid_label_map.py, build_object_labels_for_mobileclip.py,
map_labels.py) into shared building blocks, plus the reverse-mapping step
(ReverseMapping.py) and the MobileCLIP synonym-mapping step
(canonical_label / mobileclip_category_mapping_new.json) from MobileCLIP_Alone.

Two label-map shapes appear on disk in this project:
  - a flat dict   {object_id: label}                  (e.g. Label_map.json)
  - a record list [{"object_id": ..., "name": ...}]    (e.g. object_labels.json)
load_label_map() accepts either.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

MANIFEST_OBJECT_ID_PATTERN = re.compile(r"^([^/]+)/\1_(?P<object_id>.+)_IM\d+\.\w+$")


def extract_object_ids_from_manifest(
    manifest_path: Path, pattern: re.Pattern = MANIFEST_OBJECT_ID_PATTERN
) -> set:
    """Unique object IDs referenced by a manifest's image paths."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    object_ids = set()
    unmatched = []

    for entry in entries[1:]:  # entries[0] is the {"prefix": ...} dict
        match = pattern.match(entry)
        if match:
            object_ids.add(match.group("object_id"))
        else:
            unmatched.append(entry)

    if unmatched:
        print(f"WARNING: {len(unmatched)} manifest paths didn't match the expected pattern:")
        for u in unmatched[:10]:
            print(f"  {u}")

    return object_ids


def load_join_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_id_name_map(path: Path, id_column: str, name_column: str = "name") -> dict:
    """e.g. trainingdata_old.csv (id_column='trainingID') or
    testingdata_old.csv (id_column='testingID')."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row[id_column].strip(): row[name_column].strip() for row in reader}


def map_object_ids_to_labels(
    object_ids, join_config: dict | None = None, csv_fallback: dict | None = None
) -> tuple:
    """Resolve each object ID to a label: join_config first, then csv_fallback.

    Returns (object_id -> label dict, sorted list of unmapped object IDs).
    """
    join_config = join_config or {}
    csv_fallback = csv_fallback or {}

    resolved = {}
    unmapped = []

    for object_id in sorted(object_ids):
        if object_id in join_config:
            resolved[object_id] = join_config[object_id]
        elif object_id in csv_fallback:
            resolved[object_id] = csv_fallback[object_id]
        else:
            unmapped.append(object_id)

    return resolved, unmapped


def group_by_label(object_id_to_label: dict) -> dict:
    """{object_id: label} -> {label: [object_id, ...]}, sorted by label then ID."""
    label_to_ids = defaultdict(list)
    for object_id, label in object_id_to_label.items():
        label_to_ids[label].append(object_id)
    return {label: sorted(ids) for label, ids in sorted(label_to_ids.items())}


def reverse_label_map(label_to_ids: dict) -> dict:
    """{label: [object_id, ...]} -> {object_id: label}, raising on duplicate IDs."""
    object_id_to_label = {}
    for label, object_ids in label_to_ids.items():
        for object_id in object_ids:
            if object_id in object_id_to_label:
                raise ValueError(f"Duplicate ObjectID found: {object_id}")
            object_id_to_label[object_id] = label
    return object_id_to_label


def load_label_map(path: Path) -> dict:
    """Load an object_id -> label map, accepting either on-disk shape:
      - dict:  {object_id: label}
      - list:  [{"object_id": ..., "name": ...}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data

    return {record["object_id"]: record["name"] for record in data}


def normalize_label(label) -> str:
    return str(label).lower().strip().replace("_", " ")


def canonical_label(label: str, synonym_mapping: dict) -> str:
    """Map a label through a synonym table (e.g. MobileCLIP category name ->
    dataset label), normalizing both sides. Labels absent from the mapping
    pass through normalized but otherwise unchanged.
    """
    label = normalize_label(label)
    return normalize_label(synonym_mapping.get(label, label))


def load_synonym_mapping(path: Path) -> dict:
    """Load a {"category_to_objects": {category: [dataset_label, ...]}} file
    (e.g. mobileclip_category_mapping_new.json) into a flat, normalized
    {category: dataset_label} map. Only the first dataset label per category
    is used - these mappings are expected to be 1:1 in practice.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return {
        normalize_label(category): normalize_label(dataset_labels[0])
        for category, dataset_labels in data["category_to_objects"].items()
    }
