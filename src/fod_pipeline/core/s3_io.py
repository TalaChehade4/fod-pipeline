"""
Amazon S3 and dataset I/O utilities.

This module provides helper functions for interacting with Amazon S3 and
handling dataset metadata used throughout the FOD pipeline. It centralizes
common storage operations such as downloading and uploading files, copying
objects within S3, loading images directly from S3, and reading SageMaker
manifest files.

Responsibilities:
    - Create and reuse an Amazon S3 client.
    - Detect and parse S3 URIs.
    - Download and upload files between local storage and S3.
    - Copy objects between S3 locations.
    - Load images directly from S3 into PIL format.
    - Read SageMaker manifest files.
    - Extract Object IDs and batch IDs from dataset filenames.

These utilities isolate all storage-related logic from the rest of the
pipeline, allowing detection, embedding, and classification modules to work
with images and metadata without directly interacting with Amazon S3.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from urllib.parse import urlparse

import boto3
from PIL import Image


@lru_cache(maxsize=1)
def get_s3_client():
    return boto3.client("s3")


def is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def _bucket_and_key(s3_uri: str) -> tuple:
    parsed = urlparse(s3_uri)
    return parsed.netloc, parsed.path.lstrip("/")


def download_from_s3(s3_uri: str, local_path: str, s3_client=None) -> None:
    s3_client = s3_client or get_s3_client()
    bucket, key = _bucket_and_key(s3_uri)
    s3_client.download_file(bucket, key, local_path)


def upload_to_s3(local_path: str, s3_uri: str, s3_client=None) -> None:
    s3_client = s3_client or get_s3_client()
    bucket, key = _bucket_and_key(s3_uri)
    s3_client.upload_file(local_path, bucket, key)


def copy_within_s3(source_uri: str, dest_uri: str, s3_client=None) -> None:
    s3_client = s3_client or get_s3_client()
    source_bucket, source_key = _bucket_and_key(source_uri)
    dest_bucket, dest_key = _bucket_and_key(dest_uri)
    s3_client.copy_object(
        Bucket=dest_bucket,
        Key=dest_key,
        CopySource={"Bucket": source_bucket, "Key": source_key},
    )


def load_image_from_s3(s3_uri: str, s3_client=None) -> Image.Image:
    s3_client = s3_client or get_s3_client()

    bucket, key = _bucket_and_key(s3_uri)

    response = s3_client.get_object(Bucket=bucket, Key=key)

    image = Image.open(response["Body"])
    image.load()
    image = image.convert("RGB")

    response["Body"].close()

    return image


def load_manifest(path: str) -> tuple[str, list[str]]:
    with open(path, "r") as f:
        manifest = json.load(f)

    prefix = manifest[0]["prefix"]
    image_paths = manifest[1:]

    return prefix, image_paths


def extract_object_id(filename: str) -> str:
    """
    Example:
        FRS0131_5081-B_IM0001.png -> 5081-B
    """
    base = os.path.basename(filename)
    parts = base.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filename}")

    return parts[1]


def extract_batch_id(image_path: str) -> str:
    """
    Example:
        FRS0131/FRS0131_5081-B_IM0001.png -> FRS0131
    """
    return image_path.split("/")[0]
