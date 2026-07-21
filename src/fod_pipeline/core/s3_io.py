"""S3 image loading and manifest parsing.

Manifest format (used throughout this project):
    [
        {"prefix": "s3://bucket/path/"},
        "FRS0131/FRS0131_5081-B_IM0001.png",
        ...
    ]
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


def load_image_from_s3(s3_uri: str, s3_client=None) -> Image.Image:
    s3_client = s3_client or get_s3_client()

    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

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
