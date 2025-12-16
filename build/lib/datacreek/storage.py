from __future__ import annotations

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Simple interface for in-memory result storage."""

    @abstractmethod
    def save(self, key: str, data: str) -> str:
        """Persist ``data`` under ``key`` and return the key."""


class RedisStorage(StorageBackend):
    """Store results in Redis."""

    def __init__(self, client):
        self.client = client

    def save(self, key: str, data: str) -> str:
        self.client.set(key, data)
        return key


class S3Storage(StorageBackend):
    """Store results in an S3 bucket."""

    def __init__(self, bucket: str, prefix: str = "", client=None):
        import boto3

        self.bucket = bucket
        self.prefix = (
            prefix.rstrip("/") + "/" if prefix and not prefix.endswith("/") else prefix
        )
        self.client = client or boto3.client("s3")

    def save(self, key: str, data: str) -> str:
        object_key = f"{self.prefix}{key}"
        self.client.put_object(Bucket=self.bucket, Key=object_key, Body=data.encode())
        return object_key
