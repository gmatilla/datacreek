"""S3 lifecycle utilities for toxic log retention.

This module provides helpers to apply a lifecycle rule ensuring that
objects under the ``ingest-toxic/`` prefix expire automatically after a
fixed number of days. Enforcing a lifecycle rule keeps the bucket free
of stale toxic ingestion logs, satisfying retention policies.
"""

from __future__ import annotations

import boto3


def apply_toxic_log_lifecycle(
    bucket: str,
    *,
    prefix: str = "ingest-toxic/",
    days: int = 7,
    client=None,
) -> None:
    """Configure S3 lifecycle rule to expire toxic logs.

    Parameters
    ----------
    bucket:
        Name of the S3 bucket.
    prefix:
        Object key prefix where toxic logs are stored.
    days:
        Retention period in days before automatic expiration.
    client:
        Optional boto3 S3 client used for the API call. When ``None`` a new
        client is instantiated via :func:`boto3.client`.
    """

    s3 = client or boto3.client("s3")
    config = {
        "Rules": [
            {
                # Identifier for the lifecycle rule to ease future updates.
                "ID": "toxic-log-retention",
                # Filter ensures the rule only applies to the toxic prefix.
                "Filter": {"Prefix": prefix},
                "Status": "Enabled",
                # Expire objects after the configured number of days.
                "Expiration": {"Days": days},
            }
        ]
    }
    s3.put_bucket_lifecycle_configuration(Bucket=bucket, LifecycleConfiguration=config)
