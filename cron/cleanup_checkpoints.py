"""Cron job for S3 checkpoint lifecycle management.

This script removes model checkpoint directories from an S3 bucket when
both conditions are met:

* the checkpoint is older than a configurable retention window
  (30 days by default)
* the checkpoint is not among the ``top_k`` most recent ones for its run
  (a run corresponds to the first path component under the prefix)

Keeping only a handful of recent checkpoints prevents the training
bucket from growing without bound. The code relies solely on object
timestamps so it works even if no metric metadata is available.

Environment variables
---------------------
``CHECKPOINT_BUCKET``  – S3 bucket containing checkpoints (required)
``CHECKPOINT_PREFIX``  – common prefix under the bucket (default: ``""``)
``CHECKPOINT_TOP_K``   – number of recent checkpoints to retain per run
``CHECKPOINT_MIN_AGE_DAYS`` – age in days before deletion (default: 30)

Example
-------
.. code-block:: bash

    CHECKPOINT_BUCKET=my-bucket \
    CHECKPOINT_PREFIX=models/ \
    CHECKPOINT_TOP_K=2 \
    python cron/cleanup_checkpoints.py
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable

import boto3


def _list_objects(client, bucket: str, prefix: str) -> Iterable[dict]:
    """Yield all objects under ``bucket/prefix``.

    Parameters
    ----------
    client:
        Boto3 S3 client.
    bucket:
        Name of the S3 bucket to list.
    prefix:
        Key prefix under which checkpoints are stored.
    """

    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            # Skip placeholders for "directories" in S3 which end with '/'.
            if obj["Key"].endswith("/"):
                continue
            yield obj


def cleanup_old_checkpoints(
    bucket: str,
    prefix: str = "",
    *,
    top_k: int = 2,
    min_age_days: int = 30,
    client=None,
    now: datetime | None = None,
) -> list[str]:
    """Remove outdated checkpoints from S3.

    Checkpoints are grouped by their run (first path component). Only the
    ``top_k`` most recent checkpoints per run are preserved. Older
    checkpoints beyond that and older than ``min_age_days`` are deleted.

    Parameters
    ----------
    bucket:
        Name of the S3 bucket containing checkpoints.
    prefix:
        Key prefix, if checkpoints are nested under a common directory.
    top_k:
        Number of recent checkpoints to keep per run.
    min_age_days:
        Minimum age in days before a checkpoint becomes eligible for
        deletion.
    client:
        Optional S3 client. When ``None``, ``boto3.client("s3")`` is used.
    now:
        Reference time for computing checkpoint age. Defaults to the current
        UTC time. Primarily intended for testing.

    Returns
    -------
    list[str]
        Keys of checkpoints that were deleted.
    """

    s3 = client or boto3.client("s3")
    current_time = now or datetime.now(timezone.utc)
    objects = list(_list_objects(s3, bucket, prefix))

    # Group checkpoints by the first path component, assumed to be the run
    # identifier. e.g. "run_123/epoch_1.ckpt" -> "run_123".
    runs: dict[str, list[dict]] = defaultdict(list)
    for obj in objects:
        run_id = obj["Key"].split("/", 1)[0]
        runs[run_id].append(obj)

    to_delete: list[dict] = []
    for run_objs in runs.values():
        # Sort by recency (LastModified), newest first.
        run_objs.sort(key=lambda o: o["LastModified"], reverse=True)
        # Determine checkpoint keys to keep.
        keep = {o["Key"] for o in run_objs[:top_k]}
        for obj in run_objs[top_k:]:
            age = current_time - obj["LastModified"]
            if age > timedelta(days=min_age_days):
                to_delete.append({"Key": obj["Key"]})

    for batch_start in range(0, len(to_delete), 1000):
        batch = to_delete[batch_start : batch_start + 1000]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})

    return [d["Key"] for d in to_delete]


def main() -> None:
    bucket = os.environ["CHECKPOINT_BUCKET"]
    prefix = os.environ.get("CHECKPOINT_PREFIX", "")
    top_k = int(os.environ.get("CHECKPOINT_TOP_K", "2"))
    min_age = int(os.environ.get("CHECKPOINT_MIN_AGE_DAYS", "30"))
    deleted = cleanup_old_checkpoints(bucket, prefix, top_k=top_k, min_age_days=min_age)
    print(f"deleted {len(deleted)} checkpoints")


if __name__ == "__main__":
    main()
