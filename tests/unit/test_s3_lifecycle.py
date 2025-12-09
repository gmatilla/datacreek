"""Tests for S3 toxic log lifecycle configuration utilities."""

from __future__ import annotations

import boto3
from botocore.stub import Stubber

from datacreek.s3_lifecycle import apply_toxic_log_lifecycle


def test_apply_toxic_log_lifecycle_sets_correct_rule() -> None:
    """Lifecycle rule should target ``ingest-toxic/`` and expire after 7 days."""

    s3 = boto3.client("s3", region_name="us-east-1")
    stubber = Stubber(s3)
    expected = {
        "Bucket": "my-bucket",
        "LifecycleConfiguration": {
            "Rules": [
                {
                    "ID": "toxic-log-retention",
                    "Filter": {"Prefix": "ingest-toxic/"},
                    "Status": "Enabled",
                    "Expiration": {"Days": 7},
                }
            ]
        },
    }
    stubber.add_response("put_bucket_lifecycle_configuration", {}, expected)
    stubber.activate()

    apply_toxic_log_lifecycle("my-bucket", client=s3)

    stubber.assert_no_pending_responses()
    stubber.deactivate()
