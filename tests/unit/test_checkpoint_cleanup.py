from datetime import datetime, timedelta, timezone

import boto3
from botocore.stub import Stubber

from cron.cleanup_checkpoints import cleanup_old_checkpoints


def _now() -> datetime:
    return datetime(2024, 1, 31, tzinfo=timezone.utc)


def test_cleanup_deletes_only_old_non_topk() -> None:
    """Old checkpoints beyond the top-k most recent are deleted."""
    client = boto3.client("s3", region_name="us-east-1")
    stubber = Stubber(client)
    now = _now()
    contents = [
        {"Key": "run/ckpt_a", "LastModified": now - timedelta(days=40)},
        {"Key": "run/ckpt_b", "LastModified": now - timedelta(days=35)},
        {"Key": "run/ckpt_c", "LastModified": now - timedelta(days=5)},
    ]
    stubber.add_response(
        "list_objects_v2",
        {"Contents": contents},
        {"Bucket": "b", "Prefix": ""},
    )
    delete_body = {"Objects": [{"Key": "run/ckpt_b"}, {"Key": "run/ckpt_a"}]}
    stubber.add_response(
        "delete_objects",
        {"Deleted": delete_body["Objects"]},
        {"Bucket": "b", "Delete": delete_body},
    )
    stubber.activate()
    deleted = cleanup_old_checkpoints(
        "b", "", top_k=1, min_age_days=30, client=client, now=now
    )
    assert deleted == ["run/ckpt_b", "run/ckpt_a"]
    stubber.deactivate()


def test_cleanup_skips_recent_checkpoints():
    """Recent checkpoints are retained even if beyond top-k."""
    client = boto3.client("s3", region_name="us-east-1")
    stubber = Stubber(client)
    now = _now()
    contents = [
        {"Key": "run/ckpt_a", "LastModified": now - timedelta(days=10)},
        {"Key": "run/ckpt_b", "LastModified": now - timedelta(days=5)},
        {"Key": "run/ckpt_c", "LastModified": now - timedelta(days=1)},
    ]
    stubber.add_response(
        "list_objects_v2",
        {"Contents": contents},
        {"Bucket": "b", "Prefix": ""},
    )
    stubber.activate()
    deleted = cleanup_old_checkpoints(
        "b", "", top_k=1, min_age_days=30, client=client, now=now
    )
    assert deleted == []
    stubber.assert_no_pending_responses()
    stubber.deactivate()
