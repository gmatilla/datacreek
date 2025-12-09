from datetime import datetime, timedelta, timezone

from datacreek.soft_delete import Node, Vector, mark_deleted, purge_deleted


def test_mark_deleted_sets_timestamp():
    node = Node(id="n1")
    assert node.deleted_at is None
    mark_deleted(node)
    assert node.deleted_at is not None
    # timestamp should be within a second of now
    assert abs((datetime.now(timezone.utc) - node.deleted_at).total_seconds()) < 1


def test_purge_deleted_removes_old_entities():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    fresh = Node(id="fresh", deleted_at=now - timedelta(days=10))
    stale = Vector(id="stale", deleted_at=now - timedelta(days=31))
    remaining = purge_deleted([fresh, stale], now=now)
    assert remaining == [fresh]
