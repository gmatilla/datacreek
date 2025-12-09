import datetime as dt

from datacreek.hypergraph import (
    LATE_EDGE_TOTAL,
    RedisGraphHotLayer,
    process_edge_stream_with_watermark,
)


def test_late_event_within_bound_processed():
    """An event 8 minutes late should still be written to the hot layer."""
    base = dt.datetime(2024, 1, 1, 0, 0, 0)
    events = [
        {"src": "new", "dst": "later", "ts": base + dt.timedelta(minutes=10)},
        # 8 minutes older than the current max timestamp
        {"src": "old", "dst": "late", "ts": base + dt.timedelta(minutes=2)},
    ]
    hot = RedisGraphHotLayer()
    late: list[dict] = []
    process_edge_stream_with_watermark(events, hot, late_sink=late.append)

    assert set(hot.neighbours("old")) == {"late"}
    assert late == []


def test_event_beyond_bound_goes_to_late_sink():
    """Events older than the watermark are forwarded to the late sink."""
    base = dt.datetime(2024, 1, 1, 0, 0, 0)
    events = [
        {"src": "new", "dst": "later", "ts": base + dt.timedelta(minutes=10)},
        # 15 minutes older than the current max timestamp -> late
        {"src": "old", "dst": "too_late", "ts": base - dt.timedelta(minutes=5)},
    ]
    hot = RedisGraphHotLayer()
    late: list[dict] = []
    start = LATE_EDGE_TOTAL._value.get() if LATE_EDGE_TOTAL is not None else 0.0
    process_edge_stream_with_watermark(events, hot, late_sink=late.append)

    assert hot.neighbours("old") == []
    assert len(late) == 1 and late[0]["dst"] == "too_late"
    if LATE_EDGE_TOTAL is not None:
        assert LATE_EDGE_TOTAL._value.get() == start + 1
