import datetime as dt

from datacreek.hypergraph import RedisGraphHotLayer, process_edge_stream


def test_process_edge_stream_writes_edges():
    events = [
        {"src": "a", "dst": "b", "ts": dt.datetime(2024, 1, 1, 0, 0, 0)},
        {"src": "b", "dst": "c", "ts": dt.datetime(2024, 1, 1, 0, 0, 10)},
        # falls into next 30s window and should trigger a flush
        {"src": "c", "dst": "d", "ts": dt.datetime(2024, 1, 1, 0, 0, 40)},
    ]
    hot = RedisGraphHotLayer()
    process_edge_stream(events, hot)

    assert set(hot.neighbours("a")) == {"b"}
    assert set(hot.neighbours("b")) == {"c"}
    assert set(hot.neighbours("c")) == {"d"}


def test_hot_layer_p95_latency_under_threshold():
    hot = RedisGraphHotLayer()
    for i in range(100):
        hot.add_edge(f"n{i}", f"n{i+1}")
        hot.neighbours(f"n{i}")
    assert hot.p95_latency_ms() < 500
