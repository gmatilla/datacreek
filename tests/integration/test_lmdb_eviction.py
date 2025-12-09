import datacreek.analysis.mapper as mapper


def test_start_stop_eviction_thread(monkeypatch):
    events = []

    def fake_worker(path, limit_mb, ttl_h, interval):
        events.append((path, limit_mb, ttl_h, interval))

    monkeypatch.setattr(mapper, "_evict_worker", fake_worker)
    mapper.start_l2_eviction_thread("db", interval=0.01)
    assert events and events[0][0] == "db"
    assert mapper._evict_thread is not None
    mapper.stop_l2_eviction_thread()
    assert mapper._evict_thread is None
