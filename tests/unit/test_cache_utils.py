import builtins
import importlib
from types import SimpleNamespace

import pytest

import datacreek.utils.cache as cache


class DummyMetric:
    def __init__(self, value=0.0):
        self._value = SimpleNamespace(get=lambda: value)
        self._stored = []

    def inc(self):
        self._stored.append("inc")


class DummyGauge:
    def __init__(self):
        self.values = []

    def set(self, v):
        self.values.append(v)


def test_cache_l1_injects_client(monkeypatch):
    fake_client = object()

    class FakeRedisModule:
        def Redis(self):
            return fake_client

    monkeypatch.setattr(cache, "redis", FakeRedisModule())

    @cache.cache_l1
    def fn(key, *, redis_client=None):
        return redis_client

    assert fn("k") is fake_client


def test_l1_cache_hit_and_miss(monkeypatch):
    responses = {"exists": True, "get": b"cached", "setex": []}

    class FakeRedis:
        def exists(self, key):
            return responses["exists"]

        def get(self, key):
            return responses["get"]

        def setex(self, key, ttl, value):
            responses["setex"].append((key, ttl, value))

    fake_redis = FakeRedis()
    monkeypatch.setattr(cache, "redis", fake_redis)
    monkeypatch.setattr(cache, "ttl_manager", SimpleNamespace(current_ttl=42))
    hit = DummyMetric()
    miss = DummyMetric()
    monkeypatch.setattr(cache, "hits", hit)
    monkeypatch.setattr(cache, "miss", miss)

    @cache.l1_cache(lambda x: f"k:{x}")
    def compute(x):
        return f"val-{x}"

    # cache hit
    assert compute("a") == b"cached"
    assert hit._stored == ["inc"]
    assert responses["setex"] == []

    # cache miss path
    responses["exists"] = False
    responses["get"] = b"other"
    result = compute("b")
    assert result == "val-b"
    assert miss._stored == ["inc"]
    assert responses["setex"] == [("k:b", 42, "val-b")]


def test_ttl_manager_run_once(monkeypatch):
    monkeypatch.setattr(cache.TTLManager, "start", lambda self: None)
    gauge = DummyGauge()
    hits = DummyMetric(5.0)
    miss = DummyMetric(1.0)
    monkeypatch.setattr(cache, "hits", hits)
    monkeypatch.setattr(cache, "miss", miss)
    monkeypatch.setattr(cache, "hit_ratio_g", gauge)

    tm = cache.TTLManager()
    tm.current_ttl = 100
    tm._target = 0.6
    tm._kp = 10.0
    tm._ki = 0.0

    tm.run_once()

    assert gauge.values  # hit ratio recorded
    assert tm.current_ttl != 100
