import importlib
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture()
def reload_cache(monkeypatch, tmp_path):
    """Reload the cache module with stubbed dependencies and config."""
    cfg = tmp_path / "c.yml"
    cfg.write_text("cache:\n  l1_ttl_init: 5\n  l1_ttl_min: 1\n  l1_ttl_max: 10\n")
    monkeypatch.setenv("DATACREEK_CONFIG", str(cfg))

    redis_stub = types.SimpleNamespace(Redis=lambda: DummyRedis(), RedisError=Exception)
    monkeypatch.setitem(sys.modules, "redis", redis_stub)

    metrics_stub = types.SimpleNamespace(
        CollectorRegistry=lambda: None,
        Counter=lambda *a, **k: DummyCounter(),
        Gauge=lambda *a, **k: DummyGauge(),
    )
    monkeypatch.setitem(sys.modules, "prometheus_client", metrics_stub)

    if "datacreek.utils.cache" in sys.modules:
        del sys.modules["datacreek.utils.cache"]
    mod = importlib.import_module("datacreek.utils.cache")
    monkeypatch.setattr(mod.TTLManager, "start", lambda self: None)
    return mod


class DummyCounter:
    def __init__(self, value=0):
        self.value = value
        self._value = types.SimpleNamespace(get=lambda: self.value)

    def inc(self):
        self.value += 1


class DummyGauge:
    def __init__(self):
        self.value = None

    def set(self, v):
        self.value = v


class DummyRedis:
    def __init__(self):
        self.store = {}

    def exists(self, key):
        return key in self.store

    def get(self, key):
        return self.store[key]

    def setex(self, key, ttl, value):
        self.store[key] = value


@pytest.mark.heavy
def test_run_once_increases_and_decreases(reload_cache):
    cache = reload_cache
    cache.hits = DummyCounter(10)
    cache.miss = DummyCounter(0)
    cache.hit_ratio_g = DummyGauge()
    tm = cache.TTLManager()
    tm._target = 0.5
    tm._kp = 10.0
    tm._ki = 0.0
    tm.run_once()
    assert tm.current_ttl > 5
    cache.hits.value = 0
    cache.miss.value = 10
    before = tm.current_ttl
    tm.run_once()
    assert tm.current_ttl < before


@pytest.mark.heavy
def test_l1_cache_decorator(reload_cache):
    cache = reload_cache
    cache.hits = DummyCounter()
    cache.miss = DummyCounter()
    r = DummyRedis()
    cache.redis = r
    cache.ttl_manager = types.SimpleNamespace(current_ttl=9)

    @cache.l1_cache(lambda x: f"k:{x}")
    def compute(key, *, redis_client=None):
        return f"val-{key}"

    # first call -> miss -> store value
    assert compute("a") == "val-a"
    assert r.store["k:a"] == "val-a"
    assert cache.miss.value == 1
    # second call -> hit -> no new setex
    assert compute("a") == "val-a"
    assert cache.hits.value == 1
