import asyncio
import importlib.abc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_cache(monkeypatch):
    utils_pkg = ModuleType("datacreek.utils")
    utils_pkg.__path__ = [str(ROOT / "datacreek" / "utils")]
    config_stub = ModuleType("datacreek.utils.config")
    config_stub.load_config = lambda: {"cache": {}}
    monkeypatch.setitem(sys.modules, "datacreek.utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "datacreek.utils.config", config_stub)

    spec = importlib.util.spec_from_file_location(
        "datacreek.utils.cache", ROOT / "datacreek" / "utils" / "cache.py"
    )
    cache = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(cache)
    return cache


class DummyCounter:
    def __init__(self, value: int = 0):
        self.value = value
        self._value = SimpleNamespace(get=lambda: self.value)

    def inc(self):
        self.value += 1


class DummyGauge:
    def __init__(self):
        self.values = []

    def set(self, value):
        self.values.append(value)


@pytest.fixture()
def cache_mod(monkeypatch):
    cache = _load_cache(monkeypatch)
    cache.hits = DummyCounter(5)
    cache.miss = DummyCounter(5)
    cache.hit_ratio_g = DummyGauge()
    yield cache
    asyncio.run(cache.stop_ttl_manager_async())


def test_manager_lazy_start(cache_mod):
    mgr = cache_mod.get_ttl_manager(start=False)
    assert mgr._task is None

    cache_mod.get_ttl_manager()  # starts loop
    assert mgr._task is not None


def test_run_once_updates_metrics(cache_mod):
    mgr = cache_mod.get_ttl_manager(start=False)
    current = mgr.current_ttl
    mgr.run_once()
    assert cache_mod.hit_ratio_g.values
    assert mgr.current_ttl != current


def test_stop_helper_resets_singleton(cache_mod):
    cache_mod.get_ttl_manager()
    asyncio.run(cache_mod.stop_ttl_manager_async())
    assert cache_mod._ttl_manager is None
