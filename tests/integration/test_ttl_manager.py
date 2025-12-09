import asyncio
import importlib.abc
import importlib.util
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


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


@pytest.fixture(autouse=True)
def _cleanup(monkeypatch):
    cache = _load_cache(monkeypatch)
    yield cache
    loop = asyncio.new_event_loop()
    loop.run_until_complete(cache.ttl_manager.stop())
    loop.close()


def test_ttl_adaptive(monkeypatch, _cleanup):
    cache = _cleanup
    cache.ttl_manager.current_ttl = 600
    if cache.hits is None or cache.hit_ratio_g is None:
        pytest.skip("prometheus not available")
    for _ in range(3):
        cache.hits.inc()
    time.sleep(0.1)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(cache.ttl_manager.update())
    loop.close()
    assert cache.ttl_manager.current_ttl < 600


def test_ttl_manager_async_task(_cleanup):
    cache = _cleanup

    async def run_once():
        await cache.ttl_manager.update()

    loop = asyncio.new_event_loop()
    loop.run_until_complete(run_once())
    loop.close()
    assert cache.ttl_manager.current_ttl > 0


def test_ttl_manager_stop(_cleanup):
    cache = _cleanup
    loop = asyncio.new_event_loop()
    loop.run_until_complete(cache.ttl_manager.stop())
    loop.close()
    assert cache.ttl_manager._task is None


def test_ttl_pid_convergence(monkeypatch, _cleanup):
    cache = _cleanup
    cache.ttl_manager.current_ttl = 600
    cache.ttl_manager._integral_err = 0.0

    ratios = [0.2] * 3 + [0.9] * 3 + [0.45] * 6
    for r in ratios:
        hits = int(r * 100)
        miss = 100 - hits
        cache.hits._value.set(hits)
        cache.miss._value.set(miss)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(cache.ttl_manager.update())
        loop.close()
    assert 300 <= cache.ttl_manager.current_ttl <= 600
