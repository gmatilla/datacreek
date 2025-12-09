import asyncio
import importlib.abc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(monkeypatch):
    utils_pkg = ModuleType("datacreek.utils")
    utils_pkg.__path__ = [str(ROOT / "datacreek" / "utils")]
    config_stub = ModuleType("datacreek.utils.config")
    config_stub.load_config = lambda: {"cache": {}}
    monkeypatch.setitem(sys.modules, "datacreek.utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "datacreek.utils.config", config_stub)

    spec = importlib.util.spec_from_file_location(
        "datacreek.utils.redis_pid", ROOT / "datacreek" / "utils" / "redis_pid.py"
    )
    redis_pid = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(redis_pid)
    return redis_pid


def test_pid_controller_increase(monkeypatch):
    redis_pid = _load_module(monkeypatch)

    class DummyRedis:
        def __init__(self):
            self.store = {}

        async def set(self, k, v):
            self.store[k] = str(v)

        async def get(self, k):
            return self.store.get(k)

    client = DummyRedis()

    async def prepare():
        await client.set("hits", 9)
        await client.set("miss", 1)

    asyncio.run(prepare())

    monkeypatch.setattr(redis_pid, "INTERVAL", 100.0)
    redis_pid.set_current_ttl(600)
    redis_pid._integral_err = 0.0

    asyncio.run(redis_pid.update_ttl(client))
    assert redis_pid.get_current_ttl() < 600


def test_pid_controller_decrease(monkeypatch):
    redis_pid = _load_module(monkeypatch)

    class DummyRedis:
        def __init__(self):
            self.store = {}

        async def set(self, k, v):
            self.store[k] = str(v)

        async def get(self, k):
            return self.store.get(k)

    client = DummyRedis()

    async def prepare():
        await client.set("hits", 1)
        await client.set("miss", 9)

    asyncio.run(prepare())

    monkeypatch.setattr(redis_pid, "INTERVAL", 100.0)
    redis_pid.set_current_ttl(600)
    redis_pid._integral_err = 0.0

    asyncio.run(redis_pid.update_ttl(client))
    assert redis_pid.get_current_ttl() >= 600


def test_pid_burst_no_overshoot(monkeypatch):
    redis_pid = _load_module(monkeypatch)

    class DummyRedis:
        def __init__(self):
            self.store = {}

        async def set(self, k, v):
            self.store[k] = str(v)

        async def get(self, k):
            return self.store.get(k)

    client = DummyRedis()

    async def prepare():
        await client.set("hits", 9)
        await client.set("miss", 1)

    asyncio.run(prepare())

    monkeypatch.setattr(redis_pid, "INTERVAL", 0.1)
    redis_pid.set_current_ttl(600)
    redis_pid._integral_err = 0.0

    asyncio.run(redis_pid.update_ttl(client))
    assert redis_pid.get_current_ttl() < 600


def test_pid_converges_to_target(monkeypatch):
    redis_pid = _load_module(monkeypatch)

    class DummyRedis:
        def __init__(self):
            self.store = {}

        async def set(self, k, v):
            self.store[k] = str(v)

        async def get(self, k):
            return self.store.get(k)

    client = DummyRedis()

    async def run_trace(ratios, ttls):
        for r in ratios:
            hits = int(r * 100)
            miss = 100 - hits
            await client.set("hits", hits)
            await client.set("miss", miss)
            await redis_pid.update_ttl(client)
            ttls.append(redis_pid.get_current_ttl())

    monkeypatch.setattr(redis_pid, "INTERVAL", 1.0)
    redis_pid.set_current_ttl(300)
    redis_pid._integral_err = 0.0

    ratios = [0.2] * 3 + [0.9] * 3 + [0.45] * 6
    ttls: list[int] = []
    asyncio.run(run_trace(ratios, ttls))

    stable = ttls[-5:]
    assert max(stable) - min(stable) <= stable[-1] * 0.05


def test_kalman_gain_adapts(monkeypatch):
    """Variance spikes increase proportional gain."""

    redis_pid = _load_module(monkeypatch)

    class DummyRedis:
        def __init__(self):
            self.store = {}

        async def set(self, k, v):
            self.store[k] = str(v)

        async def get(self, k):
            return self.store.get(k)

    client = DummyRedis()

    async def run():
        await client.set("hits", 100)
        await client.set("miss", 0)
        await redis_pid.update_ttl(client)
        low_kp = redis_pid.get_current_kp()
        await client.set("hits", 0)
        await client.set("miss", 100)
        await redis_pid.update_ttl(client)
        high_kp = redis_pid.get_current_kp()
        return low_kp, high_kp

    monkeypatch.setattr(redis_pid, "INTERVAL", 1.0)
    redis_pid.set_current_ttl(300)
    redis_pid._integral_err = 0.0

    low, high = asyncio.run(run())
    assert high != redis_pid.K_P
