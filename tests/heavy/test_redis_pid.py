import asyncio
import types

import pytest

from datacreek.utils import redis_pid


class DummyRedis:
    def __init__(self, hits=0, miss=0):
        self.hits = hits
        self.miss = miss

    async def get(self, key):
        return {"hits": self.hits, "miss": self.miss}.get(key, 0)


def _reset_state():
    redis_pid._current_ttl = 10
    redis_pid._integral_err = 0.0
    redis_pid._kp_dynamic = redis_pid.K_P
    redis_pid._err_mean = 0.0
    redis_pid._err_var = 1.0
    redis_pid._sigma_e0 = 1.0


@pytest.mark.asyncio
async def test_update_ttl(monkeypatch):
    _reset_state()
    monkeypatch.setattr(redis_pid, "update_metric", lambda *a, **k: None)
    client = DummyRedis(hits=5, miss=5)
    await redis_pid.update_ttl(client)
    assert isinstance(redis_pid.get_current_ttl(), int)
    assert isinstance(redis_pid.get_current_kp(), float)


@pytest.mark.asyncio
async def test_pid_loop(monkeypatch):
    _reset_state()
    monkeypatch.setattr(
        redis_pid, "aioredis", types.SimpleNamespace(Redis=lambda: DummyRedis())
    )

    async def fake_sleep(_):
        stop_event.set()

    monkeypatch.setattr(redis_pid.asyncio, "sleep", fake_sleep)
    stop_event = asyncio.Event()
    await redis_pid.pid_loop(None, stop_event=stop_event)


def test_gain_schedule(monkeypatch):
    _reset_state()
    monkeypatch.setattr(redis_pid, "_get_base_kp", lambda now=None: 0.3)
    monkeypatch.setattr(redis_pid, "update_metric", lambda *a, **k: None)
    asyncio.run(redis_pid.update_ttl(DummyRedis(hits=1, miss=1)))
    day_kp = redis_pid.get_current_kp()
    monkeypatch.setattr(redis_pid, "_get_base_kp", lambda now=None: 0.5)
    asyncio.run(redis_pid.update_ttl(DummyRedis(hits=1, miss=1)))
    night_kp = redis_pid.get_current_kp()
    assert night_kp != day_kp
