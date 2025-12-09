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
async def test_update_ttl_decrease(monkeypatch):
    _reset_state()
    monkeypatch.setattr(redis_pid, "update_metric", lambda *a, **k: None)
    client = DummyRedis(hits=10, miss=0)
    await redis_pid.update_ttl(client)
    assert redis_pid.get_current_ttl() < 10
    assert redis_pid._integral_err < 0


@pytest.mark.asyncio
async def test_update_ttl_increase(monkeypatch):
    """TTL should increase when hit ratio below target."""
    _reset_state()
    monkeypatch.setattr(redis_pid, "update_metric", lambda *a, **k: None)
    monkeypatch.setattr(redis_pid, "TARGET_HIT", 0.8, raising=False)
    monkeypatch.setattr(redis_pid, "_get_base_kp", lambda now=None: 1.5)
    monkeypatch.setattr(redis_pid, "K_I", 0.1, raising=False)
    monkeypatch.setattr(redis_pid, "INTERVAL", 1, raising=False)
    client = DummyRedis(hits=0, miss=10)
    await redis_pid.update_ttl(client)
    assert redis_pid.get_current_ttl() > 10
    assert redis_pid._integral_err > 0


def test_get_set_ttl():
    redis_pid.set_current_ttl(123)
    assert redis_pid.get_current_ttl() == 123
    assert isinstance(redis_pid.get_current_kp(), float)


@pytest.mark.asyncio
async def test_pid_loop_and_start(monkeypatch):
    """pid_loop should call update_ttl and start_pid_controller must pass the stop event."""
    _reset_state()
    # make sure the loop runs even without real redis package
    monkeypatch.setattr(
        redis_pid,
        "aioredis",
        types.SimpleNamespace(Redis=lambda: DummyRedis(hits=5, miss=5)),
    )

    calls = []
    orig_sleep = redis_pid.asyncio.sleep

    async def fake_sleep(_):
        calls.append("sleep")
        stop.set()

    monkeypatch.setattr(redis_pid.asyncio, "sleep", fake_sleep)
    stop = asyncio.Event()
    await redis_pid.pid_loop(None, stop_event=stop)
    assert calls
    monkeypatch.setattr(redis_pid.asyncio, "sleep", orig_sleep)

    run = {}

    async def fake_pid(client=None, stop_event=None):
        run["stop"] = stop_event

    monkeypatch.setattr(redis_pid, "pid_loop", fake_pid)
    task = redis_pid.start_pid_controller(DummyRedis())
    await asyncio.sleep(0)
    task.stop_event.set()
    await task
    assert run["stop"] is task.stop_event


@pytest.mark.asyncio
async def test_update_ttl_error(monkeypatch):
    """update_ttl should bail out if Redis access fails."""
    _reset_state()

    async def fail_get(key):
        raise RuntimeError

    monkeypatch.setattr(redis_pid, "update_metric", lambda *a, **k: None)
    client = types.SimpleNamespace(get=fail_get)
    await redis_pid.update_ttl(client)
    assert redis_pid.get_current_ttl() == 10


@pytest.mark.asyncio
async def test_update_ttl_clamping(monkeypatch):
    """TTL updates should remain within safe bounds."""
    _reset_state()
    redis_pid._current_ttl = 1
    monkeypatch.setattr(redis_pid, "update_metric", lambda *a, **k: None)
    client = DummyRedis(hits=10, miss=0)
    await redis_pid.update_ttl(client)
    assert redis_pid.get_current_ttl() >= 1
    redis_pid._current_ttl = 86400
    client = DummyRedis(hits=0, miss=10)
    await redis_pid.update_ttl(client)
    assert redis_pid.get_current_ttl() <= 86400


@pytest.mark.asyncio
async def test_pid_loop_errors(monkeypatch):
    """pid_loop should exit early when dependencies fail."""
    _reset_state()
    calls = []
    monkeypatch.setattr(redis_pid, "aioredis", None)
    await redis_pid.pid_loop(None, stop_event=asyncio.Event())


def test_gain_schedule(monkeypatch):
    _reset_state()
    monkeypatch.setattr(redis_pid, "_get_base_kp", lambda now=None: 0.3)
    monkeypatch.setattr(redis_pid, "update_metric", lambda *a, **k: None)
    redis_pid._err_var = 0.25
    asyncio.run(redis_pid.update_ttl(DummyRedis(hits=5, miss=5)))
    day_kp = redis_pid.get_current_kp()
    monkeypatch.setattr(redis_pid, "_get_base_kp", lambda now=None: 0.5)
    asyncio.run(redis_pid.update_ttl(DummyRedis(hits=5, miss=5)))
    night_kp = redis_pid.get_current_kp()
    assert day_kp != night_kp
