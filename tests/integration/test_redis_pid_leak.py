import asyncio
import importlib.abc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

# Import redis_pid in isolation with a stub config
ROOT = Path(__file__).resolve().parents[1]


def _load_module(monkeypatch):
    utils_pkg = ModuleType("datacreek.utils")
    utils_pkg.__path__ = [str(ROOT / "datacreek" / "utils")]
    config_stub = ModuleType("datacreek.utils.config")
    config_stub.load_config = lambda: {"cache": {}, "pid": {}}
    monkeypatch.setitem(sys.modules, "datacreek.utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "datacreek.utils.config", config_stub)

    spec = importlib.util.spec_from_file_location(
        "datacreek.utils.redis_pid", ROOT / "datacreek" / "utils" / "redis_pid.py"
    )
    redis_pid = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(redis_pid)
    return redis_pid


@pytest.mark.filterwarnings("error")
def test_pid_loop_stops_clean(monkeypatch):
    redis_pid = _load_module(monkeypatch)
    calls = 0

    async def fake_update(client):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(redis_pid, "update_ttl", fake_update)
    monkeypatch.setattr(redis_pid, "INTERVAL", 0.01)
    monkeypatch.setattr(redis_pid, "aioredis", SimpleNamespace(Redis=lambda: None))

    async def run():
        event = asyncio.Event()
        task = asyncio.create_task(redis_pid.pid_loop(stop_event=event))
        await asyncio.sleep(0.03)
        event.set()
        await asyncio.wait_for(task, timeout=1)
        # ensure the background task exited and no other tasks remain
        pending = [
            t
            for t in asyncio.all_tasks()
            if not t.done() and t is not asyncio.current_task()
        ]
        assert not pending

    asyncio.run(run())
    assert calls >= 2
