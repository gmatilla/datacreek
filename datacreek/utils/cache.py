"""L1 cache utilities with Prometheus counters."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from functools import wraps
from threading import Thread
from typing import Callable, Optional

try:  # optional dependency
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional
    redis = None  # type: ignore

try:  # optional Prometheus metrics
    from prometheus_client import CollectorRegistry, Counter, Gauge
except Exception:  # pragma: no cover - optional
    Counter = None  # type: ignore
    Gauge = None  # type: ignore

from .config import load_config

cfg = load_config()
cache_cfg = cfg.get("cache", {})

if Counter is not None and Gauge is not None:
    _REGISTRY = CollectorRegistry()
    hits = Counter("redis_hits_total", "Redis L1 hits", registry=_REGISTRY)
    miss = Counter("redis_miss_total", "Redis L1 misses", registry=_REGISTRY)
    hit_ratio_g = Gauge("redis_hit_ratio", "L1 hit ratio", registry=_REGISTRY)
else:  # pragma: no cover - metrics disabled
    hits = miss = hit_ratio_g = None  # type: ignore


def cache_l1(func: Callable) -> Callable:
    """Decorator injecting a default Redis client if available."""

    @wraps(func)
    def wrapper(
        key: str,
        *args,
        redis_client: Optional["redis.Redis"] = None,
        **kwargs,
    ):
        client = redis_client
        if client is None and redis is not None:
            try:
                client = redis.Redis()
            except Exception:  # pragma: no cover - connection errors
                client = None
        result = func(key, *args, redis_client=client, **kwargs)
        return result

    return wrapper


class TTLManager:
    """Asynchronous manager adjusting TTL from hit ratio."""

    def __init__(self) -> None:  # pragma: no cover - init starts background thread
        self.current_ttl = int(cache_cfg.get("l1_ttl_init", 3600))
        pid_cfg = cache_cfg.get("ttl_pid", {})
        self._target = float(pid_cfg.get("target_hit_ratio", 0.8))
        self._kp = float(pid_cfg.get("Kp", 500.0))
        self._ki = float(pid_cfg.get("Ki", 0.05))
        self._i_max = float(pid_cfg.get("I_max", 3600))
        self._integral = 0.0
        self._alpha = 0.3
        self._ema = 0.5
        self._task: asyncio.Task | None = None
        self._stop: Optional[asyncio.Event] = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._thread: Thread | None = None
        self.start()

    async def _loop(self) -> None:  # pragma: no cover - background
        assert self._stop is not None
        while not self._stop.is_set():
            await asyncio.sleep(300)
            self.run_once()

    def start(self) -> None:  # pragma: no cover - background thread
        """Launch the TTL adjustment loop in an asyncio task."""

        if self._task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            self._event_loop = loop

            def _runner() -> None:
                asyncio.set_event_loop(loop)
                self._stop = asyncio.Event()
                self._task = loop.create_task(self._loop())
                loop.run_forever()
                loop.close()

            self._thread = Thread(target=_runner, daemon=True)
            self._thread.start()
        else:
            self._stop = asyncio.Event()
            self._event_loop = loop
            self._task = loop.create_task(self._loop())

    async def stop(self) -> None:  # pragma: no cover - background thread
        """Stop the background loop and wait for completion."""

        if self._task is None:
            return
        assert self._stop is not None
        self.run_once()  # flush last update
        if (
            self._event_loop is not None
            and self._event_loop.is_running()
            and self._event_loop is not asyncio.get_event_loop()
        ):
            self._event_loop.call_soon_threadsafe(self._stop.set)
            self._event_loop.call_soon_threadsafe(self._task.cancel)
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=1)
                self._thread = None
        else:
            self._stop.set()
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(self._task, timeout=1)
        self._task = None

    def run_once(self) -> None:
        """Update TTL based on PID controller for hit ratio."""

        if hits is None or miss is None or hit_ratio_g is None:
            return
        h = hits._value.get()
        m = miss._value.get()
        ratio = h / max(1, h + m)
        # exponential moving average for smoother control
        self._ema = self._alpha * ratio + (1 - self._alpha) * self._ema
        hit_ratio_g.set(self._ema)
        ttl_min = int(cache_cfg.get("l1_ttl_min", 300))
        ttl_max = int(cache_cfg.get("l1_ttl_max", 7200))
        error = self._ema - self._target
        # anti-windup clamp on integral term
        self._integral = max(
            -self._i_max,
            min(self._i_max, self._integral + error * 300),
        )
        delta = self._kp * error + self._ki * self._integral
        # apply bounded update
        self.current_ttl = max(
            ttl_min,
            min(ttl_max, int(self.current_ttl + delta)),
        )


ttl_manager = TTLManager()


def l1_cache(key_fn: Callable[..., str]) -> Callable[[Callable], Callable]:
    """Simple Redis memoization with adaptive TTL."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if redis is not None:
                key = key_fn(*args, **kwargs)
                try:
                    if redis.exists(key):
                        if hits is not None:
                            hits.inc()
                        return redis.get(key)
                except redis.RedisError:
                    logging.getLogger(__name__).warning(
                        "Redis error on get", exc_info=True
                    )
            result = fn(*args, **kwargs)
            if redis is not None:
                try:
                    if miss is not None:
                        miss.inc()
                    redis.setex(key, ttl_manager.current_ttl, result)
                except redis.RedisError:
                    logging.getLogger(__name__).warning(
                        "Redis error on set", exc_info=True
                    )
            return result

        return wrapper

    return decorator
