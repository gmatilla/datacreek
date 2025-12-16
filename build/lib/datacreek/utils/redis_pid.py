from __future__ import annotations

"""Asynchronous PID controller for Redis TTL.

This module adjusts the L1 cache TTL using a PI controller to
maintain a target hit ratio. It relies on Redis counters ``hits``
and ``miss`` updated elsewhere in the application.
"""

import asyncio
import logging
from datetime import datetime
from math import sqrt
from typing import Optional

from datacreek.analysis.monitoring import redis_hit_ratio_stdev, update_metric

try:  # optional dependency
    import redis.asyncio as aioredis  # type: ignore
except Exception:  # pragma: no cover - optional
    aioredis = None  # type: ignore

try:
    from .config import load_config
except Exception:  # pragma: no cover - optional dependency missing
    load_config = None  # type: ignore

if load_config is not None:
    cfg = load_config()
    cache_cfg = cfg.get("cache", {})
    pid_cfg = cfg.get("pid", {})
else:  # basic defaults for tests when config unavailable
    cache_cfg = {}
    pid_cfg = {}

# Controller parameters (fallback to legacy cache keys)
TARGET_HIT = float(pid_cfg.get("target_hit_ratio", cache_cfg.get("pid_target", 0.45)))
K_P_DAY = float(pid_cfg.get("Kp_day", 0.3))
K_P_NIGHT = float(pid_cfg.get("Kp_night", 0.5))
K_P = float(pid_cfg.get("Kp", K_P_DAY if 8 <= datetime.now().hour < 20 else K_P_NIGHT))
K_I = float(pid_cfg.get("Ki", cache_cfg.get("pid_ki", 0.05)))
# Maximum absolute value for the integral term (anti-windup)
I_MAX = float(pid_cfg.get("I_max", cache_cfg.get("pid_i_max", 5)))
INTERVAL = float(cache_cfg.get("pid_interval", 60.0))
_TTL_MIN = int(cache_cfg.get("l1_ttl_min", 300))
_TTL_MAX = int(cache_cfg.get("l1_ttl_max", 7200))
_current_ttl = int(cache_cfg.get("l1_ttl_init", 3600))

_integral_err = 0.0
_kp_dynamic = K_P
_err_mean = 0.0
_err_var = 1.0  # variance of hit-ratio overshoot used for Kalman gain
_sigma_e0 = 1.0
_Q = float(pid_cfg.get("kalman_q", 1e-4))
_R = float(pid_cfg.get("kalman_r", 1e-2))


def _get_base_kp(now: datetime | None = None) -> float:
    """Return gain based on time-of-day (day vs night)."""

    ref = now or datetime.now()
    return K_P_DAY if 8 <= ref.hour < 20 else K_P_NIGHT


async def update_ttl(client: "aioredis.Redis") -> None:
    """Single PID update step."""

    global _current_ttl, _integral_err
    try:
        hits = int(await client.get("hits") or 0)
        miss = int(await client.get("miss") or 0)
    except Exception:  # pragma: no cover - network
        return
    total = hits + miss
    ratio = hits / total if total else 0.0
    err = TARGET_HIT - ratio

    global _err_mean, _err_var, _kp_dynamic, _sigma_e0
    # Kalman filter on the error signal to estimate its variance
    pred_var = _err_var + _Q
    k = pred_var / (pred_var + _R)
    _err_mean = _err_mean + k * (err - _err_mean)
    _err_var = (1 - k) * pred_var
    sigma_e = sqrt(_err_var)
    if _sigma_e0 == 1.0 and sigma_e > 0:
        _sigma_e0 = sigma_e
    base_kp = _get_base_kp()
    _kp_dynamic = base_kp * (sigma_e / (_sigma_e0 or 1.0))
    if redis_hit_ratio_stdev is not None:
        try:
            update_metric("redis_hit_ratio_stdev", sigma_e)
        except Exception:  # pragma: no cover - metric update errors
            pass

    # Anti-windup on the integral term
    _integral_err = max(-I_MAX, min(I_MAX, _integral_err + err * INTERVAL))
    # Discrete PID control law with adaptive gain
    delta = _kp_dynamic * err + K_I * _integral_err
    # Clamp result to a safe TTL range [1s, 24h]
    new_ttl = max(1, min(86400, int(_current_ttl + delta)))
    if new_ttl != _current_ttl:
        logging.getLogger(__name__).debug("PID TTL updated to %d", new_ttl)
        _current_ttl = new_ttl


def get_current_ttl() -> int:
    """Return the current TTL managed by the PID controller."""

    return _current_ttl


def get_current_kp() -> float:
    """Return the current proportional gain."""

    return _kp_dynamic


def set_current_ttl(value: int) -> None:
    """Set the TTL value, useful for tests."""

    global _current_ttl
    _current_ttl = value


async def pid_loop(
    client: Optional["aioredis.Redis"] = None,
    *,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """Run PID updates periodically in an asyncio task."""

    if aioredis is None:  # pragma: no cover - missing dependency
        return
    if client is None:
        try:
            client = aioredis.Redis()
        except Exception:  # pragma: no cover - connection
            return
    while stop_event is None or not stop_event.is_set():
        await asyncio.sleep(INTERVAL)
        await update_ttl(client)


def start_pid_controller(client: Optional["aioredis.Redis"] = None) -> asyncio.Task:
    """Launch :func:`pid_loop` and return the task."""

    event = asyncio.Event()
    task = asyncio.create_task(pid_loop(client, stop_event=event))
    task.stop_event = event  # type: ignore[attr-defined]
    return task
