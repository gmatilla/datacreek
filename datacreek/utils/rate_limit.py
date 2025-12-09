"""Tenant-level token bucket rate limiter using Redis.

The default configuration enforces a refill rate ``r=100`` messages per
second with a bucket capacity ``C=500``. Limits may be overridden via the
``INGEST_RATE`` and ``INGEST_BURST`` environment variables.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    redis = None

__all__ = ["configure", "consume_token"]

# Default refill rate and capacity (messages per second / bucket size)
_RATE = int(os.getenv("INGEST_RATE", "100"))
_BURST = int(os.getenv("INGEST_BURST", "500"))
_CLIENT: Optional["redis.Redis"] = None
_SCRIPT_SHA: Optional[str] = None
_LOCAL_BUCKETS: Dict[str, Tuple[float, int]] = {}

# Lua script implementing token bucket
_SCRIPT = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now = tonumber(ARGV[3]) or redis.call('time')[1]
local data = redis.call('hmget', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if not tokens then tokens = burst end
if not ts then ts = now end
local delta = now - ts
if delta < 0 then delta = 0 end
local avail = tokens + delta * rate
if avail > burst then avail = burst end
if avail < 1 then
  redis.call('hmset', key, 'tokens', avail, 'ts', now)
  redis.call('expire', key, math.ceil(burst / rate))
  return 0
end
avail = avail - 1
redis.call('hmset', key, 'tokens', avail, 'ts', now)
redis.call('expire', key, math.ceil(burst / rate))
return 1
"""


def _get_client() -> Optional["redis.Redis"]:
    if redis is None:  # pragma: no cover - dependency missing
        return None
    global _CLIENT
    if _CLIENT is None:
        try:
            _CLIENT = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost"))
        except Exception:  # pragma: no cover - connection errors
            return None
    return _CLIENT


def configure(
    *,
    client: Optional["redis.Redis"] = None,
    rate: int | None = None,
    burst: int | None = None,
) -> None:
    """Configure token bucket parameters and Redis client."""
    global _CLIENT, _SCRIPT_SHA, _RATE, _BURST
    if client is not None:
        _CLIENT = client
    if rate is not None:
        _RATE = rate
    if burst is not None:
        _BURST = burst
    _LOCAL_BUCKETS.clear()
    cli = _get_client()
    if cli is not None:
        try:
            _SCRIPT_SHA = cli.script_load(_SCRIPT)
        except Exception:  # pragma: no cover - connection errors
            _SCRIPT_SHA = None


def consume_token(
    tenant: str, *, now: Optional[int] = None, client: Optional["redis.Redis"] = None
) -> bool:
    """Return ``True`` if ``tenant`` may proceed with ingestion."""
    cli = client or _get_client()
    if now is None:
        now = int(time.time())

    def _consume_local() -> bool:
        tokens, ts = _LOCAL_BUCKETS.get(tenant, (_BURST, now))
        delta = now - ts
        avail = min(tokens + delta * _RATE, _BURST)
        if avail < 1:
            _LOCAL_BUCKETS[tenant] = (avail, now)
            return False
        _LOCAL_BUCKETS[tenant] = (avail - 1, now)
        return True

    if cli is None or _SCRIPT_SHA is None:  # pragma: no cover - no redis
        return _consume_local()
    try:
        res = cli.evalsha(_SCRIPT_SHA, 1, f"tb:{tenant}", _RATE, _BURST, now)
    except redis.exceptions.NoScriptError:  # pragma: no cover - reload script
        sha = cli.script_load(_SCRIPT)
        res = cli.evalsha(sha, 1, f"tb:{tenant}", _RATE, _BURST, now)
    except Exception:  # pragma: no cover - connection errors
        return _consume_local()
    return bool(res)
