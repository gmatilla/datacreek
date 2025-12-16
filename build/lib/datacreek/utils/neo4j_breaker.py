"""Integrate a circuit breaker with Prometheus metrics."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Callable

from datacreek.analysis import monitoring

try:  # pragma: no cover - import guard exercised in environments without pybreaker
    import pybreaker
except Exception:  # pragma: no cover - same as above
    pybreaker = None

logger = logging.getLogger(__name__)

# When imported in isolation, tests may register a minimal ``datacreek`` module
# without ``__path__`` preventing further imports. Ensure the package behaves
# like a namespace package.
pkg = sys.modules.get("datacreek")
if pkg is not None and not getattr(pkg, "__path__", None):
    from pathlib import Path

    pkg.__path__ = [str(Path(__file__).resolve().parents[1])]

__all__ = ["neo4j_breaker", "CircuitBreakerError", "reconfigure"]

if pybreaker is not None:
    CircuitBreakerError = pybreaker.CircuitBreakerError
else:
    class CircuitBreakerError(RuntimeError):
        """Raised when the fallback breaker refuses a call."""


_fail_max = int(os.getenv("NEO4J_CB_FAIL_MAX", "5"))
_reset_timeout = int(os.getenv("NEO4J_CB_TIMEOUT", "30"))


def _set_metric(value: int) -> None:
    """Update the Prometheus metric if the registry is available."""
    try:  # pragma: no cover - metrics optional
        monitoring.update_metric("breaker_state", value)
    except Exception:
        pass


if pybreaker is not None:

    class _PrometheusListener(pybreaker.CircuitBreakerListener):
        """Update Prometheus gauge whenever the circuit state changes."""

        def state_change(self, cb, old_state, new_state):
            name = getattr(new_state, "name", "").lower()
            _set_metric(0 if name == "closed" else 1)

    neo4j_breaker = pybreaker.CircuitBreaker(
        fail_max=_fail_max,
        reset_timeout=_reset_timeout,
        name="neo4j",
        listeners=[_PrometheusListener()],
    )
else:

    class _PrometheusListener:
        """Compat shim so the public API stays consistent."""

        def state_change(self, cb, old_state, new_state):
            _set_metric(0)

    class _NoopCircuitBreaker:
        """Lightweight replacement when pybreaker is unavailable."""

        def __init__(self, fail_max: int, reset_timeout: int):
            self.fail_max = fail_max
            self.reset_timeout = reset_timeout
            self.current_state = 0
            logger.warning(  # pragma: no cover - log only happens without pybreaker
                "pybreaker is not installed; Neo4j circuit breaker degraded to no-op"
            )

        def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        def close(self) -> None:
            self.current_state = 0
            _set_metric(0)

        def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
            def _wrapper(*args: Any, **kwargs: Any) -> Any:
                return self.call(func, *args, **kwargs)

            return _wrapper

    neo4j_breaker = _NoopCircuitBreaker(_fail_max, _reset_timeout)

_set_metric(0)


def reconfigure(fail_max: int | None = None, timeout: int | None = None) -> None:
    """Adjust breaker parameters (testing)."""
    if fail_max is not None:
        neo4j_breaker.fail_max = fail_max
    if timeout is not None:
        neo4j_breaker.reset_timeout = timeout
    neo4j_breaker.close()
    _set_metric(0)
