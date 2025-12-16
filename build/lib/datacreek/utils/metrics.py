"""Lightweight Grafana/StatsD metrics helper."""

from __future__ import annotations

from typing import Dict


def push_metrics(
    metrics: Dict[str, float],
    *,
    prefix: str = "datacreek",
    host: str = "localhost",
    port: int = 8125,
) -> None:
    """Send ``metrics`` to a StatsD server if the client is available.

    Parameters
    ----------
    metrics:
        Mapping of metric name to value.
    prefix:
        Prefix added to every metric key. Defaults to ``"datacreek"``.
    host:
        StatsD server host name.
    port:
        StatsD UDP port.
    """

    try:
        from statsd import StatsClient  # type: ignore
    except Exception:  # pragma: no cover - optional dependency missing
        return

    client = StatsClient(host=host, port=port, prefix=prefix)
    for key, value in metrics.items():
        try:
            client.gauge(key, value)
        except Exception:  # pragma: no cover - network errors
            continue
