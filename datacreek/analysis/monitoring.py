"""Prometheus monitoring utilities."""

from __future__ import annotations

from typing import Dict

try:
    from prometheus_client import (
        REGISTRY,
        CollectorRegistry,
        Counter,
        Gauge,
        push_to_gateway,
        start_http_server,
    )
except Exception:  # pragma: no cover - optional
    CollectorRegistry = None
    Gauge = None
    push_to_gateway = None
    start_http_server = None

if Gauge is not None:

    def _metric(cls, name: str, desc: str, **kwargs):
        """Return a metric, reusing existing collectors when present."""
        existing = getattr(REGISTRY, "_names_to_collectors", {}).get(name)
        if existing:
            return existing
        return cls(name, desc, **kwargs)

    tpl_w1_g = _metric(Gauge, "tpl_w1", "Wasserstein-1 TPL")
    sheaf_score_g = _metric(Gauge, "sheaf_score", "Sheaf consistency score")
    sheaf_hyper_score_g = _metric(Gauge, "sheaf_hyper_score", "Sheaf-hyper coherence")
    gw_entropy = _metric(Gauge, "gw_entropy", "GraphWave entropy")
    snr_dynamic_thr = _metric(Gauge, "snr_dynamic_thr", "Adaptive SNR threshold in dB")
    autotune_cost_g = _metric(Gauge, "autotune_cost", "Current J(theta)")
    bias_wasserstein_last = _metric(
        Gauge, "bias_wasserstein_last", "Latest Wasserstein distance used"
    )
    haa_edges_total = _metric(Counter, "haa_edges_total", "Hyper-AA edges written")
    gp_jitter_restarts_total = _metric(
        Counter,
        "gp_jitter_restarts_total",
        "SVGP restarts due to jitter",
    )
    prune_reverts_total = _metric(
        Counter,
        "prune_reverts_total",
        "FractalNet pruning rollbacks",
    )
    redis_hit_ratio = _metric(Gauge, "redis_hit_ratio", "Redis L1 hit ratio")
    redis_hit_ratio_stdev = _metric(
        Gauge,
        "redis_hit_ratio_stdev",
        "Std dev of Redis hit ratio",
    )
    eigsh_timeouts_total = _metric(
        Counter,
        "eigsh_timeouts_total",
        "Number of eigsh timeouts triggering Lanczos fallback",
    )
    eigsh_last_duration = _metric(
        Gauge,
        "eigsh_last_runtime_seconds",
        "Duration of the last eigsh call in seconds",
    )
    ann_backend = _metric(Gauge, "ann_backend", "Approximate NN backend in use")
    redis_evictions_l2_total = _metric(
        Gauge,
        "redis_evictions_l2_total",
        "LMDB L2 evictions performed",
    )
    lmdb_evictions_total = _metric(
        Counter,
        "lmdb_evictions_total",
        "LMDB evictions by cause and type",
        labelnames=["cause", "type"],
    )
    lmdb_eviction_last_ts = _metric(
        Gauge,
        "lmdb_eviction_last_ts",
        "Timestamp of last LMDB eviction per cause",
        labelnames=["cause"],
    )
    lang_mismatch_total = _metric(
        Counter,
        "lang_mismatch_total",
        "Cross-language merge attempts rejected",
    )
    whisper_xrt = _metric(
        Gauge,
        "whisper_xrt",
        "Realtime factor of Whisper batch transcription",
        labelnames=["device"],
    )
    whisper_fallback_total = _metric(
        Counter,
        "whisper_fallback_total",
        "Whisper GPU fallbacks to CPU",
    )
    blip_called_total = _metric(
        Counter,
        "blip_called_total",
        "Images captioned with BLIP",
    )
    blip_skipped_total = _metric(
        Counter,
        "blip_skipped_total",
        "Images skipped due to phash deduplication",
    )
    ingest_total = _metric(
        Counter,
        "ingest_total",
        "Total ingest attempts",
    )
    ingest_validation_fail_total = _metric(
        Counter,
        "ingest_validation_fail_total",
        "Invalid ingest payloads rejected",
    )
    ingest_rate_limited_total = _metric(
        Counter,
        "ingest_rate_limited_total",
        "Ingest requests rejected due to rate limiting",
    )
    ingest_queue_fill_ratio = _metric(
        Gauge,
        "ingest_queue_fill_ratio",
        "Ratio of active ingest tasks to queue size",
    )
    breaker_state = _metric(
        Gauge,
        "breaker_state",
        "State of Neo4j circuit breaker (0=CLOSED,1=OPEN)",
    )
    pgvector_query_ms = _metric(
        Gauge,
        "pgvector_query_ms",
        "Duration of pgvector nearest-neighbour queries in milliseconds",
    )
    embedding_mmd_g = _metric(
        Gauge,
        "embedding_mmd",
        "Squared MMD between baseline and new embedding means",
    )
    j_cost = autotune_cost_g
else:  # pragma: no cover - optional dependency missing
    tpl_w1_g = None
    sheaf_score_g = None
    sheaf_hyper_score_g = None
    gw_entropy = None
    snr_dynamic_thr = None  # type: ignore
    autotune_cost_g = None
    bias_wasserstein_last = None
    haa_edges_total = None
    gp_jitter_restarts_total = None  # type: ignore
    prune_reverts_total = None  # type: ignore
    redis_hit_ratio = None  # type: ignore
    redis_hit_ratio_stdev = None  # type: ignore
    eigsh_timeouts_total = None  # type: ignore
    eigsh_last_duration = None  # type: ignore
    redis_evictions_l2_total = None  # type: ignore
    lmdb_evictions_total = None  # type: ignore
    lmdb_eviction_last_ts = None  # type: ignore
    lang_mismatch_total = None  # type: ignore
    ann_backend = None  # type: ignore
    whisper_xrt = None  # type: ignore
    whisper_fallback_total = None  # type: ignore
    blip_called_total = None  # type: ignore
    blip_skipped_total = None  # type: ignore
    ingest_total = None  # type: ignore
    ingest_validation_fail_total = None  # type: ignore
    ingest_rate_limited_total = None  # type: ignore
    ingest_queue_fill_ratio = None  # type: ignore
    breaker_state = None  # type: ignore
    pgvector_query_ms = None  # type: ignore
    embedding_mmd_g = None
    j_cost = None


_METRICS = {
    "sigma_db": None,
    "recall10": None,
    "sheaf_score": sheaf_score_g,
    "sheaf_hyper_score": sheaf_hyper_score_g,
    "gw_entropy": gw_entropy,
    "snr_dynamic_thr": snr_dynamic_thr,
    "tpl_w1": tpl_w1_g,
    "j_cost": j_cost,
    "autotune_cost": autotune_cost_g,
    "mapper_overlap_opt": None,
    "bias_wasserstein_last": bias_wasserstein_last,
    "haa_edges_total": haa_edges_total,
    "gp_jitter_restarts_total": gp_jitter_restarts_total,
    "prune_reverts_total": prune_reverts_total,
    "redis_hit_ratio": redis_hit_ratio,
    "redis_hit_ratio_stdev": redis_hit_ratio_stdev,
    "eigsh_timeouts_total": eigsh_timeouts_total,
    "eigsh_last_duration": eigsh_last_duration,
    "redis_evictions_l2_total": redis_evictions_l2_total,
    "ann_backend": ann_backend,
    "lang_mismatch_total": lang_mismatch_total,
    "whisper_xrt": whisper_xrt,
    "whisper_fallback_total": whisper_fallback_total,
    "blip_called_total": blip_called_total,
    "blip_skipped_total": blip_skipped_total,
    "ingest_total": ingest_total,
    "ingest_validation_fail_total": ingest_validation_fail_total,
    "ingest_rate_limited_total": ingest_rate_limited_total,
    "ingest_queue_fill_ratio": ingest_queue_fill_ratio,
    "breaker_state": breaker_state,
    "pgvector_query_ms": pgvector_query_ms,
    "embedding_mmd": embedding_mmd_g,
    # ingestion statistics
    "atoms_total": None,
    "avg_chunk_len": None,
    # embedding statistics
    "n2v_var_norm": None,
}

# ``start_metrics_server`` should run only once, even if this module is
# imported multiple times (e.g. by subprocesses or reloads).
_SERVER_STARTED = False


def start_metrics_server(port: int = 8000) -> None:
    """Start an HTTP server exposing Prometheus gauges.

    This function is idempotent: subsequent calls will simply return without
    altering the registry or spawning additional servers.
    """
    if start_http_server is None or Gauge is None:
        return
    global _SERVER_STARTED
    if _SERVER_STARTED:
        return
    for name, g in _METRICS.items():
        if g is None:
            _METRICS[name] = _metric(Gauge, name, f"{name} metric")
    try:
        # attempt to bind to the configured port
        start_http_server(port)
    except OSError:  # pragma: no cover - port already in use
        # fall back to an ephemeral port if the preferred one is taken
        start_http_server(0)
    _SERVER_STARTED = True


def push_metrics_gateway(
    metrics: Dict[str, float], gateway: str = "localhost:9091"
) -> None:
    """Push ``metrics`` to a Prometheus pushgateway if available."""
    if CollectorRegistry is None or Gauge is None or push_to_gateway is None:
        return
    registry = CollectorRegistry()
    for key, value in metrics.items():
        g = Gauge(key, f"{key} metric", registry=registry)
        g.set(value)
    try:
        push_to_gateway(gateway, job="datacreek", registry=registry)
    except Exception:  # pragma: no cover - network errors
        pass


def update_metric(
    name: str, value: float, labels: dict[str, str] | None = None
) -> None:
    """Update one of the default gauges.

    Parameters
    ----------
    name:
        Name of the metric to update.
    value:
        Value to set.
    labels:
        Optional label mapping for metrics that use ``labelnames``.
    """

    g = _METRICS.get(name)
    if g is not None:
        try:
            if labels:
                g.labels(**labels).set(value)
            else:
                g.set(value)
        except Exception:  # pragma: no cover
            pass


try:
    from ..utils.config import load_config
except Exception:  # pragma: no cover - optional config deps missing
    load_config = None  # type: ignore

if load_config is not None:
    cfg = load_config()
    start_metrics_server(int(cfg.get("monitor", {}).get("port", 8000)))
