"""Tenant-aware model serving powered by Ray Serve.

This module defines a minimal Ray Serve router and deployment that use
HTTP headers to route requests to the appropriate tenant/model version
combination. Each deployment is named using the pattern
``"{tenant}-{model_version}"`` to ensure uniqueness per tenant. A
lightweight canary mechanism is also provided: when a request targets
the production version (``X-Model-Version: prod``) the router diverts
5 % of traffic to a ``canary`` deployment and records the latency of
each call. If the canary's p99 latency exceeds twice that of the
production deployment it is automatically rolled back.

Example
-------
>>> from serving.ray_serve import TenantRouter, deploy_model
>>> import ray
>>> from ray import serve
>>> ray.init()
>>> serve.start(http_options={"port": 8001})
>>> TenantRouter.deploy()
>>> deploy_model("acme", "v1")
>>> # Requests with the appropriate headers will now be routed.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict

from prometheus_client import Counter, Histogram
from ray import serve
from starlette.requests import Request

_latencies: defaultdict[str, list[float]] = defaultdict(list)

# Prometheus metrics tracking request counts and latency per tenant and model
# version. These gauges allow operators to monitor production traffic and
# correlate latency spikes with rollback events.
REQUEST_COUNT = Counter(
    "serve_requests_total",
    "Total number of requests handled by tenant and model version.",
    ["tenant", "model_version"],
)
REQUEST_LATENCY = Histogram(
    "serve_request_latency_seconds",
    "Latency of requests handled by tenant and model version.",
    ["tenant", "model_version"],
)


def _record_latency(deployment: str, duration: float) -> None:
    """Store latency for ``deployment`` and update Prometheus metrics.

    Parameters
    ----------
    deployment:
        Name of the Ray Serve deployment in the form ``"tenant-version"``.
    duration:
        Observed latency in seconds for the handled request.
    """

    _latencies[deployment].append(duration)
    tenant, model_version = deployment.split("-", 1)
    REQUEST_COUNT.labels(tenant=tenant, model_version=model_version).inc()
    REQUEST_LATENCY.labels(tenant=tenant, model_version=model_version).observe(duration)


def _p99(samples: list[float]) -> float:
    """Return the 99th percentile latency from ``samples``.

    The implementation avoids heavy dependencies like NumPy by sorting
    the observed durations and selecting the element at the 99th
    percentile index.
    """

    if not samples:
        return 0.0
    samples = sorted(samples)
    index = max(int(len(samples) * 0.99) - 1, 0)
    return samples[index]


def _maybe_rollback(tenant: str) -> bool:
    """Delete the canary deployment if its p99 > 2 × prod p99.

    Returns ``True`` when a rollback was triggered.
    """

    prod = _p99(_latencies[f"{tenant}-prod"])
    canary = _p99(_latencies[f"{tenant}-canary"])
    if prod and canary and canary > 2 * prod:
        serve.delete_deployment(f"{tenant}-canary")
        return True
    return False


def _reset_metrics() -> None:
    """Clear recorded metrics (used in tests)."""

    _latencies.clear()
    REQUEST_COUNT.clear()
    REQUEST_LATENCY.clear()


@serve.deployment
class ModelDeployment:
    """Ray Serve deployment for a specific tenant's model.

    Parameters
    ----------
    tenant:
        Identifier of the tenant.
    model_version:
        Version string of the model.
    """

    def __init__(self, tenant: str, model_version: str) -> None:
        self.tenant = tenant
        self.model_version = model_version

    async def __call__(self, request: Request) -> dict:
        """Return deployment name and echoed routing headers."""
        return {
            "deployment": f"{self.tenant}-{self.model_version}",
            "tenant_header": request.headers.get("X-Tenant"),
            "model_header": request.headers.get("X-Model-Version"),
        }


@serve.deployment
class TenantRouter:
    """Route requests to tenant/model deployments based on headers."""

    async def __call__(self, request: Request):
        tenant = request.headers.get("X-Tenant")
        model_version = request.headers.get("X-Model-Version")
        if not tenant or not model_version:
            return {"error": "missing headers"}

        # Divert a small fraction of ``prod`` traffic to the canary deployment
        # using non-cryptographic randomness. Other model versions are routed
        # directly to their matching deployment.
        target_version = model_version
        if model_version == "prod" and random.random() < 0.05:  # nosec B311
            target_version = "canary"

        deployment_name = f"{tenant}-{target_version}"
        handle = serve.get_deployment_handle(deployment_name)

        start = time.perf_counter()
        result = await handle.remote(request)
        duration = time.perf_counter() - start
        _record_latency(deployment_name, duration)
        _maybe_rollback(tenant)
        return result


def deploy_model(tenant: str, model_version: str) -> None:
    """Deploy a tenant/model pair with the proper naming convention.

    Parameters
    ----------
    tenant:
        Identifier of the tenant.
    model_version:
        Version of the model to expose.
    """

    deployment_name = f"{tenant}-{model_version}"
    serve.run(
        ModelDeployment.bind(tenant, model_version),
        name=deployment_name,
        route_prefix=None,
    )


def start_router() -> None:
    """Run the router at the root HTTP path."""
    serve.run(TenantRouter.bind(), route_prefix="/")
