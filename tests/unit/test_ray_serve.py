import asyncio
import sys
import types
from unittest.mock import AsyncMock, patch

from starlette.requests import Request

# Stub a minimal ``ray`` module so ``serving.ray_serve`` can be imported without
# installing the real Ray dependency. Tests patch the exposed ``serve``
# namespace to simulate deployment behaviour.
serve_ns = types.SimpleNamespace()


def _decorator(obj=None, **_kwargs):
    """Return the class/function unchanged; mimics ``serve.deployment``."""

    def wrap(cls):
        return cls

    return wrap(obj) if obj else wrap


serve_ns.deployment = _decorator
serve_ns.get_deployment_handle = lambda name: None
serve_ns.run = lambda *a, **k: None
serve_ns.delete_deployment = lambda name: None

ray_stub = types.SimpleNamespace(serve=serve_ns)
sys.modules.setdefault("ray", ray_stub)

from serving.ray_serve import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ModelDeployment,
    TenantRouter,
    _maybe_rollback,
    _record_latency,
    _reset_metrics,
    deploy_model,
)


def make_request(tenant: str, model_version: str) -> Request:
    """Create a Starlette request with routing headers."""
    scope = {
        "type": "http",
        "headers": [
            (b"x-tenant", tenant.encode()),
            (b"x-model-version", model_version.encode()),
        ],
    }
    return Request(scope, receive=lambda: None)


def test_model_deployment_echoes_headers():
    request = make_request("acme", "v1")
    deployment = ModelDeployment("acme", "v1")
    result = asyncio.run(deployment(request))
    assert result["deployment"] == "acme-v1"
    assert result["tenant_header"] == "acme"
    assert result["model_header"] == "v1"


def test_router_uses_headers_to_lookup_deployment():
    request = make_request("acme", "v1")
    router = TenantRouter()
    mock_handle = AsyncMock()
    mock_handle.remote.return_value = {"ok": True}
    with patch("serving.ray_serve.serve.get_deployment_handle") as get_handle:
        get_handle.return_value = mock_handle
        result = asyncio.run(router(request))
    get_handle.assert_called_once_with("acme-v1")
    mock_handle.remote.assert_awaited_once_with(request)
    assert result == {"ok": True}


def test_deploy_model_invokes_serve_run_with_name():
    with patch("serving.ray_serve.serve.run") as run:
        with patch.object(
            __import__(
                "serving.ray_serve", fromlist=["ModelDeployment"]
            ).ModelDeployment,
            "bind",
            return_value=None,
            create=True,
        ):
            deploy_model("acme", "v1")
    run.assert_called_once()
    _, kwargs = run.call_args
    assert kwargs["name"] == "acme-v1"
    assert kwargs["route_prefix"] is None


def test_router_routes_small_portion_to_canary():
    """Random value below threshold should select canary deployment."""

    request = make_request("acme", "prod")
    router = TenantRouter()

    prod_handle = AsyncMock()
    canary_handle = AsyncMock()

    def get_handle(name: str):
        return canary_handle if name.endswith("canary") else prod_handle

    with patch("serving.ray_serve.serve.get_deployment_handle", side_effect=get_handle):
        with patch("serving.ray_serve.random.random", return_value=0.01):
            asyncio.run(router(request))

    canary_handle.remote.assert_awaited_once()
    prod_handle.remote.assert_not_awaited()


def test_maybe_rollback_triggers_when_canary_slower():
    """Canary is removed when p99 latency is double production."""

    _reset_metrics()
    for _ in range(100):
        _record_latency("acme-prod", 0.1)
    for _ in range(100):
        _record_latency("acme-canary", 0.3)

    with patch("serving.ray_serve.serve.delete_deployment") as delete:
        assert _maybe_rollback("acme")
        delete.assert_called_once_with("acme-canary")


def test_prometheus_metrics_increment_on_request():
    """Router increments Prometheus counters and histograms."""

    _reset_metrics()
    request = make_request("acme", "v1")
    router = TenantRouter()
    mock_handle = AsyncMock()
    mock_handle.remote.return_value = {"ok": True}

    with patch("serving.ray_serve.serve.get_deployment_handle") as get_handle:
        get_handle.return_value = mock_handle
        asyncio.run(router(request))

    assert REQUEST_COUNT.labels(tenant="acme", model_version="v1")._value.get() == 1
    assert REQUEST_LATENCY.labels(tenant="acme", model_version="v1")._sum.get() > 0
