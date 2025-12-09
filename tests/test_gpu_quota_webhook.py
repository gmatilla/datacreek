"""Tests for the GPU quota admission webhook."""

from __future__ import annotations

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from datacreek.gpu_quota_webhook import create_app, gpu_credits_left, gpu_requests_total
from metrics_prometheus.gpu_billing import QuotaController


def test_job_rejected_when_over_budget() -> None:
    """Submitting a job whose expected GPU time exceeds credits returns 403."""

    gpu_credits_left.clear()  # ensure a clean registry
    gpu_requests_total.clear()
    controller = QuotaController({"tenant-a": 50})
    app = create_app(controller)
    client = TestClient(app)

    # First submit a small job consuming 20 minutes to leave 30 credits.
    resp_ok = client.post(
        "/mutate", json={"tenant": "tenant-a", "t_epoch": 10, "n_epoch": 2}
    )
    assert resp_ok.status_code == 200
    assert resp_ok.json()["credits_left"] == 30
    # Gauge and counter should reflect the accepted request
    sample = REGISTRY.get_sample_value(
        "gpu_credits_left", labels={"tenant": "tenant-a"}
    )
    assert sample == 30
    accepted = REGISTRY.get_sample_value(
        "gpu_requests_total", labels={"tenant": "tenant-a", "status": "accepted"}
    )
    assert accepted == 1

    # Second job requires 40 minutes which exceeds remaining 30 credits
    resp = client.post(
        "/mutate", json={"tenant": "tenant-a", "t_epoch": 20, "n_epoch": 2}
    )
    assert resp.status_code == 403
    # Balance and rejected counter should update accordingly
    sample_after = REGISTRY.get_sample_value(
        "gpu_credits_left", labels={"tenant": "tenant-a"}
    )
    assert sample_after == 30
    rejected = REGISTRY.get_sample_value(
        "gpu_requests_total", labels={"tenant": "tenant-a", "status": "rejected"}
    )
    assert rejected == 1
