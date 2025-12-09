"""Admission webhook enforcing per-tenant GPU credit quotas.

This module exposes a small :mod:`fastapi` application that estimates the
GPU time required for a training job and rejects the request if the tenant lacks
sufficient credits.  Remaining credits are exported via a Prometheus gauge so
external billing systems can scrape the balance.

The admission controller computes the expected GPU minutes using

.. math::

    E_{gpu} = t_{epoch} \cdot n_{epoch}

where ``t_epoch`` is the duration in minutes of a single epoch and ``n_epoch``
is the number of epochs in the submitted job.  When ``E_gpu`` exceeds the
available credits the webhook responds with HTTP ``403`` to signal that the job
should not be scheduled.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge
from pydantic import BaseModel

from metrics_prometheus.gpu_billing import QuotaController, QuotaExceededError

# Prometheus gauge tracking the remaining GPU credits for each tenant.  The
# value follows ``credits_left = credits_0 - \int gpu\_minutes\,dt``.
gpu_credits_left = Gauge(
    "gpu_credits_left", "Remaining GPU credits for a tenant", ["tenant"]
)

# Counter tracking total admission requests and whether they were accepted or
# rejected.  The ``status`` label records ``accepted`` or ``rejected`` to
# simplify monitoring dashboards.
gpu_requests_total = Counter(
    "gpu_requests_total",
    "GPU admission requests by tenant and status",
    ["tenant", "status"],
)


class JobSpec(BaseModel):
    """Specification of the training job submitted by a tenant."""

    tenant: str
    t_epoch: float  # minutes per epoch
    n_epoch: int


def create_app(controller: QuotaController | None = None) -> FastAPI:
    """Return a FastAPI app enforcing GPU credit quotas.

    Parameters
    ----------
    controller:
        Instance managing the credit balance for each tenant.  A default empty
        controller is created when omitted which grants no credits.
    """

    app = FastAPI()
    qc = controller or QuotaController({})

    @app.post("/mutate")
    def mutate(job: JobSpec) -> dict[str, float]:
        """Approve or reject the submitted job based on remaining credits.

        The expected GPU time ``E_gpu`` is computed as ``t_epoch * n_epoch``.  If
        the tenant lacks sufficient credits an HTTP 403 error is raised;
        otherwise credits are decremented and the remaining balance is returned.
        """

        e_gpu = job.t_epoch * job.n_epoch
        remaining = qc.get_remaining(job.tenant)
        if remaining < e_gpu:
            gpu_requests_total.labels(job.tenant, "rejected").inc()
            raise HTTPException(status_code=403, detail="insufficient credits")
        try:
            new_balance = qc.consume(job.tenant, e_gpu)
        except QuotaExceededError as exc:  # pragma: no cover - defensive
            gpu_requests_total.labels(job.tenant, "rejected").inc()
            # Consume may still raise if credits changed concurrently; surface
            # the error as HTTP 403 to match admission semantics.
            raise HTTPException(status_code=403, detail=str(exc))
        gpu_requests_total.labels(job.tenant, "accepted").inc()
        gpu_credits_left.labels(tenant=job.tenant).set(new_balance)
        return {"credits_left": new_balance}

    return app
