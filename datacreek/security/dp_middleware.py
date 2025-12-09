from __future__ import annotations

"""FastAPI middleware enforcing differential privacy budgets."""

import json
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("datacreek.privacy")

from datacreek.dp import compute_epsilon


class DPBudgetMiddleware(BaseHTTPMiddleware):
    """Check and decrement tenant privacy budgets on each request.

    The middleware expects headers ``X-Tenant`` (integer tenant id) and
    ``X-Epsilon`` (float amount). If either is missing the request is
    passed through unchanged. Privacy budget is tracked in the database
    and Renyi epsilon is computed with :func:`compute_epsilon`. Remaining
    epsilon is returned via ``X-Epsilon-Remaining`` on the response. If the
    budget is insufficient the request is rejected with HTTP 403 and the header
    indicates the available remainder.
    """

    header_tenant = "X-Tenant"
    header_epsilon = "X-Epsilon"
    alphas = [2]

    async def dispatch(self, request: Request, call_next):
        tenant = request.headers.get(self.header_tenant)
        epsilon = request.headers.get(self.header_epsilon)
        if tenant is None or epsilon is None:
            return await call_next(request)
        try:
            tid = int(tenant)
            amount = float(epsilon)
        except ValueError:
            return Response("invalid epsilon", status_code=400)

        from datacreek import db as dbmodule

        with dbmodule.SessionLocal() as db:
            entry = db.get(dbmodule.TenantPrivacy, tid)
            if entry is None:
                return Response(
                    status_code=403,
                    headers={"X-Epsilon-Remaining": "0.000000"},
                )

            current = compute_epsilon([entry.epsilon_used], alphas=self.alphas)
            new_total = compute_epsilon(
                [entry.epsilon_used + amount], alphas=self.alphas
            )
            if new_total > entry.epsilon_max:
                logger.info(
                    json.dumps({"tenant": tid, "eps_req": amount, "allowed": False})
                )
                return Response(
                    status_code=403,
                    headers={"X-Epsilon-Remaining": "0.000000"},
                )

            entry.epsilon_used = new_total
            db.commit()
            remaining = entry.epsilon_max - entry.epsilon_used

        response = await call_next(request)
        response.headers["X-Epsilon-Remaining"] = f"{remaining:.6f}"
        logger.info(json.dumps({"tenant": tid, "eps_req": amount, "allowed": True}))
        return response
