"""Simple edge review UI and backend integration.

This module exposes a small :mod:`fastapi` application that lists edges with a
large change in the eigenvalue (``Δλ``) and lets a curator accept or reject
those edges.  Decisions are written back to Neo4j via :class:`Neo4jFabricClient`
and recorded in an in-memory ``edge_repair_log`` for auditability.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .neo4j_fabric import Neo4jFabricClient

# Threshold above which an edge requires human review.
TAU = 0.1


class Edge(BaseModel):
    """Edge candidate flagged for review."""

    id: int
    src: str
    dst: str
    delta_lambda: float


# In-memory stores used by the review UI.  They are intentionally simple
# so unit tests can interact with them without external services.
EDGES: List[Edge] = []
EDGE_REPAIR_LOG: List[Dict[str, str]] = []


class Decision(BaseModel):
    """Payload for accepting or rejecting an edge."""

    tenant: str
    action: str  # "accept" or "reject"


def create_app(client: Neo4jFabricClient, tau: float = TAU) -> FastAPI:
    """Return a FastAPI app exposing the edge review UI.

    Parameters
    ----------
    client:
        Neo4j client used to persist curator decisions.
    tau:
        Threshold on ``Δλ`` above which an edge is shown for review.
    """

    app = FastAPI()

    @app.get("/ui/edge_review", response_model=List[Edge])
    def list_edges() -> List[Edge]:
        """Return edges whose ``Δλ`` exceeds ``tau``."""

        return [e for e in EDGES if e.delta_lambda > tau]

    @app.patch("/ui/edge_review/{edge_id}")
    def patch_edge(edge_id: int, decision: Decision) -> Dict[str, str]:
        """Accept or reject ``edge_id`` and record the action.

        The change is written back to Neo4j and logged in
        ``EDGE_REPAIR_LOG`` with a UTC timestamp so auditors can track
        modifications performed by curators.
        """

        for edge in EDGES:
            if edge.id == edge_id:
                client.run(
                    decision.tenant,
                    "MATCH ()-[e]->() WHERE id(e)=$id SET e.status=$status",
                    id=edge_id,
                    status=decision.action,
                )
                EDGE_REPAIR_LOG.append(
                    {
                        "edge_id": str(edge_id),
                        "action": decision.action,
                        "ts": datetime.utcnow().isoformat(),
                    }
                )
                return {"status": "ok"}

        raise HTTPException(status_code=404, detail="edge not found")

    @app.get("/ui/edge_review/log")
    def get_log() -> List[Dict[str, str]]:
        """Return the version history of curator decisions."""

        return EDGE_REPAIR_LOG

    return app
