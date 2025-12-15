"""Simple edge review UI and backend integration.

This module exposes a small :mod:`fastapi` application that lists edges with a
large change in the eigenvalue (``Δλ``) and lets a curator accept or reject
those edges.  Decisions are written back to Neo4j via :class:`Neo4jFabricClient`
and recorded in a durable log so auditors can trace every action.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from .neo4j_fabric import Neo4jFabricClient

# Threshold above which an edge requires human review.
TAU = 0.1

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
LOGGER = logging.getLogger(__name__)

STORAGE_DIR = Path(
    os.getenv("EDGE_REVIEW_STORAGE", Path(__file__).resolve().parent / ".." / "cache")
).resolve()
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
EDGES_FILE = STORAGE_DIR / "edges.json"
PATCHSET_FILE = STORAGE_DIR / "patchsets.json"
LOG_FILE = STORAGE_DIR / "edge_review.log"

def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        LOGGER.exception("Failed to read %s", path.name)
        return default

def _write_json(path: Path, data):
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        LOGGER.exception("Failed to persist %s", path.name)

_EDGES_LOCK = threading.RLock()
_PATCHSET_LOCK = threading.RLock()
_PATCHSET_REGISTRY: Dict[str, Dict[str, str]] = {}
_LOG_LOCK = threading.RLock()


def require_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> None:
    expected = os.getenv("EDGE_REVIEW_API_KEY")
    if not expected or api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


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


def _load_edges() -> List[Edge]:
    with _EDGES_LOCK:
        payload = _read_json(EDGES_FILE, [])
        edges = [Edge(**data) for data in payload if isinstance(data, dict)]
        EDGES.clear()
        EDGES.extend(edges)
        return list(EDGES)


def _persist_edges(edges: List[Edge]) -> None:
    with _EDGES_LOCK:
        _write_json(EDGES_FILE, [edge.dict() for edge in edges])
        EDGES.clear()
        EDGES.extend(edges)


def _append_log(entry: Dict[str, str]) -> None:
    with _LOG_LOCK:
        log = _read_json(LOG_FILE, [])
        log.append(entry)
        _write_json(LOG_FILE, log)
        EDGE_REPAIR_LOG.append(entry)


def _update_patchset(patchset_id: str, edge_id: int, action: str) -> None:
    with _PATCHSET_LOCK:
        current = _read_json(PATCHSET_FILE, {})
        patch = current.setdefault(patchset_id, {})
        patch[str(edge_id)] = action
        _write_json(PATCHSET_FILE, current)
        _PATCHSET_REGISTRY[patchset_id] = patch


class Decision(BaseModel):
    """Payload for accepting or rejecting an edge."""

    tenant: str
    action: Literal["accept", "reject"]


def create_app(
    client: Neo4jFabricClient,
    tau: float = TAU,
    *,
    page_size: int = 20,
) -> FastAPI:
    """Return a FastAPI app exposing the edge review UI.

    Parameters
    ----------
    client:
        Neo4j client used to persist curator decisions.
    tau:
        Threshold on ``Δλ`` above which an edge is shown for review.
    """

    app = FastAPI()

    @app.get(
        "/ui/edge_review",
        response_model=List[Edge],
        dependencies=[Depends(require_api_key)],
    )
    def list_edges(
        limit: int = Query(page_size, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> List[Edge]:
        """Return edges whose ``Δλ`` exceeds ``tau``."""

        edges = [e for e in _load_edges() if e.delta_lambda > tau]
        return edges[offset : offset + limit]

    @app.patch(
        "/ui/edge_review/{edge_id}",
        dependencies=[Depends(require_api_key)],
    )
    def patch_edge(
        edge_id: int,
        decision: Decision,
        patchset_id: str | None = Query(None),
    ) -> Dict[str, str]:
        """Accept or reject ``edge_id`` and record the action.

        The change is written back to Neo4j and logged in
        ``EDGE_REPAIR_LOG`` with a UTC timestamp so auditors can track
        modifications performed by curators.
        """

        for edge in _load_edges():
            if edge.id == edge_id:
                try:
                    client.run(
                        decision.tenant,
                        "MATCH ()-[e]->() WHERE id(e)=$id SET e.status=$status",
                        id=edge_id,
                        status=decision.action,
                    )
                except Exception as exc:
                    LOGGER.exception("Neo4j update failed for edge %s", edge_id)
                    raise HTTPException(
                        status_code=500, detail="Neo4j unavailable"
                    ) from exc
                entry = {
                    "edge_id": str(edge_id),
                    "action": decision.action,
                    "ts": datetime.utcnow().isoformat(),
                }
                _append_log(entry)
                if patchset_id:
                    _update_patchset(patchset_id, edge_id, decision.action)
                return {"status": "ok"}

        raise HTTPException(status_code=404, detail="edge not found")

    @app.get(
        "/ui/edge_review/log",
        dependencies=[Depends(require_api_key)],
    )
    def get_log(limit: int = Query(100, ge=1, le=500)) -> List[Dict[str, str]]:
        """Return the version history of curator decisions."""

        return EDGE_REPAIR_LOG[-limit:]

    return app


_PATCHSET_REGISTRY.update(_read_json(PATCHSET_FILE, {}))
PATCHSET_REGISTRY: Dict[str, Dict[str, str]] = _PATCHSET_REGISTRY
EDGE_REPAIR_LOG.extend(_read_json(LOG_FILE, []))
_load_edges()
