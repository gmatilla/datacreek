from __future__ import annotations

"""Vector search API router."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from datacreek.db import SessionLocal

from .explain_router import _load_dataset  # reuse helper

router = APIRouter(prefix="/vector", tags=["vector"])


class VectorSearchRequest(BaseModel):
    """Request body for the vector search endpoint."""

    dataset: str = Field(..., json_schema_extra={"examples": ["demo"]})
    query: str = Field(..., json_schema_extra={"examples": ["graph"]})
    k: int = Field(5, ge=1, le=50, json_schema_extra={"examples": [5]})
    node_type: str = Field(
        "chunk",
        json_schema_extra={"examples": ["chunk"]},
    )




@router.post("/search", summary="Search dataset using hybrid vector search")
def vector_search(
    payload: VectorSearchRequest,
) -> JSONResponse:
    """Return node IDs relevant to ``query``.

    Results combine lexical and embedding-based matches for the dataset.

    Example
    -------
    ``curl``::

        curl -X POST -H "X-API-Key: <token>" -H "Content-Type: application/json" \
            -d '{"dataset": "demo", "query": "graph"}' \
            "http://localhost:8000/vector/search"

    JavaScript ``fetch``::

        fetch("/vector/search", {
            method: "POST",
            headers: {"X-API-Key": "<token>", "Content-Type": "application/json"},
            body: JSON.stringify({dataset: "demo", query: "graph"})
        }).then(r => r.json());
    """

    ds = _load_dataset(payload.dataset)
    ids = ds.search_hybrid(payload.query, k=payload.k, node_type=payload.node_type)
    return JSONResponse(ids)
