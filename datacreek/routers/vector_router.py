from __future__ import annotations

"""Vector search API router."""

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from datacreek.db import SessionLocal, User
from datacreek.services import get_user_by_key

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


def get_current_user(api_key: str = Header(..., alias="X-API-Key")) -> User:
    """Authenticate using the API key stored in the database or Redis."""
    with SessionLocal() as db:
        user = get_user_by_key(db, api_key)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return user


@router.post("/search", summary="Search dataset using hybrid vector search")
def vector_search(
    payload: VectorSearchRequest,
    user: User = Depends(get_current_user),
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

    ds = _load_dataset(payload.dataset, user)
    ids = ds.search_hybrid(payload.query, k=payload.k, node_type=payload.node_type)
    return JSONResponse(ids)
