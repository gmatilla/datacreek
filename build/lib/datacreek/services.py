import json
import secrets
import logging
from hashlib import sha256

from sqlalchemy.orm import Session

from datacreek.backends import get_redis_client
from datacreek.db import Dataset, SourceData


def _ensure_str(value: str | None, name: str, *, allow_blank: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not allow_blank and not value.strip():
        raise ValueError(f"{name} must not be empty or whitespace")
    return value


def _persist(db: Session, obj: Dataset | SourceData) -> Dataset | SourceData:
    try:
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj
    except Exception:
        db.rollback()
        LOGGER.exception("Failed to persist %s", obj.__class__.__name__)
        raise


LOGGER = logging.getLogger(__name__)


def _cache_dataset(ds: Dataset) -> None:
    """Persist dataset metadata in Redis for quick lookup."""

    client = get_redis_client()
    if not client:
        return
    try:
        client.hset(
            f"dataset_record:{ds.id}",
            mapping={
                "source_id": ds.source_id,
                "path": ds.path,
                "content": ds.content or "",
            },
        )
    except Exception:
        # caching is best-effort
        pass


def get_dataset_by_id(db: Session, ds_id: int) -> Dataset | None:
    """Return dataset record using Redis cache when available."""

    client = get_redis_client()
    if client:
        data = client.hgetall(f"dataset_record:{ds_id}")
        if data:
            src = data.get("source_id") or data.get(b"source_id")
            if isinstance(src, bytes):
                src = src.decode()
            source_id = int(src) if src else 0
            path = data.get("path") or data.get(b"path")
            if isinstance(path, bytes):
                path = path.decode()
            content = data.get("content") or data.get(b"content")
            if isinstance(content, bytes):
                content = content.decode()
            return Dataset(
                id=ds_id,
                source_id=source_id,
                path=path or "",
                content=content or None,
            )
    ds = db.get(Dataset, ds_id)
    if ds:
        _cache_dataset(ds)
    return ds


def create_source(
    db: Session,
    path: str,
    content: str,
    *,
    entities: list[str] | None = None,
    facts: list[dict[str, str]] | None = None,
) -> SourceData:
    _ensure_str(path, "path")
    _ensure_str(content, "content", allow_blank=True)
    src = SourceData(
        path=path,
        content=content,
        entities=json.dumps(entities) if entities else None,
        facts=json.dumps(facts) if facts else None,
    )
    return _persist(db, src)


def create_dataset(
    db: Session,
    source_id: int,
    *,
    path: str | None = None,
    content: str | None = None,
) -> Dataset:
    _ensure_positive(source_id, "source_id")
    if path is not None:
        _ensure_str(path, "path", allow_blank=True)
    if content is not None:
        _ensure_str(content, "content", allow_blank=True)
    ds = Dataset(
        source_id=source_id,
        path=path or "",
        content=content,
    )
    persisted = _persist(db, ds)
    _cache_dataset(persisted)
    return persisted


def _ensure_positive(value: int, name: str) -> int:
    if value is None or value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value
