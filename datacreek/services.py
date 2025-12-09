import json
import secrets
from hashlib import sha256

from sqlalchemy.orm import Session

try:
    from werkzeug.security import generate_password_hash
except Exception:  # pragma: no cover - optional dependency

    def generate_password_hash(p: str) -> str:
        """Fallback when ``werkzeug`` isn't installed."""
        import hashlib

        return hashlib.sha256(p.encode()).hexdigest()


from datacreek.backends import get_redis_client
from datacreek.db import Dataset, SourceData, User


def hash_key(api_key: str) -> str:
    return sha256(api_key.encode()).hexdigest()


def generate_api_key() -> str:
    """Return a new random API key."""
    return secrets.token_hex(16)


def _cache_user(user: User) -> None:
    """Persist user information in Redis for quick lookup."""

    client = get_redis_client()
    if not client:
        return
    try:
        client.hset(
            f"user:{user.id}",
            mapping={
                "username": user.username,
                "api_key": user.api_key,
                "password_hash": user.password_hash,
            },
        )
        client.hset("users:keys", user.api_key, user.id)
    except Exception:
        # caching is best-effort
        pass


def _cache_dataset(ds: Dataset) -> None:
    """Persist dataset metadata in Redis for quick lookup."""

    client = get_redis_client()
    if not client:
        return
    try:
        client.hset(
            f"dataset_record:{ds.id}",
            mapping={
                "owner_id": ds.owner_id if ds.owner_id is not None else "",
                "source_id": ds.source_id,
                "path": ds.path,
                "content": ds.content or "",
            },
        )
    except Exception:
        # caching is best-effort
        pass


def get_user_by_key(db: Session, api_key: str) -> User | None:
    hashed = hash_key(api_key)
    client = get_redis_client()
    if client:
        user_id = client.hget("users:keys", hashed)
        if user_id:
            if isinstance(user_id, bytes):
                user_id = user_id.decode()
            data = client.hgetall(f"user:{user_id}")
            if data:
                uname = data.get("username") or data.get(b"username")
                if isinstance(uname, bytes):
                    uname = uname.decode()
                api_key_val = data.get("api_key") or data.get(b"api_key")
                if isinstance(api_key_val, bytes):
                    api_key_val = api_key_val.decode()
                pw = data.get("password_hash") or data.get(b"password_hash")
                if isinstance(pw, bytes):
                    pw = pw.decode()
                return User(
                    id=int(user_id),
                    username=uname or "",
                    api_key=api_key_val or "",
                    password_hash=pw or "",
                )
    return db.query(User).filter_by(api_key=hashed).first()


def get_dataset_by_id(db: Session, ds_id: int) -> Dataset | None:
    """Return dataset record using Redis cache when available."""

    client = get_redis_client()
    if client:
        data = client.hgetall(f"dataset_record:{ds_id}")
        if data:
            owner = data.get("owner_id") or data.get(b"owner_id")
            if isinstance(owner, bytes):
                owner = owner.decode()
            owner_id = int(owner) if owner else None
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
                owner_id=owner_id,
                source_id=source_id,
                path=path or "",
                content=content or None,
            )
    ds = db.get(Dataset, ds_id)
    if ds:
        _cache_dataset(ds)
    return ds


def create_user(
    db: Session, username: str, api_key: str, password: str | None = None
) -> User:
    user = User(
        username=username,
        api_key=hash_key(api_key),
        password_hash=generate_password_hash(password or ""),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    _cache_user(user)
    return user


def create_user_with_generated_key(
    db: Session, username: str, password: str | None = None
) -> tuple[User, str]:
    """Create a user and return the record along with the plain API key."""
    api_key = generate_api_key()
    user = create_user(db, username, api_key, password=password)
    return user, api_key


def create_source(
    db: Session,
    owner_id: int | None,
    path: str,
    content: str,
    *,
    entities: list[str] | None = None,
    facts: list[dict[str, str]] | None = None,
) -> SourceData:
    src = SourceData(
        owner_id=owner_id,
        path=path,
        content=content,
        entities=json.dumps(entities) if entities else None,
        facts=json.dumps(facts) if facts else None,
    )
    db.add(src)
    db.commit()
    db.refresh(src)
    return src


def create_dataset(
    db: Session,
    owner_id: int | None,
    source_id: int,
    *,
    path: str | None = None,
    content: str | None = None,
) -> Dataset:
    ds = Dataset(
        owner_id=owner_id, source_id=source_id, path=path or "", content=content
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    _cache_dataset(ds)
    return ds
