import importlib
import os
import sys

import fakeredis

os.environ["DATABASE_URL"] = "sqlite:///test.db"
if "datacreek.db" in sys.modules:
    importlib.reload(sys.modules["datacreek.db"])
else:
    import datacreek.db  # pragma: no cover
if "datacreek.services" in sys.modules:
    importlib.reload(sys.modules["datacreek.services"])
from datacreek.db import SessionLocal
from datacreek.services import create_dataset, get_dataset_by_id


def test_dataset_cached(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    with SessionLocal() as db:
        ds = create_dataset(db, None, 1, path="p")
    val = redis_client.hget(f"dataset_record:{ds.id}", "path")
    val = val.decode() if isinstance(val, bytes) else val
    assert val == "p"


def test_get_dataset_by_id_uses_cache(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    with SessionLocal() as db:
        ds = create_dataset(db, None, 1, path="p")
        db.delete(db.get(type(ds), ds.id))
        db.commit()
    with SessionLocal() as db2:
        cached = get_dataset_by_id(db2, ds.id)
    assert cached is not None
    assert cached.path == "p"
