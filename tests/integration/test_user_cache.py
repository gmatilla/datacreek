import time

import fakeredis

from datacreek.db import SessionLocal, User
from datacreek.services import create_user_with_generated_key, get_user_by_key, hash_key


def test_user_cached(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    name = f"u{int(time.time()*1000)}"
    with SessionLocal() as db:
        user, key = create_user_with_generated_key(db, name)
    val = redis_client.hget(f"user:{user.id}", "username")
    val = val.decode() if isinstance(val, bytes) else val
    assert val == name
    v2 = redis_client.hget("users:keys", hash_key(key))
    v2 = v2.decode() if isinstance(v2, bytes) else v2
    assert v2 == str(user.id)


def test_get_user_by_key_uses_cache(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    name = f"u{int(time.time()*1000)}"
    with SessionLocal() as db:
        user, key = create_user_with_generated_key(db, name)
        db.delete(db.get(User, user.id))
        db.commit()
    with SessionLocal() as db2:
        found = get_user_by_key(db2, key)
    assert found is not None
    assert found.username == name
