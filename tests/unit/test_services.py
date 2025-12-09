import types

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from datacreek import services
from datacreek.db import Base, Dataset, SourceData, User


class FakeRedis:
    def __init__(self):
        self.store = {}

    def hset(self, name, key=None, value=None, mapping=None):
        if mapping is not None:
            self.store.setdefault(name, {}).update(mapping)
        else:
            self.store.setdefault(name, {})[key] = value

    def hget(self, name, key):
        return self.store.get(name, {}).get(key)

    def hgetall(self, name):
        return self.store.get(name, {})


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def fake_redis(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr(services, "get_redis_client", lambda: r)
    return r


def test_user_cache_and_retrieval(db_session, fake_redis):
    user = services.create_user(db_session, "alice", "key123", password="pw")
    # Should be cached in redis under hashed key
    cached = fake_redis.hgetall(f"user:{user.id}")
    assert cached["username"] == "alice"
    # Retrieval via api key hits redis and returns User instance
    found = services.get_user_by_key(db_session, "key123")
    assert found.id == user.id
    assert found.username == "alice"


def test_user_db_fallback(db_session, monkeypatch):
    # disable redis
    monkeypatch.setattr(services, "get_redis_client", lambda: None)
    user = services.create_user(db_session, "bob", "k2")
    # fetch via DB when cache unavailable
    found = services.get_user_by_key(db_session, "k2")
    assert found.id == user.id


def test_dataset_cache_and_retrieval(db_session, fake_redis):
    src = services.create_source(db_session, None, "f", "data")
    ds = services.create_dataset(db_session, None, src.id, path="p")
    cached = fake_redis.hgetall(f"dataset_record:{ds.id}")
    assert cached["path"] == "p"
    found = services.get_dataset_by_id(db_session, ds.id)
    assert found.id == ds.id
    assert found.path == "p"


def test_dataset_db_fallback(db_session, monkeypatch):
    monkeypatch.setattr(services, "get_redis_client", lambda: None)
    src = services.create_source(db_session, None, "f2", "c")
    ds = services.create_dataset(db_session, None, src.id)
    found = services.get_dataset_by_id(db_session, ds.id)
    assert found.id == ds.id


def test_generated_key_helper(db_session, fake_redis):
    user, key = services.create_user_with_generated_key(db_session, "charlie")
    assert len(key) == 32
    # verify hash stored
    assert services.hash_key(key) == user.api_key
