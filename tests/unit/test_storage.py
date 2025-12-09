import sys
import types

import fakeredis

from datacreek.storage import RedisStorage, S3Storage


class DummyS3:
    def __init__(self):
        self.calls = []

    def put_object(self, Bucket, Key, Body):
        self.calls.append((Bucket, Key, Body))


def test_redis_storage():
    client = fakeredis.FakeRedis()
    store = RedisStorage(client)
    key = store.save("k", "v")
    assert key == "k"
    assert client.get("k") == b"v"


def test_s3_storage(monkeypatch):
    s3 = DummyS3()
    if "boto3" not in sys.modules:
        sys.modules["boto3"] = types.SimpleNamespace(client=lambda *_: s3)
    else:
        monkeypatch.setattr("boto3.client", lambda *_: s3)
    store = S3Storage("bucket", "prefix")
    key = store.save("foo", "bar")
    assert key == "prefix/foo"
    assert s3.calls == [("bucket", "prefix/foo", b"bar")]
