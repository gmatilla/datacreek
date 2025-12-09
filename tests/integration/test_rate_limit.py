import pytest

pytest.importorskip("fakeredis")
pytest.importorskip("pydantic")

import fakeredis

from datacreek.utils.rate_limit import configure, consume_token


def test_token_bucket_basic(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    configure(client=client, rate=2, burst=2)
    assert consume_token("t", client=client, now=0)
    assert consume_token("t", client=client, now=0)
    assert not consume_token("t", client=client, now=0)
    assert consume_token("t", client=client, now=1)
    assert consume_token("t", client=client, now=1)
    assert not consume_token("t", client=client, now=1)


from datacreek.analysis.monitoring import ingest_rate_limited_total
from datacreek.core.dataset import DatasetBuilder, DatasetType
from datacreek.tasks import dataset_ingest_task


def setup_fake(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    return client


def test_dataset_ingest_rate_limited(tmp_path, monkeypatch):
    client = setup_fake(monkeypatch)
    configure(client=client, rate=1, burst=1)
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")
    f = tmp_path / "doc.txt"
    f.write_text("hi")
    dataset_ingest_task.delay("demo", str(f)).get()
    start = 0.0
    if ingest_rate_limited_total is not None:
        start = ingest_rate_limited_total._value.get()
    with pytest.raises(RuntimeError):
        dataset_ingest_task.delay("demo", str(f)).get()
    if ingest_rate_limited_total is not None:
        assert ingest_rate_limited_total._value.get() == start + 1
