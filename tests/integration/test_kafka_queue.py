import types

import pytest

from datacreek.utils import kafka_queue


def test_get_producer_config(monkeypatch):
    created = {}

    class FakeProducer:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def send(self, topic, value):
            return types.SimpleNamespace()

    monkeypatch.setattr(kafka_queue, "KafkaProducer", FakeProducer)
    kafka_queue._PRODUCER = None
    producer = kafka_queue.get_producer()
    assert producer is not None
    assert created["acks"] == "all"
    assert created["compression_type"] == "lz4"
    assert created["enable_idempotence"] is True
    assert created["transactional_id"] == "datacreek_ingest"


def test_enqueue_ingest(monkeypatch):
    sent = {}

    class Stub:
        def send(self, topic, value):
            sent["topic"] = topic
            sent["value"] = value
            return types.SimpleNamespace(get=lambda timeout=None: None)

    monkeypatch.setattr(kafka_queue, "get_producer", lambda: Stub())
    res = kafka_queue.enqueue_ingest("demo", "/tmp/x", user_id=1, extra=True)
    assert sent["topic"] == "datacreek_ingest"
    assert sent["value"]["extra"]
    assert hasattr(res, "get")


def test_get_consumer_config(monkeypatch):
    created = {}

    class FakeConsumer:
        def __init__(self, *topics, **kwargs):
            created.update(kwargs)

        def __iter__(self):
            return iter([])

    monkeypatch.setattr(kafka_queue, "KafkaConsumer", FakeConsumer)
    kafka_queue._CONSUMER = None
    consumer = kafka_queue.get_consumer("demo")
    assert consumer is not None
    assert created["enable_auto_commit"] is False
    assert created["isolation_level"] == "read_committed"


def test_process_messages(monkeypatch):
    msgs = [types.SimpleNamespace(value=i) for i in range(3)]

    class Stub:
        def __iter__(self):
            return iter(msgs)

        def commit(self):
            commits.append(True)

    handled = []
    commits = []

    kafka_queue.process_messages(Stub(), lambda v: handled.append(v))
    assert handled == [0, 1, 2]
    assert len(commits) == 3


def test_process_messages_transaction(monkeypatch):
    msgs = [
        types.SimpleNamespace(topic="t", partition=0, offset=i, value=i)
        for i in range(2)
    ]

    class CStub:
        config = {"group_id": "g"}

        def __iter__(self):
            return iter(msgs)

        def commit(self):
            raise AssertionError("no commit")

    class PStub:
        def __init__(self):
            self.steps = []

        def begin_transaction(self):
            self.steps.append("begin")

        def send_offsets_to_transaction(self, offsets, group_id):
            self.steps.append(("off", offsets, group_id))

        def commit_transaction(self):
            self.steps.append("commit")

    p = PStub()
    handled = []
    kafka_queue.process_messages(CStub(), lambda v: handled.append(v), producer=p)
    assert handled == [0, 1]
    assert p.steps.count("begin") == 2
    assert p.steps.count("commit") == 2
    assert ("off", {("t", 0): 1}, "g") in p.steps
