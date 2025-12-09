import types

from datacreek.utils import kafka_queue


def test_get_producer_stub(monkeypatch):
    monkeypatch.setattr(kafka_queue, "KafkaProducer", None)
    kafka_queue._PRODUCER = None
    prod = kafka_queue.get_producer()
    prod.send("t", {"a": 1})
    assert prod.sent == [("t", {"a": 1})]


def test_enqueue_ingest(monkeypatch):
    sent = {}

    class Stub:
        def send(self, topic, value):
            sent["topic"] = topic
            sent["value"] = value
            return types.SimpleNamespace(get=lambda timeout=None: None)

    monkeypatch.setattr(kafka_queue, "get_producer", lambda: Stub())
    res = kafka_queue.enqueue_ingest("demo", "/tmp/x", user_id=2, extra=True)
    assert sent["topic"] == "datacreek_ingest"
    assert sent["value"]["extra"] is True
    assert hasattr(res, "get")
