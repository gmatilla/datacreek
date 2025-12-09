import types

from datacreek.utils import kafka_queue as kq


def test_get_producer_stub(monkeypatch):
    monkeypatch.setattr(kq, "KafkaProducer", None)
    kq._PRODUCER = None
    prod = kq.get_producer()
    prod.send("t", b"d")
    assert prod.sent == [("t", b"d")]


def test_enqueue_ingest(monkeypatch):
    sent = {}

    class Stub:
        def send(self, topic, value):
            sent["topic"] = topic
            sent["value"] = value
            return types.SimpleNamespace(get=lambda timeout=None: "ok")

    monkeypatch.setattr(kq, "get_producer", lambda: Stub())
    res = kq.enqueue_ingest("n", "/p", user_id=1, extra=True)
    assert sent["value"]["extra"]
    assert res.get() == "ok"
