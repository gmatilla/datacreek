"""Kafka producer/consumer utilities with exactly-once semantics."""

import json
import os
from typing import Any

try:  # optional kafka dependency
    from kafka import KafkaConsumer, KafkaProducer
except Exception:  # pragma: no cover - when kafka not available
    KafkaProducer = None  # type: ignore
    KafkaConsumer = None  # type: ignore

__all__ = [
    "get_producer",
    "get_consumer",
    "enqueue_ingest",
    "process_messages",
]

_PRODUCER = None
_CONSUMER = None


def get_producer() -> Any:
    """Return a Kafka producer configured for ingestion.

    The producer uses exactly-once semantics with ``enable_idempotence=True``
    and a ``transactional_id``. Compression is ``lz4`` and acks are set to
    ``'all'``. When the ``kafka`` package is missing, a lightweight stub is
    returned that simply records messages in memory for tests.
    """

    global _PRODUCER
    if _PRODUCER is not None:
        return _PRODUCER
    if KafkaProducer is None:
        # very small stub used during tests when kafka-python is missing
        class _Stub:
            def __init__(self):
                self.sent = []

            def send(self, topic: str, value: bytes):
                self.sent.append((topic, value))

        _PRODUCER = _Stub()
        return _PRODUCER
    servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
    transactional_id = os.getenv("KAFKA_TRANSACTION_ID", "datacreek_ingest")
    _PRODUCER = KafkaProducer(  # pragma: no cover - heavy dependency
        bootstrap_servers=servers,  # pragma: no cover - heavy dependency
        acks="all",  # pragma: no cover - heavy dependency
        compression_type="lz4",  # pragma: no cover - heavy dependency
        enable_idempotence=True,  # pragma: no cover - heavy dependency
        transactional_id=transactional_id,  # pragma: no cover - heavy dependency
        value_serializer=lambda v: json.dumps(v).encode(
            "utf-8"
        ),  # pragma: no cover - heavy
    )
    return _PRODUCER


def enqueue_ingest(
    name: str, path: str, user_id: int | None = None, **kwargs: Any
) -> Any:
    """Send an ingestion request to the Kafka topic.

    The topic defaults to ``datacreek_ingest`` unless ``KAFKA_INGEST_TOPIC`` is
    set. The returned object mimics the :class:`kafka.producer.future.Future`
    interface minimally with a ``get`` method returning ``None``.
    """

    topic = os.getenv("KAFKA_INGEST_TOPIC", "datacreek_ingest")
    producer = get_producer()
    payload = {"name": name, "path": path, "user_id": user_id}
    payload.update(kwargs)
    future = producer.send(topic, payload)

    # Ensure the returned object has a ``get`` method for compatibility
    class _Result:
        def __init__(self, f: Any):
            self._f = f
            self.id = None  # Celery-style attribute

        def get(self, timeout: float | None = None) -> Any:
            if hasattr(self._f, "get"):
                return self._f.get(timeout=timeout)
            return None

    return _Result(future)


def get_consumer(topic: str | None = None) -> Any:
    """Return a Kafka consumer for ingestion events.

    The consumer is configured for manual offset commits and read-committed
    isolation level. When ``KAFKA_TRANSACTION_ID`` is set, the value is attached
    as ``transactional_id`` on the resulting object so that offsets can be
    committed within producer transactions. A lightweight stub is returned when
    ``kafka`` is missing.
    """

    global _CONSUMER
    if _CONSUMER is not None:
        return _CONSUMER
    if KafkaConsumer is None:

        class _Stub:
            def __init__(self):
                self.subscribed = []
                self.commits = 0
                self.transactional_id = os.getenv(
                    "KAFKA_TRANSACTION_ID", "datacreek_ingest"
                )
                self.config = {
                    "group_id": os.getenv("KAFKA_CONSUMER_GROUP", "datacreek"),
                    "transactional_id": self.transactional_id,
                }

            def subscribe(self, topics):
                self.subscribed.extend(topics)

            def __iter__(self):
                return iter([])

            def commit(self):
                self.commits += 1

        _CONSUMER = _Stub()
        if topic:
            _CONSUMER.subscribe([topic])
        return _CONSUMER

    servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
    group_id = os.getenv("KAFKA_CONSUMER_GROUP", "datacreek")
    transactional_id = os.getenv("KAFKA_TRANSACTION_ID", "datacreek_ingest")
    if topic is None:
        topic = os.getenv("KAFKA_INGEST_TOPIC", "datacreek_ingest")
    _CONSUMER = KafkaConsumer(
        topic,  # pragma: no cover - heavy
        bootstrap_servers=servers,  # pragma: no cover - heavy
        enable_auto_commit=False,  # pragma: no cover - heavy
        group_id=group_id,  # pragma: no cover - heavy
        isolation_level="read_committed",  # pragma: no cover - heavy
        value_deserializer=lambda b: json.loads(
            b.decode("utf-8")
        ),  # pragma: no cover - heavy
    )
    # Attach config details for transactional offset commits
    setattr(
        _CONSUMER,
        "config",
        {"group_id": group_id, "transactional_id": transactional_id},
    )
    return _CONSUMER


def process_messages(
    consumer: Any, handler: callable, *, producer: Any | None = None
) -> None:
    """Consume messages and commit offsets only after successful handling.

    When ``producer`` is provided and exposes transactional helpers,
    offsets are sent to the transaction before committing. This enables
    exactly-once semantics when producing results to Kafka and merging
    them into the graph.
    """

    for msg in consumer:
        if producer and hasattr(producer, "begin_transaction"):
            producer.begin_transaction()
        handler(msg.value)
        if producer and hasattr(producer, "send_offsets_to_transaction"):
            offsets = {(msg.topic, msg.partition): msg.offset + 1}
            group = getattr(consumer, "config", {}).get("group_id")
            producer.send_offsets_to_transaction(offsets, group)
            if hasattr(producer, "commit_transaction"):
                producer.commit_transaction()
        elif hasattr(consumer, "commit"):
            consumer.commit()
