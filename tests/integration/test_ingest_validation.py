import json

import pytest

pytest.importorskip("fakeredis")
pytest.importorskip("pydantic")
import fakeredis
from pydantic import ValidationError

pytest.importorskip("torch")

from datacreek.analysis.monitoring import ingest_total, ingest_validation_fail_total
from datacreek.core import dataset_full

DatasetBuilder = dataset_full.DatasetBuilder
DatasetType = dataset_full.DatasetType
from datacreek.tasks import dataset_ingest_task


def setup_fake(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    import datacreek.tasks as tasks_mod

    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    monkeypatch.setattr("datacreek.tasks.DatasetBuilder", dataset_full.DatasetBuilder)
    # avoid heavy PDF parsing dependencies
    import datacreek.core.ingest as ingest_mod

    orig_determine = ingest_mod.determine_parser

    def _determine(file_path, config=None):
        if str(file_path).endswith((".pdf", ".png", ".wav")):
            return type("Dummy", (), {"parse": lambda self, p, **k: ""})()
        return orig_determine(file_path, config)

    monkeypatch.setattr("datacreek.core.ingest.determine_parser", _determine)
    tasks_mod.celery_app.conf.task_always_eager = True
    return client


def test_dataset_ingest_validation_metric(tmp_path, monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    start_fail = start_total = 0.0
    if ingest_validation_fail_total is not None:
        start_fail = ingest_validation_fail_total._value.get()
    if ingest_total is not None:
        start_total = ingest_total._value.get()
    with pytest.raises(ValidationError):
        dataset_ingest_task.delay("demo", "", high_res=True).get()
    if ingest_validation_fail_total is not None:
        assert ingest_validation_fail_total._value.get() == start_fail + 1
    if ingest_total is not None:
        assert ingest_total._value.get() == start_total + 1


@pytest.mark.parametrize("kind", ["image", "audio", "pdf"])
def test_dataset_ingest_quality_gates(tmp_path, monkeypatch, kind):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)

    if kind == "image":
        from PIL import Image

        img_path = tmp_path / "small.png"
        Image.new("L", (128, 128)).save(img_path)
        with pytest.raises(ValidationError):
            dataset_ingest_task.delay("demo", str(img_path)).get()
    elif kind == "audio":
        import wave

        import numpy as np

        wav_path = tmp_path / "noise.wav"
        sr = 8000
        noise = np.random.randn(sr) * 1000
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(np.asarray(noise, dtype=np.int16).tobytes())
        with pytest.raises(ValidationError):
            dataset_ingest_task.delay("demo", str(wav_path)).get()
    else:
        pdf_path = tmp_path / "low.pdf"
        pdf_path.write_text("a" * 512)
        with pytest.raises(ValidationError):
            dataset_ingest_task.delay("demo", str(pdf_path)).get()


def test_ingest_total_metric(tmp_path, monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)

    file_path = tmp_path / "ok.txt"
    file_path.write_text("hello")

    start = ingest_total._value.get() if ingest_total is not None else 0.0
    dataset_ingest_task.delay("demo", str(file_path)).get()
    if ingest_total is not None:
        assert ingest_total._value.get() == start + 1
