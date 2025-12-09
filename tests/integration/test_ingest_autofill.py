import json
import math
import wave

import fakeredis
import numpy as np
import pytest
from PIL import Image

from datacreek.core import dataset_full

DatasetBuilder = dataset_full.DatasetBuilder
DatasetType = dataset_full.DatasetType
from datacreek.tasks import dataset_ingest_task


def _make_grid(path):
    img = Image.new("L", (300, 300))
    for i in range(300):
        for j in range(300):
            img.putpixel((i, j), 0 if (i + j) % 2 == 0 else 255)
    img.save(path)


def setup_fake(monkeypatch):
    import datacreek.tasks as tasks_mod

    client = fakeredis.FakeStrictRedis()
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


@pytest.mark.parametrize("kind", ["image", "audio", "pdf"])
def test_ingest_autofill_metrics(tmp_path, monkeypatch, kind):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)

    if kind == "image":
        img_path = tmp_path / "img.png"
        _make_grid(img_path)
        dataset_ingest_task.delay("demo", str(img_path)).get()
        params = json.loads(client.hget("dataset:demo:progress", "ingestion_params"))
        assert params["width"] == 300
        assert params["height"] == 300
        assert 0 <= params["blur_score"] <= 1
    elif kind == "audio":
        wav_path = tmp_path / "snd.wav"
        sr = 8000
        t = np.arange(sr, dtype=float) / sr
        tone = np.asarray(10000 * np.sin(2 * math.pi * 440 * t), dtype=np.int16)
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(tone.tobytes())
        dataset_ingest_task.delay("demo", str(wav_path)).get()
        params = json.loads(client.hget("dataset:demo:progress", "ingestion_params"))
        assert params["sample_rate"] == sr
        assert math.isclose(params["duration"], 1.0, rel_tol=1e-3)
        assert params["snr"] > 10.0
    else:
        pdf_path = tmp_path / "doc.pdf"
        text = "".join(chr(97 + i % 26) for i in range(520))
        pdf_path.write_text(text)
        dataset_ingest_task.delay("demo", str(pdf_path)).get()
        params = json.loads(client.hget("dataset:demo:progress", "ingestion_params"))
        assert params["entropy"] > 3.5
