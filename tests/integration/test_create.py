import json
import os
import types
from pathlib import Path

import fakeredis

from datacreek.core.create import _base_name, load_document_text, process_file
from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.storage import RedisStorage
from datacreek.utils import load_config


def test_base_name_helper():
    assert _base_name("/tmp/foo.txt") == "foo"
    assert _base_name(None) == "input"


def test_load_document_text(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("hello")
    assert load_document_text(str(p)) == "hello"


def test_process_file_no_output(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"x")

    called = {"resolved": False}

    def fake_resolve(path):
        called["resolved"] = True
        return str(tmp_path / "out")

    class DummyParser:
        def parse(self, file_path, **kwargs):
            return "ok"

    class DummyGenerator:
        def __init__(self, *a, **k):
            pass

        def process_document(self, *a, **k):
            return types.SimpleNamespace(to_dict=lambda: {"qa_pairs": []})

    monkeypatch.setattr(
        "datacreek.core.ingest.determine_parser", lambda f, c: DummyParser()
    )
    monkeypatch.setattr("datacreek.generators.qa_generator.QAGenerator", DummyGenerator)
    monkeypatch.setattr(
        "datacreek.core.create.init_llm_client",
        lambda *a, **k: types.SimpleNamespace(provider="test", config={}),
    )

    text = process_file(
        str(pdf),
        kg=KnowledgeGraph(),
    )

    assert text == {"qa_pairs": []}
    assert called["resolved"] is False


def test_process_file_redis_output(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"x")

    class DummyParser:
        def parse(self, file_path, **kwargs):
            return "ok"

    class DummyGenerator:
        def __init__(self, *a, **k):
            pass

        def process_document(self, *a, **k):
            return types.SimpleNamespace(to_dict=lambda: {"qa_pairs": []})

    monkeypatch.setattr(
        "datacreek.core.ingest.determine_parser", lambda f, c: DummyParser()
    )
    monkeypatch.setattr("datacreek.generators.qa_generator.QAGenerator", DummyGenerator)
    monkeypatch.setattr(
        "datacreek.core.create.init_llm_client",
        lambda *a, **k: types.SimpleNamespace(provider="test", config={}),
    )

    client = fakeredis.FakeStrictRedis()

    key = "out:data"
    result = process_file(
        str(pdf),
        kg=KnowledgeGraph(),
        redis_client=client,
        redis_key=key,
    )

    assert result == key
    assert json.loads(client.get(key)) == {"qa_pairs": []}


class DummyBackend(RedisStorage):
    def __init__(self, client):
        super().__init__(client)


def test_process_file_backend(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"x")

    class DummyParser:
        def parse(self, file_path, **kwargs):
            return "ok"

    class DummyGenerator:
        def __init__(self, *a, **k):
            pass

        def process_document(self, *a, **k):
            return types.SimpleNamespace(to_dict=lambda: {"qa_pairs": []})

    monkeypatch.setattr(
        "datacreek.core.ingest.determine_parser", lambda f, c: DummyParser()
    )
    monkeypatch.setattr("datacreek.generators.qa_generator.QAGenerator", DummyGenerator)
    monkeypatch.setattr(
        "datacreek.core.create.init_llm_client",
        lambda *a, **k: types.SimpleNamespace(provider="test", config={}),
    )

    client = fakeredis.FakeStrictRedis()
    backend = DummyBackend(client)
    key = "out:data"
    result = process_file(
        str(pdf),
        kg=KnowledgeGraph(),
        backend=backend,
        redis_key=key,
    )

    assert result == key
    assert json.loads(client.get(key)) == {"qa_pairs": []}
