import asyncio
import types
from types import SimpleNamespace

import pytest

import datacreek.core.create as create
from datacreek.models.content_type import ContentType


class DummyLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.provider = kwargs.get("provider")
        self.config = {}


def test_load_document_text_and_base_name(tmp_path, monkeypatch):
    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello")
    assert create.load_document_text(str(file_path)) == "hello"
    assert create._base_name(str(file_path)) == "doc"
    assert create._base_name(None) == "input"

    monkeypatch.setattr(create, "LLMClient", DummyLLM)
    client = create.init_llm_client(file_path, provider="test")
    assert isinstance(client, DummyLLM)
    assert client.kwargs["provider"] == "test"


def test_process_file_unknown_content_type(monkeypatch):
    monkeypatch.setattr(create, "LLMClient", DummyLLM)
    kg = SimpleNamespace(to_text=lambda: "data")
    with pytest.raises(ValueError):
        create.process_file(None, content_type="invalid", kg=kg)


def test_process_file_qa(monkeypatch):
    monkeypatch.setattr(create, "LLMClient", DummyLLM)

    class DummyGen:
        def __init__(self, *a, **k):
            pass

        def process_document(self, text, num_pairs, verbose):
            assert text == "txt"
            return SimpleNamespace(to_dict=lambda: {"pairs": num_pairs})

    monkeypatch.setattr("datacreek.generators.qa_generator.QAGenerator", DummyGen)
    kg = SimpleNamespace(to_text=lambda: "txt")
    result = create.process_file(None, kg=kg, num_pairs=2, content_type=ContentType.QA)
    assert result == {"pairs": 2}


def test_process_file_async(monkeypatch):
    monkeypatch.setattr(create, "LLMClient", DummyLLM)
    monkeypatch.setattr(create, "process_file", lambda *a, **k: "sync")
    kg = SimpleNamespace(to_text=lambda: "t")
    res = asyncio.run(
        create.process_file_async("file", kg=kg, content_type=ContentType.SUMMARY)
    )
    assert res == "sync"

    class DummyGen:
        def __init__(self, *a, **k):
            pass

        async def process_document_async(self, text, num_pairs, verbose):
            assert text == "txt"
            return SimpleNamespace(to_dict=lambda: {"async": num_pairs})

    monkeypatch.setattr("datacreek.generators.qa_generator.QAGenerator", DummyGen)
    res2 = asyncio.run(
        create.process_file_async(
            None, kg=SimpleNamespace(to_text=lambda: "txt"), num_pairs=3
        )
    )
    assert res2 == {"async": 3}
