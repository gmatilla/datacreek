import json
import types

import pytest

from datacreek.core.save_as import _format_pairs, convert_format
from datacreek.storage import StorageBackend


def test_format_pairs_basic():
    pairs = [{"question": "q", "answer": "a"}]
    assert _format_pairs(pairs, "jsonl") == json.dumps(pairs[0])
    alpaca = json.loads(_format_pairs(pairs, "alpaca"))
    assert alpaca[0]["instruction"] == "q"
    messages = json.loads(_format_pairs(pairs, "ft"))
    assert messages[0]["messages"][1]["content"] == "q"
    with pytest.raises(ValueError):
        _format_pairs(pairs, "bad")


class DummyBackend(StorageBackend):
    def __init__(self):
        self.saved = []

    def save(self, key: str, data: str) -> str:
        self.saved.append((key, data))
        return f"ok:{key}"


class DummyRedis:
    def __init__(self):
        self.data = {}

    def set(self, key, val):
        self.data[key] = val


def test_convert_format_variants(tmp_path, monkeypatch):
    pairs = [{"question": "q1", "answer": "a1"}]
    file = tmp_path / "in.json"
    file.write_text(json.dumps({"qa_pairs": pairs}))
    assert "q1" in convert_format(str(file), "jsonl")

    json_str = json.dumps({"qa_pairs": pairs})
    assert "q1" in convert_format(json_str, "alpaca")

    convs = {
        "conversations": [
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ]
        ]
    }
    formatted = convert_format(convs, "ft")
    assert "q2" in formatted

    assert "q1" in convert_format(pairs, "jsonl")


def test_convert_format_hf_and_storage(monkeypatch):
    pairs = [{"question": "q", "answer": "a"}]
    backend = DummyBackend()
    monkeypatch.setattr(
        "datacreek.core.save_as.to_hf_dataset", lambda data, key: f"HF:{key}"
    )
    result = convert_format(
        {"qa_pairs": pairs},
        "jsonl",
        storage_format="hf",
        backend=backend,
        redis_key="k",
    )
    assert result == "HF:k"


def test_convert_format_to_backend_and_redis(monkeypatch):
    pairs = [{"question": "q", "answer": "a"}]
    backend = DummyBackend()
    redis_client = DummyRedis()
    monkeypatch.setattr(
        "datacreek.core.save_as.logger",
        types.SimpleNamespace(exception=lambda *a, **k: None),
    )
    key = convert_format({"qa_pairs": pairs}, "jsonl", backend=backend, redis_key="k")
    assert backend.saved == [("k", _format_pairs(pairs, "jsonl"))]
    assert key == "ok:k"

    key = convert_format(
        {"qa_pairs": pairs}, "jsonl", redis_client=redis_client, redis_key="r"
    )
    assert redis_client.data["r"]
    assert key == "r"
