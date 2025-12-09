import json

import pytest

import datacreek.utils.chunking as chunking
from datacreek.utils.chunking import (
    chunk_by_sentences,
    chunk_by_tokens,
    contextual_chunk_split,
    semantic_chunk_split,
    sliding_window_chunks,
    summarized_chunk_split,
)
from datacreek.utils.crypto import (
    decrypt_pii_fields,
    encrypt_pii_fields,
    xor_decrypt,
    xor_encrypt,
)
from datacreek.utils.redis_helpers import decode_hash


def test_sliding_window_chunks():
    text = "abcdefgh"
    assert sliding_window_chunks(text, window_size=3, overlap=1) == [
        "abc",
        "cde",
        "efg",
        "gh",
    ]


def test_sliding_window_chunks_errors():
    with pytest.raises(ValueError):
        sliding_window_chunks("abc", window_size=0, overlap=0)
    with pytest.raises(ValueError):
        sliding_window_chunks("abc", window_size=3, overlap=3)


def test_chunk_by_tokens():
    text = "one two three four five six"
    assert chunk_by_tokens(text, max_tokens=2, overlap=1) == [
        "one two",
        "two three",
        "three four",
        "four five",
        "five six",
    ]


def test_chunk_by_tokens_errors():
    with pytest.raises(ValueError):
        chunk_by_tokens("a b", max_tokens=0)
    with pytest.raises(ValueError):
        chunk_by_tokens("a b", max_tokens=2, overlap=2)


def test_chunk_by_sentences():
    text = "A. B. C. D."
    assert chunk_by_sentences(text, max_sentences=2) == ["A. B", "C. D"]


def test_chunk_by_sentences_error():
    with pytest.raises(ValueError):
        chunk_by_sentences("A", max_sentences=0)


def test_semantic_chunk_split_requires_deps():
    with pytest.raises(ImportError):
        semantic_chunk_split("a. b.", max_tokens=10)


def test_semantic_chunk_split(monkeypatch):
    """Cover semantic_chunk_split with stubbed dependencies."""

    class FakeVectorizer:
        def fit(self, sentences):
            return self

        def transform(self, sentences):
            class Arr:
                def toarray(self) -> list[list[float]]:
                    return [[1.0] for _ in sentences]

            return Arr()

    monkeypatch.setattr(chunking, "TfidfVectorizer", FakeVectorizer)

    import types

    monkeypatch.setattr(chunking, "np", types.SimpleNamespace(dot=lambda a, b: 1.0))

    result = semantic_chunk_split("A. B. C.", max_tokens=50)
    assert result == ["A. B. C"]


def test_contextual_chunk_split():
    text = "one two three four five six seven eight"
    chunks = contextual_chunk_split(text, max_tokens=3, context_size=2)
    assert chunks[0] == "one two three"
    assert chunks[1].startswith("two three four")


def test_summarized_chunk_split(monkeypatch):
    called = {}

    def fake_semantic(text: str, max_tokens: int) -> list[str]:
        called["args"] = (text, max_tokens)
        return ["aaa bbb", "ccc ddd", "eee fff"]

    monkeypatch.setattr("datacreek.utils.chunking.semantic_chunk_split", fake_semantic)
    chunks = summarized_chunk_split("irrelevant", max_tokens=5, summary_len=1)
    assert chunks[0] == "aaa bbb"
    assert chunks[1] == "bbb ccc ddd"
    assert called["args"] == ("irrelevant", 5)


def test_crypto_roundtrip():
    key = "secret"
    token = xor_encrypt("hello", key)
    assert xor_decrypt(token, key) == "hello"


def test_encrypt_decrypt_fields():
    record = {"name": "Bob", "ssn": "123"}
    encrypted = encrypt_pii_fields(record.copy(), key="k", fields=["ssn"])
    assert encrypted["ssn"] != "123"
    decrypted = decrypt_pii_fields(encrypted, key="k", fields=["ssn"])
    assert decrypted["ssn"] == "123"


def test_decode_hash():
    data = {"a": b"1", b"b": b'{"x": 1}'}
    assert decode_hash(data) == {"a": 1, "b": {"x": 1}}
