import types

import pytest

from datacreek.utils import chunking


class FakeVectorizer:
    def fit(self, sentences):
        self.sentences = list(sentences)
        return self

    def transform(self, sentences):
        class Arr:
            def __init__(self, sents):
                self._data = [[len(s)] for s in sents]

            def toarray(self):
                return self._data

        return Arr(sentences)


def fake_np_dot(a, b):
    return a[0] * b[0]


def test_sliding_window_chunks_basic():
    text = "abcdefg"
    res = chunking.sliding_window_chunks(text, window_size=3, overlap=1)
    # The final chunk stops at the end of the string
    assert res == ["abc", "cde", "efg"]


@pytest.mark.parametrize("w,o", [(0, 0), (2, 2)])
def test_sliding_window_invalid(w, o):
    with pytest.raises(ValueError):
        chunking.sliding_window_chunks("text", w, o)


def test_chunk_by_tokens_and_sentences():
    text = "one two three four five six seven"
    parts = chunking.chunk_by_tokens(text, max_tokens=3, overlap=1)
    assert parts == ["one two three", "three four five", "five six seven"]
    with pytest.raises(ValueError):
        chunking.chunk_by_tokens(text, 0)
    with pytest.raises(ValueError):
        chunking.chunk_by_tokens(text, 2, overlap=2)

    sent_text = "A. B. C. D."
    assert chunking.chunk_by_sentences(sent_text, 2) == ["A. B", "C. D"]
    with pytest.raises(ValueError):
        chunking.chunk_by_sentences(sent_text, 0)


def test_contextual_chunk_split():
    text = "one two three four five six"
    out = chunking.contextual_chunk_split(text, max_tokens=2, context_size=1)
    assert out == ["one two", "two three four", "four five six"]


def test_semantic_chunk_split(monkeypatch):
    monkeypatch.setattr(chunking, "TfidfVectorizer", FakeVectorizer)
    monkeypatch.setattr(chunking, "np", types.SimpleNamespace(dot=fake_np_dot))
    text = "one two. three four. five"
    res = chunking.semantic_chunk_split(text, max_tokens=20, similarity_drop=0.5)
    assert res == ["one two. three four", "five"]

    monkeypatch.setattr(chunking, "TfidfVectorizer", None)
    with pytest.raises(ImportError):
        chunking.semantic_chunk_split("x", 5)
    monkeypatch.setattr(chunking, "TfidfVectorizer", FakeVectorizer)
    monkeypatch.setattr(chunking, "np", None)
    with pytest.raises(ImportError):
        chunking.semantic_chunk_split("x", 5)


def test_summarized_chunk_split(monkeypatch):
    monkeypatch.setattr(
        chunking, "semantic_chunk_split", lambda t, max_tokens: ["c1", "c2", "c3"]
    )
    res = chunking.summarized_chunk_split("irrelevant", max_tokens=5, summary_len=1)
    assert res == ["c1", "c1 c2", "c2 c3"]
