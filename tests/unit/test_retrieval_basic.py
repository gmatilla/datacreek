import builtins
import importlib
import sys
import types

import pytest

from datacreek.utils import retrieval


@pytest.fixture(autouse=True)
def reload_module(monkeypatch):
    sys.modules["hnswlib"] = types.SimpleNamespace(Index=object)
    importlib.reload(retrieval)
    yield
    importlib.reload(retrieval)
    sys.modules.pop("hnswlib", None)


def test_embedding_index_basic_search():
    idx = retrieval.EmbeddingIndex()
    idx.add("1", "foo bar")
    idx.add("2", "bar baz")
    idx.add("3", "foo baz qux")
    idx.build()
    assert idx._matrix is not None
    if idx._vectorizer is not None:
        res = idx.search("foo", k=2)
        assert len(res) == 2
        texts = [idx.get_text(i) for i in res]
        assert any("foo" in t for t in texts)


def test_remove_and_embed_transform():
    idx = retrieval.EmbeddingIndex()
    idx.add("1", "alpha beta")
    idx.add("2", "beta gamma")
    idx.build()
    if idx._vectorizer is not None:
        emb = idx.embed("alpha")
        assert emb.size
    idx.remove("1")
    idx.build()
    assert idx.get_id(0) == "2"
    mat = idx.transform(["beta gamma"])
    assert mat.shape[0] in (0, 1)


def test_build_fallback_no_sklearn(monkeypatch):
    monkeypatch.setattr(retrieval, "TfidfVectorizer", None)
    monkeypatch.setattr(retrieval, "NearestNeighbors", None)
    monkeypatch.setattr(retrieval, "np", None)

    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    idx = retrieval.EmbeddingIndex()
    idx.add("1", "hello world")
    idx.build()
    assert idx._vectorizer is None
    assert idx._matrix is not None
    assert idx._matrix.shape[0] == 1
