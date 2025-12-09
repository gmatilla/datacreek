import random

from datacreek.analysis.privacy import k_out_randomized_response


def test_k_out_randomized_response_empty():
    assert k_out_randomized_response([], k=2) == []


def test_k_out_randomized_response_replace(monkeypatch):
    items = ["a", "b", "c"]
    seq = iter([0.1, 0.9, 0.2])  # replace second item only
    monkeypatch.setattr(random, "random", lambda: next(seq))
    monkeypatch.setattr(random, "choice", lambda pool: "x")
    result = k_out_randomized_response(items, k=2)
    assert result == ["a", "x", "c"]
