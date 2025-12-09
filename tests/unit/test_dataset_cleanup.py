import pytest

from datacreek.utils.dataset_cleanup import deduplicate_pairs


def test_deduplicate_pairs_removes_near_duplicates():
    pairs = [
        {"question": "How are you?", "answer": "Fine"},
        {"question": "How are you?", "answer": "Fine."},  # near-duplicate
        {"question": "What is your name?", "answer": "Alice"},
        {"question": "How are you today?", "answer": "Fine"},
    ]
    unique = deduplicate_pairs(pairs, threshold=0.85)
    # Should keep first, third, and fourth (second removed)
    assert len(unique) == 3
    assert unique[0]["question"] == "How are you?"
    assert unique[1]["question"] == "What is your name?"
    assert unique[2]["question"] == "How are you today?"


def test_deduplicate_pairs_threshold_strict():
    pairs = [
        {"question": "hello", "answer": "hi"},
        {"question": "hello", "answer": "hi"},
    ]
    unique = deduplicate_pairs(pairs, threshold=1.0)
    # Only identical Q/A removed
    assert len(unique) == 1
    assert unique[0] == {"question": "hello", "answer": "hi"}


def test_deduplicate_pairs_handles_missing_fields():
    pairs = [
        {"question": "hello"},
        {"answer": "hi"},
        {},
    ]
    unique = deduplicate_pairs(pairs)
    # All unique since missing fields cause ratio comparisons with empty string
    assert len(unique) == 3


def test_deduplicate_pairs_question_only_duplicate():
    pairs = [
        {"question": "hello there", "answer": "yes"},
        {"question": "hello there", "answer": "no"},
    ]
    unique = deduplicate_pairs(pairs)
    # answers differ so both remain
    assert len(unique) == 2


def test_deduplicate_pairs_answer_only_duplicate():
    pairs = [
        {"question": "hi", "answer": "greet"},
        {"question": "hello", "answer": "greet"},
    ]
    unique = deduplicate_pairs(pairs)
    # questions differ so both remain
    assert len(unique) == 2
