from datacreek.utils.dataset_cleanup import deduplicate_pairs


def test_deduplicate_pairs_exact():
    pairs = [
        {"question": "q", "answer": "a"},
        {"question": "q", "answer": "a"},
        {"question": "q2", "answer": "b"},
    ]
    result = deduplicate_pairs(pairs)
    assert len(result) == 2


def test_deduplicate_pairs_threshold():
    pairs = [
        {"question": "Hello", "answer": "World"},
        {"question": "Hello!", "answer": "World"},
    ]
    result = deduplicate_pairs(pairs, threshold=0.8)
    assert len(result) == 1
