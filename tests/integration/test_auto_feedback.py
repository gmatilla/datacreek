"""Integration tests for graph-based RLHF rewards."""

from training.auto_feedback import build_reward_fn, extract_triplets


def test_extract_triplets_basic():
    """The extractor should return lowercase triplets."""
    text = "Paris located_in France. Alice eats apples."
    assert extract_triplets(text) == [
        ("paris", "located_in", "france"),
        ("alice", "eats", "apples"),
    ]


def test_reward_fn_verification_ratio():
    """Reward function should reflect fraction of verified triplets."""
    graph = {
        ("paris", "located_in"): {"france"},
        ("alice", "eats"): {"apples"},
        ("earth", "is"): {"round"},
    }
    text = (
        "Paris located_in France. Alice eats apples. "
        "Earth is round. Bob eats stones."
    )
    reward_fn = build_reward_fn(graph)
    reward = reward_fn(text)
    assert reward == 0.75
    assert reward > 0.7
