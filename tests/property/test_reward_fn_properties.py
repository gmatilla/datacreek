"""Property-based tests for the alias-aware reward."""

pytest = __import__("pytest")
pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from training.reward_fn import build_alias_reward_fn

# Strategies generating valid triplet strings compatible with ``extract_triplets``
subjects = st.sampled_from(["alice", "bob", "paris", "london"])
predicates = st.sampled_from(
    ["is", "likes", "has", "knows", "located_in", "capital_of"]
)
objects = st.sampled_from(["apples", "carol", "france", "uk", "usa"])
triplets = st.builds(lambda s, p, o: f"{s} {p} {o}", subjects, predicates, objects)


@given(triplets)
@settings(max_examples=30)
def test_reward_is_one_when_truthful(fact: str) -> None:
    """If the response exactly matches the fact, reward should be 1."""
    s, p, o = fact.split()
    graph = {(s, p): {o}}
    reward_fn = build_alias_reward_fn(graph, None)
    assert reward_fn(fact) == 1.0


@given(triplets)
@settings(max_examples=30)
def test_reward_zero_when_response_empty(fact: str) -> None:
    """Empty responses should yield zero reward regardless of the fact."""
    s, p, o = fact.split()
    graph = {(s, p): {o}}
    reward_fn = build_alias_reward_fn(graph, None)
    assert reward_fn("") == 0.0
