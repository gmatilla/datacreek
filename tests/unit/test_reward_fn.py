"""Unit tests for alias-aware reward function."""

from training.reward_fn import build_alias_reward_fn


class DummySession:
    """Minimal in-memory stand-in for a Neo4j session."""

    def __init__(self, mapping):
        self.mapping = mapping

    def run(self, query, n):  # pragma: no cover - trivial passthrough
        canonical = self.mapping.get(n)
        return [{"canonical": canonical}] if canonical else []

    def __enter__(self):  # pragma: no cover - context manager protocol
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover
        return False


class DummyDriver:
    """Neo4j driver stub returning ``DummySession`` objects."""

    def __init__(self, mapping):
        self.mapping = mapping

    def session(self):  # pragma: no cover
        return DummySession(self.mapping)


def test_alias_resolution_counts_as_valid_fact():
    """Facts matching only after alias canonicalisation should be rewarded."""
    graph = {("new_york_city", "located_in"): {"usa"}}
    driver = DummyDriver({"nyc": "new_york_city"})
    reward_fn = build_alias_reward_fn(graph, driver)
    assert reward_fn("NYC located_in USA") == 1.0


def test_ratio_accounts_for_invalid_facts():
    """Unverified facts reduce the overall reward proportionally."""
    graph = {
        ("new_york_city", "located_in"): {"usa"},
        ("paris", "located_in"): {"france"},
    }
    driver = DummyDriver({"nyc": "new_york_city"})
    text = "NYC located_in USA. Paris located_in France. Berlin located_in Mars."
    reward_fn = build_alias_reward_fn(graph, driver)
    assert reward_fn(text) == 2 / 3
