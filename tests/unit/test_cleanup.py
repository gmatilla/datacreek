import importlib

router = importlib.import_module("datacreek.core.cleanup")


class DummyKG:
    def __init__(self):
        self.calls = []

    def deduplicate_chunks(self, sim):
        self.calls.append(("dedup", sim))
        return 1

    def clean_chunk_texts(self):
        self.calls.append(("clean",))
        return 2

    def normalize_date_fields(self):
        self.calls.append(("norm",))

    def resolve_entities(self, threshold, aliases=None):
        self.calls.append(("resolve", threshold, aliases))

    def mark_conflicting_facts(self):
        self.calls.append(("mark",))

    def validate_coherence(self):
        self.calls.append(("validate",))


class DummyBuilder:
    def __init__(self):
        self.kwargs = None

    def cleanup_graph(self, **kwargs):
        self.kwargs = kwargs
        return 3, 4


def test_cleanup_no_builder(monkeypatch):
    kg = DummyKG()
    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.verify_thresholds", lambda: None
    )
    stats = router.cleanup_knowledge_graph(kg, resolve_aliases={"a": ["b"]})
    assert stats.removed == 1 and stats.cleaned == 2
    assert ("resolve", 0.8, {"a": ["b"]}) in kg.calls
    assert ("norm",) in kg.calls


def test_cleanup_with_builder(monkeypatch):
    kg = DummyKG()
    builder = DummyBuilder()
    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.verify_thresholds", lambda: None
    )
    stats = router.cleanup_knowledge_graph(
        kg,
        dataset_builder=builder,
        resolve_threshold=0.9,
        dedup_similarity=0.5,
        normalize_dates=False,
        mark_conflicts=True,
        validate=True,
    )
    assert stats.removed == 3 and stats.cleaned == 4
    assert builder.kwargs["resolve_threshold"] == 0.9
    assert builder.kwargs["dedup_similarity"] == 0.5
    assert not any(call[0] == "dedup" for call in kg.calls)
