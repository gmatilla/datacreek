import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import networkx as nx
import pytest

pytest.importorskip("datacreek.api", reason="api deps missing")

from datacreek.analysis.mapper import inverse_mapper, mapper_nerve

try:
    from datacreek.core.dataset import DatasetBuilder
    from datacreek.core.knowledge_graph import KnowledgeGraph
    from datacreek.pipelines import DatasetType
except Exception:  # pragma: no cover - optional deps missing
    DatasetBuilder = None  # type: ignore
    KnowledgeGraph = None  # type: ignore


def test_mapper_functions():
    g = nx.path_graph(5)
    nerve, cover = mapper_nerve(g, radius=1)
    assert nerve.number_of_nodes() == len(cover)
    recon = inverse_mapper(nerve, cover)
    for u, v in g.edges():
        assert recon.has_edge(u, v)


def test_dataset_mapper_wrappers():
    if DatasetBuilder is None:
        pytest.skip("deps missing")
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    for i in range(5):
        ds.add_chunk("d", f"c{i}", str(i))
    for i in range(4):
        ds.graph.graph.add_edge(f"c{i}", f"c{i+1}")
    nerve, cover = ds.mapper_nerve(radius=1)
    assert len(cover) == nerve.number_of_nodes()
    g = ds.inverse_mapper(nerve, cover)
    for u, v in ds.graph.graph.edges():
        assert g.has_edge(u, v)
    assert any(e.operation == "mapper_nerve" for e in ds.events)
    assert any(e.operation == "inverse_mapper" for e in ds.events)


def test_mapper_cache(monkeypatch):
    calls: list[int] = []

    def fake_mapper_nerve(g, radius):
        calls.append(radius)
        return nx.Graph(), []

    monkeypatch.setattr("datacreek.analysis.mapper.mapper_nerve", fake_mapper_nerve)

    if DatasetBuilder is None:
        pytest.skip("deps missing")
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    for i in range(3):
        ds.add_chunk("d", f"c{i}", str(i))
    for i in range(2):
        ds.graph.graph.add_edge(f"c{i}", f"c{i+1}")

    ds.mapper_nerve(radius=1)
    # second call should use cache
    ds.mapper_nerve(radius=1)
    assert len(calls) == 1

    ds.clear_mapper_cache()
    ds.mapper_nerve(radius=1)
    assert len(calls) == 2
    assert any(e.operation == "clear_mapper_cache" for e in ds.events)
