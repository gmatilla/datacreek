import numpy as np

from datacreek.analysis.fractal import fractal_net_prune, fractalnet_compress

try:
    from datacreek.core.dataset import DatasetBuilder
    from datacreek.core.knowledge_graph import KnowledgeGraph
    from datacreek.pipelines import DatasetType
except Exception:  # pragma: no cover - optional deps missing
    DatasetBuilder = None  # type: ignore
    KnowledgeGraph = None  # type: ignore

import pytest


def test_prune_sources_kg():
    if KnowledgeGraph is None:
        pytest.skip("kg deps missing")
    kg = KnowledgeGraph()
    kg.add_document("doc1", source="bad")
    kg.add_chunk("doc1", "c1", "text", source="bad")
    kg.add_entity("e1", "ent", source="good")
    kg.link_entity("c1", "e1", provenance="bad")

    removed = kg.prune_sources(["bad"])
    assert removed == 2
    assert "doc1" not in kg.graph.nodes
    assert "c1" not in kg.graph.nodes
    # entity should remain
    assert "e1" in kg.graph.nodes


def test_prune_sources_wrapper():
    if DatasetBuilder is None:
        pytest.skip("deps missing")
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("doc1", source="bad")
    ds.add_chunk("doc1", "c1", "text", source="bad")
    ds.add_document("doc2", source="good")

    removed = ds.prune_sources(["bad"])
    assert removed == 2
    assert ds.search_documents("doc1") == []
    assert ds.search_documents("doc2") == ["doc2"]


def test_fractal_net_prune():
    emb = {"a": [0.0, 0.0], "b": [0.02, 0.0], "c": [1.0, 0.0]}
    pruned, mapping = fractal_net_prune(emb, tol=0.05)
    assert len(pruned) == 2
    assert mapping["a"] == mapping["b"]


def test_fractalnet_compress():
    emb = {"a": [0.0, 0.0], "b": [0.2, 0.0], "c": [1.0, 0.0]}
    levels = {"a": 0, "b": 0, "c": 1}
    comps = fractalnet_compress(emb, levels)
    assert len(comps) == 2
    assert np.allclose(comps[0], [0.1, 0.0])


def test_prune_embeddings_wrapper():
    if DatasetBuilder is None:
        pytest.skip("deps missing")
    ds = DatasetBuilder(DatasetType.TEXT)
    for node, vec in {"a": [0.0, 0.0], "b": [0.02, 0.0], "c": [1.0, 0.0]}.items():
        ds.graph.graph.add_node(node, embedding=vec)
    mapping = ds.prune_embeddings(tol=0.05)
    assert len(set(mapping.values())) == 2
    assert any(e.operation == "prune_embeddings" for e in ds.events)


def test_fractalnet_compress_wrapper():
    if DatasetBuilder is None:
        pytest.skip("deps missing")
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.graph.graph.add_node("a", embedding=[0.0, 0.0], fractal_level=0)
    ds.graph.graph.add_node("b", embedding=[0.2, 0.0], fractal_level=0)
    ds.graph.graph.add_node("c", embedding=[1.0, 0.0], fractal_level=1)
    comp = ds.fractalnet_compress()
    assert len(comp) == 2
    assert any(e.operation == "fractalnet_compress" for e in ds.events)
