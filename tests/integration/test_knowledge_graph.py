import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest

pytest.importorskip("datacreek.api", reason="api deps missing")

import numpy as np
import pytest

try:
    import networkx as nx
    import requests
except Exception:  # pragma: no cover - deps missing
    nx = None  # type: ignore
    requests = None  # type: ignore

from datacreek.analysis import bottleneck_distance

try:
    from datacreek import AutoTuneState
    from datacreek.core.knowledge_graph import KnowledgeGraph
except Exception:  # pragma: no cover - deps missing
    AutoTuneState = None  # type: ignore
    KnowledgeGraph = None  # type: ignore


def test_add_document_and_chunk():
    kg = KnowledgeGraph()
    kg.add_document("doc1", source="paper.pdf")
    kg.add_chunk("doc1", "chunk1", "hello")

    assert kg.graph.nodes["doc1"]["source"] == "paper.pdf"
    assert kg.graph.nodes["chunk1"]["source"] == "paper.pdf"
    assert ("doc1", "chunk1") in kg.graph.edges


def test_search_chunks():
    kg = KnowledgeGraph()
    kg.add_document("doc1", source="paper.pdf")
    kg.add_chunk("doc1", "c1", "hello world")
    kg.add_chunk("doc1", "c2", "another line")

    matches = kg.search_chunks("world")
    assert matches == ["c1"]


def test_generic_search():
    kg = KnowledgeGraph()
    kg.add_document("doc1", source="paper.pdf")
    kg.add_chunk("doc1", "c1", "hello world")
    assert kg.search("hello") == ["c1"]


def test_document_helpers():
    kg = KnowledgeGraph()
    kg.add_document("doc1", source="paper.pdf")
    kg.add_document("guide", source="guide.txt")
    kg.add_chunk("doc1", "c1", "text1")
    kg.add_chunk("doc1", "c2", "text2")

    assert set(kg.search_documents("doc")) == {"doc1"}
    assert kg.get_chunks_for_document("doc1") == ["c1", "c2"]


def test_embedding_search():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello world")
    kg.add_chunk("d", "c2", "another world")
    kg.index.build()
    results = kg.search_embeddings("hello", k=1)
    assert results[0] == "c1"


def test_duplicate_checks():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    with pytest.raises(ValueError):
        kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "text")
    with pytest.raises(ValueError):
        kg.add_chunk("d", "c1", "text")


def test_hybrid_search():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "bonjour le monde")
    kg.add_chunk("doc", "c3", "greetings planet")
    kg.index.build()

    results = kg.search_hybrid("hello", k=2)
    assert results[0] == "c1"
    assert len(results) == 2


def test_chunk_order_and_next_relations():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "t1")
    kg.add_chunk("doc", "c2", "t2")
    kg.add_chunk("doc", "c3", "t3")

    # edges should have sequence numbers
    assert kg.graph.edges["doc", "c1"]["sequence"] == 0
    assert kg.graph.edges["doc", "c2"]["sequence"] == 1
    assert kg.graph.edges["doc", "c3"]["sequence"] == 2

    # next_chunk relations preserve order
    assert ("c1", "c2") in kg.graph.edges
    assert kg.graph.edges["c1", "c2"]["relation"] == "next_chunk"
    assert kg.get_chunks_for_document("doc") == ["c1", "c2", "c3"]
    assert kg.get_next_chunk("c1") == "c2"
    assert kg.get_previous_chunk("c3") == "c2"


def test_serialization_preserves_order():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "first")
    kg.add_chunk("d", "c2", "second")

    data = kg.to_dict()
    loaded = KnowledgeGraph.from_dict(data)
    assert loaded.get_chunks_for_document("d") == ["c1", "c2"]


def test_section_helpers():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "s1", title="Intro")
    kg.add_section("doc", "s2", title="Body")
    kg.add_chunk("doc", "c1", "t1", section_id="s1")
    kg.add_chunk("doc", "c2", "t2", section_id="s2")

    assert kg.get_sections_for_document("doc") == ["s1", "s2"]
    assert kg.get_chunks_for_section("s1") == ["c1"]
    assert kg.get_section_for_chunk("c2") == "s2"
    assert kg.get_next_section("s1") == "s2"
    assert kg.get_previous_section("s2") == "s1"


def test_link_similar_chunks():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "hello planet")
    kg.add_chunk("doc", "c3", "unrelated text")

    kg.index.build()
    kg.link_similar_chunks(k=1)

    assert ("c1", "c2") in kg.graph.edges
    edge = kg.graph.edges["c1", "c2"]
    assert edge["relation"] == "similar_to"
    assert 0 <= edge["similarity"] <= 1


def test_link_similar_sections():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "s1", title="Intro")
    kg.add_section("doc", "s2", title="Introduction")
    kg.add_section("doc", "s3", title="Other")

    kg.index.build()
    kg.link_similar_sections(k=1)

    assert ("s1", "s2") in kg.graph.edges
    edge = kg.graph.edges["s1", "s2"]
    assert edge["relation"] == "similar_to"
    assert 0 <= edge["similarity"] <= 1


def test_embeddings_filter_by_type():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_entity("e1", "hello world")
    kg.index.build()

    # Should return only chunk IDs when searching embeddings
    results = kg.search_embeddings("hello", k=1, node_type="chunk")
    assert results == ["c1"]


def test_hybrid_filter_by_type():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_entity("e1", "hello world")
    kg.index.build()

    # Should not return entity nodes when filtering for chunks
    results = kg.search_hybrid("hello", k=2, node_type="chunk")
    assert results == ["c1"]

    # Should return the entity when requested
    results = kg.search_hybrid("hello", k=1, node_type="entity")
    assert results == ["e1"]


def test_link_similar_chunks_ignores_entities():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "hello planet")
    kg.add_entity("e1", "hello universe")
    kg.index.build()
    kg.link_similar_chunks(k=1)

    # no similar_to edge should involve entity nodes
    for u, v, d in kg.graph.edges(data=True):
        if d.get("relation") == "similar_to":
            assert kg.graph.nodes[u]["type"] == "chunk"
            assert kg.graph.nodes[v]["type"] == "chunk"


def test_get_similar_chunks():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "hello planet")
    kg.add_chunk("doc", "c3", "different text")

    kg.index.build()
    sims = kg.get_similar_chunks("c1", k=2)
    assert "c1" not in sims
    assert "c2" in sims


def test_get_similar_chunks_unknown():
    kg = KnowledgeGraph()
    assert kg.get_similar_chunks("missing") == []


def test_get_similar_chunks_data():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "hello planet")
    kg.add_chunk("doc", "c3", "other")
    kg.index.build()

    data = kg.get_similar_chunks_data("c1", k=2)
    ids = [d["id"] for d in data]
    assert "c1" not in ids
    assert "c2" in ids


def test_get_chunk_neighbors_data():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "hello planet")
    kg.add_chunk("doc", "c3", "different words")
    kg.index.build()

    data = kg.get_chunk_neighbors_data(k=1)
    assert set(data.keys()) == {"c1", "c2", "c3"}
    assert data["c1"][0]["id"] in {"c2", "c3"}


def test_get_chunk_context():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "t1")
    kg.add_chunk("doc", "c2", "t2")
    kg.add_chunk("doc", "c3", "t3")
    ctx = kg.get_chunk_context("c2", before=1, after=1)
    assert ctx == ["c1", "c2", "c3"]


def test_get_similar_sections():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "s1", title="Introduction")
    kg.add_section("doc", "s2", title="Introductory remarks")
    kg.add_section("doc", "s3", title="Other section")

    kg.index.build()
    sims = kg.get_similar_sections("s1", k=2)
    assert "s1" not in sims
    assert "s2" in sims


def test_get_similar_sections_unknown():
    kg = KnowledgeGraph()
    assert kg.get_similar_sections("missing") == []


def test_link_similar_documents():
    kg = KnowledgeGraph()
    kg.add_document("d1", source="s", text="hello world")
    kg.add_document("d2", source="s", text="hello planet")
    kg.add_document("d3", source="s", text="other text")

    kg.index.build()
    kg.link_similar_documents(k=1)

    assert ("d1", "d2") in kg.graph.edges
    edge = kg.graph.edges["d1", "d2"]
    assert edge["relation"] == "similar_to"
    assert 0 <= edge["similarity"] <= 1


def test_get_similar_documents():
    kg = KnowledgeGraph()
    kg.add_document("d1", source="s", text="hello world")
    kg.add_document("d2", source="s", text="hello planet")
    kg.add_document("d3", source="s", text="unrelated")

    kg.index.build()
    sims = kg.get_similar_documents("d1", k=2)
    assert "d1" not in sims
    assert "d2" in sims


def test_get_similar_documents_unknown():
    kg = KnowledgeGraph()
    assert kg.get_similar_documents("missing") == []


def test_get_chunk_context_unknown():
    kg = KnowledgeGraph()
    assert kg.get_chunk_context("missing") == []


def test_page_for_chunk():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "text", page=5)

    assert kg.get_page_for_chunk("c1") == 5


def test_page_for_section():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "s1", page=3)
    kg.add_chunk("doc", "c1", "t", section_id="s1", page=3)

    assert kg.get_page_for_section("s1") == 3


def test_next_section_fallback():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "s1")
    kg.add_section("doc", "s2")
    kg.add_section("doc", "s3")

    # remove explicit next_section edges
    for u, v, d in list(kg.graph.edges(data=True)):
        if d.get("relation") == "next_section":
            kg.graph.remove_edge(u, v)

    assert kg.get_next_section("s1") == "s2"
    assert kg.get_previous_section("s3") == "s2"


def test_document_lookup_helpers():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "sec1")
    kg.add_chunk("doc", "c1", "text", section_id="sec1")

    assert kg.get_document_for_section("sec1") == "doc"
    assert kg.get_document_for_chunk("c1") == "doc"

    kg.graph.remove_edge("doc", "c1")
    assert kg.get_document_for_chunk("c1") == "doc"


def test_fact_search():
    kg = KnowledgeGraph()
    fid = kg.add_fact("Paris", "capital_of", "France")
    kg.index.build()

    assert kg.search("capital_of", node_type="fact") == [fid]
    assert kg.search_embeddings("capital_of", k=1, node_type="fact") == [fid]
    assert kg.search_hybrid("capital_of", k=1, node_type="fact") == [fid]


def test_search_with_links():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "hello planet")
    kg.add_chunk("doc", "c3", "different text")

    kg.index.build()
    kg.link_similar_chunks(k=1)

    # search should return c1 and also c2 via the similarity edge
    results = kg.search_with_links("hello", k=1, hops=1)
    assert "c1" in results
    assert "c2" in results


def test_search_with_links_data():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "hello world")
    kg.add_chunk("doc", "c2", "hello planet")
    kg.add_chunk("doc", "c3", "different text")

    kg.index.build()
    kg.link_similar_chunks(k=1)

    results = kg.search_with_links_data("hello", k=1, hops=1)
    ids = [r["id"] for r in results]
    assert "c1" in ids
    assert "c2" in ids
    first = next(r for r in results if r["id"] == "c1")
    assert first["text"] == "hello world"
    assert first["document"] == "doc"
    assert first["depth"] == 0
    assert first["path"] == ["c1"]
    second = next(r for r in results if r["id"] == "c2")
    assert second["depth"] == 1
    assert second["path"] == ["c1", "c2"]


def test_community_and_trust():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    for i in range(3):
        kg.add_chunk("doc", f"c{i}", f"text {i}")
    kg.index.build()
    kg.cluster_chunks(n_clusters=1)
    kg.summarize_communities()
    kg.score_trust()

    comms = [n for n, d in kg.graph.nodes(data=True) if d.get("type") == "community"]
    assert len(comms) == 1
    cid = comms[0]
    assert "summary" in kg.graph.nodes[cid]
    for n, d in kg.graph.nodes(data=True):
        if d.get("type") == "chunk":
            assert "trust" in d


def test_entity_groups():
    kg = KnowledgeGraph()
    kg.add_entity("e1", "first entity")
    kg.add_entity("e2", "second entity")
    kg.add_entity("e3", "other text")
    kg.index.build()
    kg.cluster_entities(n_clusters=1)
    kg.summarize_entity_groups()

    groups = [
        n for n, d in kg.graph.nodes(data=True) if d.get("type") == "entity_group"
    ]
    assert len(groups) == 1
    gid = groups[0]
    assert "summary" in kg.graph.nodes[gid]
    members = [u for u, _ in kg.graph.in_edges(gid)]
    assert set(members) == {"e1", "e2", "e3"}


def test_edge_provenance_and_trust():
    kg = KnowledgeGraph()
    kg.add_document("d", source="src")
    kg.add_chunk("d", "c1", "text", source="src")
    kg.add_entity("e", "ent", source="src")
    kg.link_entity("c1", "e", provenance="src")
    kg.index.build()
    kg.score_trust()

    assert kg.graph.edges["d", "c1"]["provenance"] == "src"
    assert "trust" in kg.graph.edges["d", "c1"]
    assert kg.graph.edges["c1", "e"]["provenance"] == "src"
    assert "trust" in kg.graph.edges["c1", "e"]


def test_update_embeddings():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello world")
    kg.index.build()
    kg.update_embeddings()
    emb = kg.graph.nodes["c1"].get("embedding")
    assert isinstance(emb, list)
    assert len(emb) > 0


def test_deduplicate_and_resolve_entities(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "hello")
    kg.add_entity("e1", "Beethoven")
    kg.add_entity("e2", "Ludwig van Beethoven")
    removed = kg.deduplicate_chunks()
    assert removed == 1
    merged = kg.resolve_entities(threshold=0.5)
    assert merged >= 1
    assert "e2" not in kg.graph.nodes


def test_resolve_entities_with_aliases():
    kg = KnowledgeGraph()
    kg.add_entity("e1", "IBM")
    kg.add_entity("e2", "International Business Machines")

    merged = kg.resolve_entities(
        threshold=1.0, aliases={"IBM": ["international business machines"]}
    )

    assert merged == 1
    assert "e2" not in kg.graph.nodes


def test_enrich_entity_wikidata(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Beethoven")

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

    def fake_get(url, params=None, timeout=10):
        return FakeResponse({"search": [{"id": "Q1", "description": "composer"}]})

    monkeypatch.setattr(requests, "get", fake_get)
    kg.enrich_entity_wikidata("e1")
    node = kg.graph.nodes["e1"]
    assert node.get("wikidata_id") == "Q1"
    assert node.get("description") == "composer"


def test_enrich_entity_dbpedia(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Beethoven")

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

    def fake_get(url, params=None, headers=None, timeout=10):
        return FakeResponse(
            {
                "results": [
                    {
                        "id": "http://dbpedia.org/resource/Beethoven",
                        "description": "desc",
                    }
                ]
            }
        )

    monkeypatch.setattr(requests, "get", fake_get)
    kg.enrich_entity_dbpedia("e1")
    node = kg.graph.nodes["e1"]
    assert node.get("dbpedia_uri") == "http://dbpedia.org/resource/Beethoven"
    assert node.get("description_dbpedia") == "desc"


def test_compute_centrality():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    kg.add_entity("e1", "A")
    kg.add_entity("e2", "B")
    kg.link_entity("c1", "e1")
    kg.link_entity("c2", "e2")
    kg.compute_centrality()
    assert "centrality" in kg.graph.nodes["e1"]


def test_compute_node2vec_embeddings():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    kg.add_entity("e1", "A")
    kg.add_entity("e2", "B")
    kg.link_entity("c1", "e1")
    kg.link_entity("c2", "e2")
    kg.compute_node2vec_embeddings(dimensions=8, walk_length=4, num_walks=5, seed=42)
    assert isinstance(kg.graph.nodes["e1"].get("embedding"), list)
    assert len(kg.graph.nodes["e1"]["embedding"]) == 8


def test_compute_graphwave_embeddings():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    kg.add_entity("e1", "A")
    kg.add_entity("e2", "B")
    kg.link_entity("c1", "e1")
    kg.link_entity("c2", "e2")
    kg.compute_graphwave_embeddings(scales=[0.5], num_points=4)
    vec = kg.graph.nodes["e1"].get("graphwave_embedding")
    assert isinstance(vec, list)
    assert len(vec) == 8


def test_graphwave_entropy_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    for i in range(3):
        kg.add_chunk("d", f"c{i}", "txt")
    for i in range(2):
        kg.add_entity(f"e{i}", "E")
    kg.link_entity("c0", "e0")
    kg.link_entity("c1", "e1")
    kg.compute_graphwave_embeddings(scales=[0.5], num_points=4)
    h = kg.graphwave_entropy()
    assert isinstance(h, float)


def test_ensure_graphwave_entropy_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "a")
    kg.add_chunk("d", "c2", "b")
    kg.add_entity("e1", "A")
    kg.link_entity("c1", "e1")
    kg.compute_graphwave_embeddings(scales=[0.5], num_points=4)
    val = kg.ensure_graphwave_entropy(0.1, scales=[0.5], num_points=4)
    assert isinstance(val, float)


def test_embedding_entropy_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    for i in range(3):
        kg.add_chunk("d", f"c{i}", "txt")
    kg.compute_node2vec_embeddings(dimensions=2, walk_length=2, num_walks=5, seed=0)
    ent = kg.embedding_entropy()
    assert isinstance(ent, float)


def test_embedding_box_counting_dimension_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    for i in range(4):
        kg.add_chunk("d", f"c{i}", "txt")
    kg.compute_node2vec_embeddings(dimensions=2, walk_length=2, num_walks=5, seed=0)
    dim, counts = kg.embedding_box_counting_dimension("embedding", [0.5, 1.0])
    assert isinstance(dim, float)
    assert counts


def test_compute_poincare_embeddings():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    kg.add_entity("e1", "A")
    kg.add_entity("e2", "B")
    kg.link_entity("c1", "e1")
    kg.link_entity("c2", "e2")
    kg.compute_poincare_embeddings(dim=2, negative=2, epochs=5)
    vec = kg.graph.nodes["e1"].get("poincare_embedding")
    assert isinstance(vec, list)
    assert len(vec) == 2


def test_compute_graphsage_embeddings():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    kg.add_entity("e1", "A")
    kg.add_entity("e2", "B")
    kg.link_entity("c1", "e1")
    kg.link_entity("c2", "e2")
    kg.compute_graphsage_embeddings(dimensions=8, num_layers=1)
    vec = kg.graph.nodes["e1"].get("graphsage_embedding")
    assert isinstance(vec, list)
    assert len(vec) == 8


def test_compute_hyperbolic_hypergraph_embeddings_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "a")
    kg.add_chunk("d", "c2", "b")
    kg.add_hyperedge("he1", ["c1", "c2"])
    try:
        res = kg.compute_hyperbolic_hypergraph_embeddings(dim=2, negative=2, epochs=5)
    except RuntimeError:
        pytest.skip("gensim not installed")
    assert set(res) == {"c1", "c2"}
    assert "hyperbolic_embedding" in kg.graph.nodes["c1"]


def test_compute_transe_embeddings():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    kg.graph.add_edge("c1", "c2", relation="related")
    kg.compute_transe_embeddings(dimensions=8)
    emb = kg.graph.edges["c1", "c2"].get("transe_embedding")
    assert isinstance(emb, list)
    assert len(emb) == 8


def test_compute_distmult_embeddings():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    kg.graph.add_edge("c1", "c2", relation="related")
    kg.compute_distmult_embeddings(dimensions=8)
    emb = kg.graph.edges["c1", "c2"].get("distmult_embedding")
    assert isinstance(emb, list)
    assert len(emb) == 8


def test_box_counting_dimension_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    dim, counts = kg.box_counting_dimension([1])
    assert dim >= 0
    assert counts and counts[0][0] == 1


def test_persistence_diagrams_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    diagrams = kg.persistence_diagrams(max_dim=1)
    assert 0 in diagrams
    assert diagrams[0].shape[1] == 2


def test_persistence_wasserstein_distance_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    from datacreek.analysis import persistence_wasserstein_distance as _pwd

    if (
        _pwd.__module__ == "datacreek.analysis.fractal"
        and getattr(__import__("datacreek.analysis.fractal", fromlist=["gd"]), "gd")
        is None
    ):
        pytest.skip("gudhi not available")
    other = nx.path_graph(1)
    d = kg.persistence_wasserstein_distance(other)
    assert d >= 0


def test_topological_signature_hash_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    h = kg.topological_signature_hash(max_dim=1)
    assert isinstance(h, str)
    assert len(h) == 32


def test_graph_fourier_transform_methods():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "a")
    kg.add_chunk("d", "c2", "b")
    signal = {n: i for i, n in enumerate(kg.graph.nodes)}
    coeffs = kg.graph_fourier_transform(signal)
    recon = kg.inverse_graph_fourier_transform(coeffs)
    for val, node in zip(recon, kg.graph.nodes):
        assert pytest.approx(val, rel=1e-6) == signal[node]


def test_spectral_entropy_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    ent = kg.spectral_entropy()
    assert ent >= 0


def test_spectral_gap_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    gap = kg.spectral_gap()
    assert gap >= 0


def test_laplacian_energy_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    energy = kg.laplacian_energy()
    assert energy >= 0


def test_sheaf_laplacian_method():
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b", sheaf_sign=-1)
    L = kg.sheaf_laplacian()
    assert L.shape == (2, 2)
    assert L[0, 1] == 1


def test_sheaf_convolution_method():
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b", sheaf_sign=-1)
    features = {"a": [1.0], "b": [0.0]}
    out = kg.sheaf_convolution(features, alpha=0.5)
    assert set(out) == {"a", "b"}


def test_sheaf_neural_network_method():
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b", sheaf_sign=-1)
    feats = {"a": [1.0], "b": [0.0]}
    out = kg.sheaf_neural_network(feats, layers=2, alpha=0.5)
    assert set(out) == {"a", "b"}


def test_sheaf_cohomology_method():
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b", sheaf_sign=1)
    h1 = kg.sheaf_cohomology()
    assert isinstance(h1, int) and h1 >= 0


def test_resolve_sheaf_obstruction_method():
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b", sheaf_sign=1)
    kg.graph.add_edge("b", "c", sheaf_sign=1)
    kg.graph.add_edge("c", "a", sheaf_sign=1)
    before = kg.sheaf_cohomology()
    after = kg.resolve_sheaf_obstruction(max_iter=5)
    assert after <= before


def test_sheaf_consistency_score_method():
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b", sheaf_sign=1)
    score = kg.sheaf_consistency_score()
    assert 0.0 <= score <= 1.0


def test_sheaf_cohomology_blocksmith_method():
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b", sheaf_sign=1)
    val = kg.sheaf_cohomology_blocksmith(block_size=10)
    assert isinstance(val, int) and val >= 0


def test_sheaf_consistency_score_batched_method():
    kg = KnowledgeGraph()
    kg.graph.add_edge("0", "1", sheaf_sign=1)
    kg.graph.add_edge("1", "2", sheaf_sign=1)
    scores = kg.sheaf_consistency_score_batched([["0", "1"], ["1", "2"]])
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_spectral_bound_exceeded_method():
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b", sheaf_sign=1)
    flag = kg.spectral_bound_exceeded(2, 1.0)
    assert flag


def test_path_to_text_method():
    kg = KnowledgeGraph()
    kg.graph.add_node("a", text="A")
    kg.graph.add_node("b", text="B")
    kg.graph.add_edge("a", "b", relation="rel")
    sent = kg.path_to_text(["a", "b"])
    assert "A" in sent and "B" in sent


def test_neighborhood_to_sentence_method():
    kg = KnowledgeGraph()
    kg.graph.add_node("a", text="A")
    kg.graph.add_node("b", text="B")
    kg.graph.add_edge("a", "b", relation="rel")
    sent = kg.neighborhood_to_sentence(["a", "b"])
    assert "A" in sent and "B" in sent


def test_subgraph_to_text_method():
    kg = KnowledgeGraph()
    kg.graph.add_node("a", text="A")
    kg.graph.add_node("b", text="B")
    kg.graph.add_edge("a", "b", relation="rel")
    txt = kg.subgraph_to_text(["a", "b"])
    assert "A" in txt and "B" in txt


def test_graph_to_text_method():
    kg = KnowledgeGraph()
    kg.graph.add_node("a", text="A")
    kg.graph.add_node("b", text="B")
    kg.graph.add_edge("a", "b", relation="rel")
    txt = kg.graph_to_text()
    assert "A" in txt and "B" in txt


def test_auto_tool_calls_method():
    kg = KnowledgeGraph()
    kg.graph.add_node("a", text="search cats")
    out = kg.auto_tool_calls("a", [("search", r"search\s+\w+")])
    assert "[TOOL:search" in out
    assert kg.graph.nodes["a"]["text"] == out


def test_auto_tool_calls_all_method():
    kg = KnowledgeGraph()
    kg.graph.add_node("a", text="search cats")
    kg.graph.add_node("b", text="search dogs")
    result = kg.auto_tool_calls_all([("search", r"search\s+\w+")])
    assert set(result) == {"a", "b"}
    for text in result.values():
        assert "[TOOL:search" in text


def test_apply_perception_all_method():
    kg = KnowledgeGraph()
    kg.graph.add_node("a", text="hello")
    kg.graph.add_node("b", text="world")
    updated = kg.apply_perception_all(lambda t: t.upper(), perception_id="p")
    assert set(updated) == {"a", "b"}
    assert kg.graph.nodes["a"]["text"] == "HELLO"
    assert kg.graph.nodes["b"]["text"] == "WORLD"


def test_fractalize_level_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    coarse, mapping = kg.fractalize_level(1)
    assert coarse.number_of_nodes() >= 1
    assert "d" in mapping


def test_fractalize_optimal_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    coarse, mapping, radius = kg.fractalize_optimal([1, 2])
    assert coarse.number_of_nodes() >= 1
    assert "d" in mapping
    assert radius in {1, 2}


def test_optimize_topology_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "a")
    kg.add_chunk("d", "c2", "b")
    kg.graph.add_edge("c1", "c2", relation="perception_link")

    target = nx.cycle_graph(3)
    mapping = {i: n for i, n in enumerate(["c1", "c2", "d"])}
    target = nx.relabel_nodes(target, mapping)

    before = bottleneck_distance(kg.graph.to_undirected(), target)
    dist = kg.optimize_topology(target, max_iter=5, seed=0)
    after = bottleneck_distance(kg.graph.to_undirected(), target)
    assert after <= before
    assert dist == pytest.approx(after, rel=1e-9)


def test_optimize_topology_constrained_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "a")
    kg.add_chunk("d", "c2", "b")
    kg.graph.add_edge("c1", "c2", relation="perception_link")

    target = nx.cycle_graph(3)
    mapping = {i: n for i, n in enumerate(["c1", "c2", "d"])}
    target = nx.relabel_nodes(target, mapping)

    before = bottleneck_distance(kg.graph.to_undirected(), target)
    dist, diff = kg.optimize_topology_constrained(
        target, [1, 2], max_iter=5, seed=0, delta=1.0
    )
    after = bottleneck_distance(kg.graph.to_undirected(), target)
    assert after <= before
    assert dist == pytest.approx(after, rel=1e-9)
    assert diff >= 0


def test_validate_topology_method():
    import pytest

    from datacreek.analysis import bottleneck_distance

    if (
        bottleneck_distance.__module__ == "datacreek.analysis.fractal"
        and getattr(__import__("datacreek.analysis.fractal", fromlist=["gd"]), "gd")
        is None
    ):
        pytest.skip("gudhi not available")
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "a")
    kg.add_chunk("d", "c2", "b")
    kg.graph.add_edge("c1", "c2", relation="perception_link")

    target = nx.cycle_graph(3)
    mapping = {i: n for i, n in enumerate(["c1", "c2", "d"])}
    target = nx.relabel_nodes(target, mapping)

    dist, diff = kg.validate_topology(target, [1, 2])
    assert dist >= 0
    assert diff >= 0


def test_predict_links():
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Beethoven")
    kg.add_entity("e2", "Ludwig van Beethoven")
    kg.predict_links(threshold=0.4)
    assert kg.graph.has_edge("e1", "e2")
    # Node2Vec based prediction
    kg2 = KnowledgeGraph()
    kg2.add_entity("a1", "X")
    kg2.add_entity("a2", "X")
    kg2.add_document("d", source="s")
    kg2.add_chunk("d", "c1", "x")
    kg2.link_entity("c1", "a1")
    kg2.link_entity("c1", "a2")
    kg2.compute_node2vec_embeddings(dimensions=8, walk_length=4, num_walks=5, seed=42)
    kg2.predict_links(threshold=0.1, use_graph_embeddings=True)
    assert "embedding" in kg2.graph.nodes["a1"]


def test_mark_conflicts():
    kg = KnowledgeGraph()
    kg.add_entity("A", "A")
    kg.add_entity("B", "B")
    kg.add_entity("C", "C")
    kg.graph.add_edge("A", "B", relation="related")
    kg.graph.add_edge("A", "C", relation="related")
    marked = kg.mark_conflicting_facts()
    assert marked == 2
    assert kg.graph.edges["A", "B"].get("conflict") is True
    assert kg.graph.edges["A", "C"].get("conflict") is True


def test_consolidate_schema():
    kg = KnowledgeGraph()
    kg.add_entity("e1", "A")
    kg.add_entity("e2", "B")
    kg.graph.nodes["e1"]["type"] = "ENTITY"
    kg.graph.add_edge("e1", "e2", relation="RELATED")

    kg.consolidate_schema()

    assert kg.graph.nodes["e1"]["type"] == "entity"
    assert kg.graph.edges["e1", "e2"]["relation"] == "related"


def test_entity_helpers():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "Paris is nice")
    kg.add_chunk("d", "c2", "France is big")
    kg.add_entity("Paris", "Paris")
    kg.add_entity("France", "France")
    kg.link_entity("c1", "Paris")
    kg.link_entity("c2", "France")
    fid = kg.add_fact("Paris", "capital_of", "France")

    assert kg.get_chunks_for_entity("Paris") == ["c1"]
    assert kg.get_facts_for_entity("France") == [fid]


def test_fact_helpers():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "text")
    fid = kg.add_fact("A", "related", "B")
    kg.graph.add_edge("c1", fid, relation="has_fact")

    assert kg.get_facts_for_chunk("c1") == [fid]
    assert kg.get_facts_for_document("d") == [fid]
    assert kg.get_chunks_for_fact(fid) == ["c1"]
    assert set(kg.get_entities_for_fact(fid)) == {"A", "B"}


def test_find_facts():
    kg = KnowledgeGraph()
    kg.add_fact("A", "likes", "B", fact_id="f1")
    kg.add_fact("A", "likes", "C", fact_id="f2")

    assert set(kg.find_facts(subject="A", predicate="likes")) == {"f1", "f2"}


def test_fact_confidence_short_path():
    kg = KnowledgeGraph()
    kg.add_fact("X", "related", "Y")
    conf = kg.fact_confidence("X", "related", "Y")
    assert conf == 1.0

    kg.add_entity("Z", "Z")
    kg.graph.add_edge("Y", "Z", relation="other")
    conf2 = kg.fact_confidence("X", "related", "Z", max_hops=3)
    assert conf2 < 0.5


def test_annotate_mdl_levels_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    kg.annotate_mdl_levels([1, 2], max_levels=2)
    assert kg.graph.nodes["c1"].get("fractal_level")
    assert kg.graph.nodes["c2"].get("fractal_level")


def test_entity_lookup_helpers():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "Paris is nice")
    kg.add_chunk("doc", "c2", "Berlin is big")
    kg.add_entity("Paris", "Paris")
    kg.add_entity("Berlin", "Berlin")
    kg.link_entity("c1", "Paris")
    kg.link_entity("c2", "Berlin")

    assert kg.get_entities_for_chunk("c1") == ["Paris"]
    assert set(kg.get_entities_for_document("doc")) == {"Paris", "Berlin"}
    assert kg.get_documents_for_entity("Berlin") == ["doc"]


def test_entity_pages_helper():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "Paris is nice", page=2)
    kg.add_entity("Paris", "Paris")
    kg.link_entity("c1", "Paris")

    assert kg.get_pages_for_entity("Paris") == [2]


def test_fact_lookup_helpers():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "sec1")
    kg.add_chunk("doc", "c1", "A is B", section_id="sec1")
    fid = kg.add_fact("A", "is", "B")
    kg.graph.add_edge("c1", fid, relation="has_fact")

    assert kg.get_sections_for_fact(fid) == ["sec1"]
    assert kg.get_documents_for_fact(fid) == ["doc"]
    assert kg.get_pages_for_fact(fid) == [1]


def test_extract_entities():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "Albert Einstein was born in Ulm.")
    kg.extract_entities(model=None)
    ents = set(kg.get_entities_for_chunk("c1"))
    assert "Albert Einstein" in ents
    assert "Ulm" in ents


def test_link_chunks_by_entity():
    kg = KnowledgeGraph()
    kg.add_document("doc1", source="s")
    kg.add_document("doc2", source="s")
    kg.add_chunk("doc1", "c1", "Paris is big")
    kg.add_chunk("doc2", "c2", "I love Paris")
    kg.add_entity("Paris", "Paris")
    kg.link_entity("c1", "Paris")
    kg.link_entity("c2", "Paris")

    added = kg.link_chunks_by_entity()
    assert added == 1
    assert kg.graph.has_edge("c1", "c2")
    assert kg.graph.edges["c1", "c2"]["relation"] == "co_mentions"


def test_link_documents_by_entity():
    kg = KnowledgeGraph()
    kg.add_document("doc1", source="s")
    kg.add_document("doc2", source="s")
    kg.add_chunk("doc1", "c1", "Paris is big")
    kg.add_chunk("doc2", "c2", "I love Paris")
    kg.add_entity("Paris", "Paris")
    kg.link_entity("c1", "Paris")
    kg.link_entity("c2", "Paris")

    added = kg.link_documents_by_entity()
    assert added == 1
    assert kg.graph.has_edge("doc1", "doc2")
    assert kg.graph.edges["doc1", "doc2"]["relation"] == "co_mentions"


def test_link_sections_by_entity():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_section("doc", "s1")
    kg.add_section("doc", "s2")
    kg.add_chunk("doc", "c1", "Paris", section_id="s1")
    kg.add_chunk("doc", "c2", "Paris", section_id="s2")
    kg.add_entity("Paris", "Paris")
    kg.link_entity("c1", "Paris")
    kg.link_entity("c2", "Paris")

    added = kg.link_sections_by_entity()
    assert added == 1
    assert kg.graph.has_edge("s1", "s2")
    assert kg.graph.edges["s1", "s2"]["relation"] == "co_mentions"


def test_clean_chunk_texts():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s")
    kg.add_chunk("doc", "c1", "<p>Hello\n world</p>")

    changed = kg.clean_chunk_texts()
    assert changed == 1
    assert kg.graph.nodes["c1"]["text"] == "Hello world"


def test_normalize_date_fields():
    kg = KnowledgeGraph()
    kg.graph.add_node("e1", type="entity", birth_date="Jan 2, 2024")

    changed = kg.normalize_date_fields()

    assert changed == 1
    assert kg.graph.nodes["e1"]["birth_date"] == "2024-01-02"


def test_validate_coherence():
    kg = KnowledgeGraph()
    kg.graph.add_node("p", type="entity", birth_date="2024-01-01")
    kg.graph.add_node("c", type="entity", birth_date="2023-01-01")
    kg.graph.add_edge("p", "c", relation="parent_of")

    marked = kg.validate_coherence()

    assert marked == 1
    assert kg.graph.edges["p", "c"].get("inconsistent") is True


def test_link_authors_organizations():
    kg = KnowledgeGraph()
    kg.add_document("doc", source="s", author="alice", organization="acme")

    added = kg.link_authors_organizations()

    assert added == 1
    assert kg.graph.has_edge("alice", "acme")
    assert kg.graph.edges["alice", "acme"]["relation"] == "affiliated_with"


def test_remove_document_rebuilds_index():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello world")
    kg.add_chunk("d", "c2", "more text")
    assert kg.search_embeddings("hello", k=1, fetch_neighbors=False) == ["c1"]
    kg.remove_document("d")
    assert kg.search_embeddings("hello", k=1, fetch_neighbors=False) == []


def test_deduplicate_chunks_similarity():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "Hello world!")
    kg.add_chunk("d", "c2", "Hello world")

    removed = kg.deduplicate_chunks(similarity=0.9)
    assert removed == 1
    assert "c2" not in kg.graph.nodes or "c1" not in kg.graph.nodes


class _DummySession:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def run(self, query: str, **params: Any):
        self.queries.append(query)
        if "wcc.stream" in query:
            return [
                {"nodeId": 0, "componentId": 0},
                {"nodeId": 1, "componentId": 1},
            ]
        if "nodeSimilarity.stream" in query:
            return [{"node1": 0, "node2": 1, "similarity": 0.96}]
        if "adamicAdar.stream" in query:
            return [{"sourceNodeId": 0, "targetNodeId": 2, "score": 2.0}]
        if "degree.stream" in query:
            return [{"nodeId": 0, "score": 5}, {"nodeId": 1, "score": 1}]
        if "betweenness.stream" in query:
            return [{"nodeId": 0, "score": 0.8}, {"nodeId": 1, "score": 0.1}]
        if "triangleCount.stream" in query:
            return [
                {"nodeId": 0, "triangleCount": 0},
                {"nodeId": 1, "triangleCount": 2},
            ]
        if "hypergraph.linkprediction" in query:
            return []
        if "CREATE INDEX haa_pair" in query:
            return []
        if "RETURN count(r) AS c" in query:

            class _Rec(dict):
                def single(self):
                    return {"c": 2}

            return _Rec()
        if "RETURN n.id AS name" in query:

            class _Rec(dict):
                def single(self):
                    return {"name": "n"}

            return _Rec()
        if "MATCH (a)-[r]->(b) RETURN" in query:
            return [{"src": 0, "tgt": 1, "attention": 0.05}]
        if "id(n) IN $ids" in query:
            return [{"id": 0}]
        return []

    def __enter__(self) -> "_DummySession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass


class _DummyDriver:
    def __init__(self) -> None:
        self.session_obj = _DummySession()

    def session(self) -> _DummySession:
        return self.session_obj


def test_gds_quality_check_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")

    driver = _DummyDriver()
    res = kg.gds_quality_check(driver)
    assert set(res["removed_nodes"]) == {0, 1}
    assert res["duplicates"]
    assert res["suggested_links"]
    assert isinstance(res["hubs"], list)
    assert isinstance(res["weak_links"], list)
    assert isinstance(res["triangles_removed"], int)
    assert any("SUGGESTED_HYPER_AA" in q for q in driver.session_obj.queries)
    assert any("CREATE INDEX haa_pair" in q for q in driver.session_obj.queries)
    assert any("SET r.startNodeId" in q for q in driver.session_obj.queries)
    assert any("FOREACH (x IN tail" in q for q in driver.session_obj.queries)


def test_quality_check_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "a")
    kg.graph.add_node("iso")
    res = kg.quality_check(min_component_size=2)
    assert res["removed_nodes"] == 1


def test_atom_and_molecule_methods():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_atom("d", "a1", "hello", "NarrativeText")
    kg.add_atom("d", "a2", "world", "NarrativeText")
    kg.add_molecule("d", "m1", ["a1", "a2"])
    assert kg.get_atoms_for_document("d") == ["a1", "a2"]
    assert kg.get_molecules_for_document("d") == ["m1"]
    assert kg.graph.has_edge("m1", "a1")


def test_graph_information_bottleneck():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    for i in range(4):
        kg.add_atom("d", f"a{i}", str(i), "text")
    kg.compute_node2vec_embeddings(
        dimensions=2, walk_length=2, num_walks=5, workers=1, seed=0
    )
    labels = {f"a{i}": i % 2 for i in range(4)}
    loss = kg.graph_information_bottleneck(labels, beta=0.5)
    assert loss > 0


def test_graph_entropy_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    for i in range(3):
        kg.add_atom("d", f"a{i}", str(i), "text")
    h = kg.graph_entropy()
    assert h >= 0


def test_subgraph_entropy_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    for i in range(4):
        kg.add_atom("d", f"a{i}", str(i), "text")
    kg.graph.add_edge("a0", "a1")
    kg.graph.add_edge("a1", "a2")
    val = kg.subgraph_entropy(["a0", "a1", "a2"])
    assert val >= 0


def test_structural_entropy_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    for i in range(3):
        kg.add_atom("d", f"a{i}", str(i), "text")
    kg.graph.add_edges_from([("a0", "a1"), ("a1", "a2"), ("a0", "a2")])
    val = kg.structural_entropy(1)
    assert val >= 0


def test_prototype_subgraph():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    for i in range(4):
        kg.add_atom("d", f"a{i}", str(i), "text")
    kg.compute_node2vec_embeddings(
        dimensions=2, walk_length=2, num_walks=5, workers=1, seed=0
    )
    labels = {f"a{i}": i % 2 for i in range(4)}
    sub = kg.prototype_subgraph(labels, 1, radius=1)
    assert isinstance(sub, nx.Graph)


def test_diversification_score_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_atom("d", "a1", "x", "text")
    kg.add_atom("d", "a2", "y", "text")
    score = kg.diversification_score(["a1"], [1])
    assert isinstance(score, float)


def test_hyperbolic_neighbors_method():
    kg = KnowledgeGraph()
    kg.graph.add_node("a", hyperbolic_embedding=[0.1, 0.0])
    kg.graph.add_node("b", hyperbolic_embedding=[0.2, 0.05])
    kg.graph.add_node("c", hyperbolic_embedding=[0.9, 0.1])
    neighs = kg.hyperbolic_neighbors("a", k=1)
    assert neighs and neighs[0][0] == "b"


def test_hyperbolic_reasoning_method():
    kg = KnowledgeGraph()
    kg.graph.add_node("a", hyperbolic_embedding=[0.1, 0.0])
    kg.graph.add_node("b", hyperbolic_embedding=[0.2, 0.05])
    kg.graph.add_node("c", hyperbolic_embedding=[0.3, 0.06])
    path = kg.hyperbolic_reasoning("a", "c", max_steps=3)
    assert path[0] == "a" and path[-1] == "c"


def test_hyperbolic_hypergraph_reasoning_method():
    kg = KnowledgeGraph()
    kg.graph.add_node("a", hyperbolic_embedding=[0.1, 0.0])
    kg.graph.add_node("b", hyperbolic_embedding=[0.2, 0.05])
    kg.graph.add_node("c", hyperbolic_embedding=[0.3, 0.06])
    kg.graph.add_node("h", type="hyperedge", hyperbolic_embedding=[0.5, 0.1])
    kg.graph.add_edge("a", "h")
    kg.graph.add_edge("b", "h")
    kg.graph.add_edge("c", "h")
    path = kg.hyperbolic_hypergraph_reasoning("a", "c", max_steps=4)
    assert path[0] == "a" and path[-1] == "c"


def test_hyperbolic_multi_curvature_reasoning_method():
    kg = KnowledgeGraph()
    for n in ["a", "b", "c"]:
        kg.graph.add_node(
            n,
            **{
                "hyperbolic_embedding_-1": [0.1 * (ord(n) - 96), 0.0],
                "hyperbolic_embedding_-0.5": [0.05 * (ord(n) - 96), 0.01],
            },
        )
    path = kg.hyperbolic_multi_curvature_reasoning(
        "a", "c", curvatures=[-1, -0.5], max_steps=3
    )
    assert path[0] == "a" and path[-1] == "c"


def test_add_image_creates_caption_edge():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_image("d", "img1", "path.png", alt_text="a cat")
    caps = kg.get_captions_for_document("d")
    assert caps
    cap = caps[0]
    assert kg.graph.nodes[cap]["type"] == "caption"
    assert kg.graph.edges[cap, "img1"]["relation"] == "caption_of"


def test_cypher_ann_query_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello world")
    kg.index.build()

    class _Drv(_DummyDriver):
        pass

    res = kg.cypher_ann_query(
        _Drv(), "hello", "MATCH (n) WHERE id(n) IN $ids RETURN id(n) AS id"
    )
    assert res and res[0]["id"] == 0


def test_select_mdl_motifs_method():
    kg = KnowledgeGraph()
    kg.graph.add_edges_from([("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")])
    motif = nx.Graph([("a", "b"), ("b", "c"), ("a", "c")])
    selected = kg.select_mdl_motifs([motif])
    assert selected and isinstance(selected[0], nx.Graph)


def test_autotune_step_method():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    for i in range(3):
        kg.add_atom("d", f"a{i}", str(i), "text")
        kg.graph.nodes[f"a{i}"]["embedding"] = np.array([i, i], dtype=float)
    kg.graph.add_edge("a0", "a1")
    motifs = [kg.graph.subgraph(["a0", "a1"]).copy()]
    labels = {f"a{i}": i % 2 for i in range(3)}
    state = AutoTuneState()
    res = kg.autotune_step(labels, motifs, state, penalty_cfg={"lambda_sigma": 1.0})
    assert "cost" in res


def test_faiss_index_build_and_search():
    kg = KnowledgeGraph()
    kg.graph.add_node("a", embedding=[1.0, 0.0])
    kg.graph.add_node("b", embedding=[0.0, 1.0])
    try:
        kg.build_faiss_index()
    except RuntimeError:
        pytest.skip("faiss not installed")
    res = kg.search_faiss([1.0, 0.0], k=1)
    assert res == ["a"]


def test_fractal_coverage_methods():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "a")
    kg.add_chunk("d", "c2", "b")
    kg.annotate_fractal_levels([1], max_levels=1)
    cov = kg.fractal_coverage()
    assert 0.0 < cov <= 1.0
    cov2 = kg.ensure_fractal_coverage(1.0, [1], max_levels=1)
    assert cov2 >= 1.0


def test_fractalnet_compress_method():
    if KnowledgeGraph is None:
        pytest.skip("deps missing")
    kg = KnowledgeGraph()
    kg.graph.add_node("a", embedding=[0.0, 0.0], fractal_level=0)
    kg.graph.add_node("b", embedding=[0.2, 0.0], fractal_level=0)
    kg.graph.add_node("c", embedding=[1.0, 0.0], fractal_level=1)
    comp = kg.fractalnet_compress()
    assert len(comp) == 2


def test_mapper_cache_method(monkeypatch):
    calls: list[int] = []

    def fake_mapper_nerve(g, radius):
        calls.append(radius)
        return nx.Graph(), []

    monkeypatch.setattr("datacreek.analysis.mapper.mapper_nerve", fake_mapper_nerve)

    kg = KnowledgeGraph()
    kg.graph.add_nodes_from(range(3))
    kg.graph.add_edges_from([(0, 1), (1, 2)])

    kg.mapper_nerve(1)
    kg.mapper_nerve(1)
    assert len(calls) == 1

    kg.clear_mapper_cache()
    kg.mapper_nerve(1)
    assert len(calls) == 2


def test_svgp_ei_propose_method():
    kg = KnowledgeGraph()
    vec = kg.svgp_ei_propose(
        [([0.0, 0.0], 1.0)], [(0.0, 1.0), (0.0, 1.0)], m=10, n_samples=20
    )
    assert len(vec) == 2
    assert 0.0 <= vec[0] <= 1.0


def test_prune_fractalnet_weights_method():
    if KnowledgeGraph is None:
        pytest.skip("deps missing")
    kg = KnowledgeGraph()
    w = [1, -2, 3, -4]
    out = kg.prune_fractalnet_weights(w, ratio=0.5)
    assert len([x for x in out if x != 0]) == 2


def test_explain_node_method():
    if KnowledgeGraph is None:
        pytest.skip("deps missing")
    kg = KnowledgeGraph()
    kg.graph.add_node("n1", embedding=[0.0, 0.0])
    kg.graph.add_node("n2", embedding=[1.0, 0.0])
    kg.graph.add_edge("n1", "n2", relation="linked")

    result = kg.explain_node("n1", hops=1)
    assert "n1" in result["nodes"] and "n2" in result["nodes"]
    assert ("n1", "n2") in result["edges"] or ("n2", "n1") in result["edges"]
    assert any(
        key.startswith("n1->") or key.endswith("->n1") for key in result["attention"]
    )
