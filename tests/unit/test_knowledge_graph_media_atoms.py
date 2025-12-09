import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


def test_add_image_creates_caption_edges():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src")
    kg.add_image("d1", "img1", "/path", alt_text="Caption")
    # Caption node should be created and linked
    caption_id = "img1_caption"
    assert caption_id in kg.graph
    assert ("d1", caption_id) in kg.graph.edges
    assert (caption_id, "img1") in kg.graph.edges
    assert kg.graph.edges[caption_id, "img1"]["relation"] == "caption_of"


def test_remove_chunk_renumbers_sequences():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src")
    kg.add_chunk("d1", "c1", "text1")
    kg.add_chunk("d1", "c2", "text2")
    kg.add_chunk("d1", "c3", "text3")
    # initial sequences
    assert kg.graph.edges["d1", "c1"]["sequence"] == 0
    assert kg.graph.edges["d1", "c2"]["sequence"] == 1
    assert kg.graph.edges["d1", "c3"]["sequence"] == 2
    kg.remove_chunk("c1")
    # c2 should now be first with no predecessor
    assert ("d1", "c2") in kg.graph.edges and kg.graph.edges["d1", "c2"][
        "sequence"
    ] == 0
    assert not any(
        kg.graph.edges[cid, "c2"].get("relation") == "next_chunk"
        for cid in kg.graph.predecessors("c2")
        if cid != "d1"
    )
    assert kg.graph.edges["d1", "c3"]["sequence"] == 1


def test_atom_molecule_and_hyperedge_simplex():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src")
    kg.add_atom("d1", "a1", "A1", "word")
    kg.add_atom("d1", "a2", "A2", "word")
    kg.add_molecule("d1", "m1", ["a1", "a2"])
    assert ("m1", "a1") in kg.graph.edges
    assert ("m1", "a2") in kg.graph.edges
    kg.add_hyperedge("he1", ["a1", "a2"])
    kg.add_simplex("s1", ["a1", "a2"])
    assert ("he1", "a1") in kg.graph.edges
    assert kg.graph.nodes["s1"]["dimension"] == 1
