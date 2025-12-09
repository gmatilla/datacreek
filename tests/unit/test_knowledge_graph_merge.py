import types

from datacreek.core.knowledge_graph import KnowledgeGraph


def test_merge_entity_nodes():
    kg = KnowledgeGraph()
    # create two entities and additional nodes to connect
    kg.add_entity("x", "node x")
    kg.add_entity("e1", "one")
    kg.add_entity("e2", "two")
    kg.add_entity("y", "node y")
    # edges pointing to and from e2
    kg.graph.add_edge("x", "e2", relation="rel")
    kg.graph.add_edge("e2", "y", relation="rel")
    kg._merge_entity_nodes("e1", "e2")
    assert "e2" not in kg.graph
    # edges should be redirected to e1
    assert kg.graph.has_edge("x", "e1")
    assert kg.graph.has_edge("e1", "y")
