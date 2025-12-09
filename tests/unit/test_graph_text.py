import networkx as nx

import datacreek.utils.graph_text as gt


def build_graph():
    g = nx.Graph()
    g.add_node("A", text="Alice")
    g.add_node("B", text="Bob")
    g.add_node("C", text="Carol")
    g.add_node("D", text="Dan")
    g.add_edge("A", "B", relation="knows")
    g.add_edge("B", "C", relation="likes")
    g.add_edge("B", "D", relation="friend")
    return g


def test_neighborhood_sentence_basic():
    g = build_graph()
    sent = gt.neighborhood_to_sentence(g, ["A", "B", "C"], depth=0)
    assert sent == "Alice knows Bob likes Carol."


def test_neighborhood_sentence_with_neighbors():
    g = build_graph()
    sent = gt.neighborhood_to_sentence(g, ["A", "B", "C"], depth=1)
    assert "friend Dan" in sent
    assert sent.startswith("Alice knows") and sent.endswith("Carol.")


def test_neighborhood_sentence_missing_edge():
    g = build_graph()
    sent = gt.neighborhood_to_sentence(g, ["A", "C"], depth=0)
    assert sent == "Alice -> Carol."


def test_subgraph_to_text():
    g = build_graph()
    summary = gt.subgraph_to_text(g, ["A", "B", "C"])
    assert "Alice knows Bob" in summary and summary.endswith("Carol.")


def test_graph_to_text():
    g = build_graph()
    text = gt.graph_to_text(g)
    assert "Alice knows Bob" in text and text.endswith("Dan.")
