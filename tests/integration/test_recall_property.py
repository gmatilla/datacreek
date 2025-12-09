import networkx as nx

from datacreek.core.dataset import DatasetBuilder
from datacreek.core.knowledge_graph import KnowledgeGraph


def test_recall_property_set():
    kg = KnowledgeGraph(nx.Graph())
    kg.graph.add_node(
        "a", embedding=[1, 0], graphwave_embedding=[1, 0], poincare_embedding=[1, 0]
    )
    kg.graph.add_node(
        "b", embedding=[0, 1], graphwave_embedding=[0, 1], poincare_embedding=[0, 1]
    )
    ds = DatasetBuilder(kg)
    r = ds.recall_at_k(["a"], {"a": ["b"]}, k=10)
    assert kg.graph.graph.get("recall10") == r
