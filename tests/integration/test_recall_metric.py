import networkx as nx
import pytest

neo4j = pytest.importorskip("neo4j", reason="neo4j not installed")
redis = pytest.importorskip("redis", reason="redis not installed")

from datacreek.core.dataset import DatasetBuilder
from datacreek.core.knowledge_graph import KnowledgeGraph


def test_recall_metric_simple():
    G = nx.Graph()
    G.add_node(
        "a", embedding=[1, 0], graphwave_embedding=[1, 0], poincare_embedding=[1, 0]
    )
    G.add_node(
        "b", embedding=[0, 1], graphwave_embedding=[0, 1], poincare_embedding=[0, 1]
    )
    G.add_node(
        "c",
        embedding=[0.9, 0.1],
        graphwave_embedding=[0.9, 0.1],
        poincare_embedding=[0.9, 0.1],
    )
    kg = KnowledgeGraph(G)
    ds = DatasetBuilder(kg)
    recall = ds.recall_at_k(["a"], {"a": ["c"]}, k=1)
    assert 0.0 <= recall <= 1.0
