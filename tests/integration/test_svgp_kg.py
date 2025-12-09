import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datacreek.core.knowledge_graph import KnowledgeGraph


def test_svgp_ei_propose_method_simple():
    kg = KnowledgeGraph()
    vec = kg.svgp_ei_propose(
        [([0.0, 0.0], 1.0)], [(0.0, 1.0), (0.0, 1.0)], m=5, n_samples=10
    )
    assert len(vec) == 2
    assert 0.0 <= vec[0] <= 1.0
