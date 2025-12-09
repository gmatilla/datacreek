import networkx as nx

from datacreek.analysis.fractal import tpl_motif_injection
from datacreek.analysis.generation import sheaf_score
from datacreek.analysis.sheaf import sheaf_laplacian
from datacreek.analysis.tpl import tpl_correct_graph
from datacreek.core.knowledge_graph import KnowledgeGraph


def test_tpl_cycle_injection():
    kg = KnowledgeGraph()
    kg.graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    tpl_correct_graph(kg.graph, kg.graph, epsilon=0.0, max_iter=1)
    score = tpl_motif_injection(kg.graph, {"tpl": {"rnn_size": 3}})
    Delta = sheaf_laplacian(kg.graph)
    s_sheaf = sheaf_score([0.0] * Delta.shape[0], Delta)
    assert score >= 0.0
    assert s_sheaf >= 0.8
    assert kg.has_label("RNN_PATCH")
