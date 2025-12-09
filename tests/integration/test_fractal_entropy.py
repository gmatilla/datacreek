import networkx as nx
import numpy as np

import datacreek.analysis.fractal as fractal


def test_graphwave_and_embedding_entropy():
    emb = {0: [1.0, 0.0], 1: [0.0, 1.0]}
    assert fractal.graphwave_entropy(emb) >= -1e-9
    assert np.isfinite(fractal.embedding_entropy(emb))


def test_lacunarity_and_coverage():
    g = nx.path_graph(4)
    for n in g.nodes():
        g.nodes[n]["fractal_level"] = n % 2
    assert fractal.graph_lacunarity(g, radius=1) >= 1.0
    cov = fractal.fractal_level_coverage(g)
    assert 0.0 < cov <= 1.0
