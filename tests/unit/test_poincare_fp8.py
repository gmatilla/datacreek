import numpy as np

import datacreek.analysis.fractal as frac
from datacreek.analysis.compression import fp8_dequantize
from datacreek.core.knowledge_graph import KnowledgeGraph


def test_poincare_fp8(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_document("d", "src")
    kg.add_chunk("d", "c", "txt")

    monkeypatch.setattr(
        frac,
        "poincare_embedding",
        lambda G, **k: {"d": np.array([0.1, 0.2]), "c": np.array([0.2, 0.3])},
    )
    kg.compute_poincare_embeddings(dim=2)
    for node, expected in {"d": [0.1, 0.2], "c": [0.2, 0.3]}.items():
        q = np.array(kg.graph.nodes[node]["poincare_fp8"], dtype=np.int8)
        s = kg.graph.nodes[node]["poincare_scale"]
        restored = fp8_dequantize(q, s)
        assert np.allclose(restored, expected, atol=1e-2)
