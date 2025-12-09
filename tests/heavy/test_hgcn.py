import numpy as np
import pytest

from datacreek.core.dataset_full import DatasetBuilder, DatasetType
from datacreek.core.knowledge_graph import KnowledgeGraph


@pytest.mark.heavy
def test_hgcn_sagnn_embeddings_shape():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    kg.add_chunk("d", "c1", "hello")
    kg.add_chunk("d", "c2", "world")
    kg.add_hyperedge("h", ["c1", "c2"])
    for n in ["c1", "c2"]:
        kg.graph.nodes[n]["embedding"] = [1.0, 0.0]

    res = kg.compute_hgcn_sagnn_embeddings(K=2, embed_dim=2, seed=0)
    assert isinstance(res, dict)
    assert len(res["h"]) == 4


@pytest.mark.heavy
def test_dataset_hgcn_wrapper_records_event():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "x")
    ds.add_chunk("d", "c2", "y")
    ds.add_hyperedge("h", ["c1", "c2"])
    for n in ["c1", "c2"]:
        ds.graph.graph.nodes[n]["embedding"] = [0.0, 1.0]

    out = ds.compute_hgcn_sagnn_embeddings(K=1, embed_dim=2, seed=0)
    assert "h" in out and len(out["h"]) == 4
    assert any(e.operation == "compute_hgcn_sagnn_embeddings" for e in ds.events)
