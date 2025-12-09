import numpy as np
import pytest

from datacreek.core.dataset_full import DatasetBuilder, DatasetType


@pytest.mark.heavy
def test_dataset_learned_hyperbolic_embeddings():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "x")
    ds.add_chunk("d", "c2", "y")
    for n in ["c1", "c2"]:
        ds.graph.graph.nodes[n]["embedding"] = [float(ord(n[-1]))]

    out = ds.compute_learned_hyperbolic_embeddings(dim=2, epochs=5)
    assert set(out) == {"c1", "c2"}
    assert any(
        e.operation == "compute_learned_hyperbolic_embeddings" for e in ds.events
    )
