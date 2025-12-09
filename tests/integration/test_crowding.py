import numpy as np

from datacreek.analysis.governance import average_hyperbolic_radius
from datacreek.core.dataset import DatasetBuilder, DatasetType


def test_average_hyperbolic_radius_function():
    emb = {"a": [0.1, 0.0], "b": [0.0, 0.2]}
    radius = average_hyperbolic_radius(emb)
    assert radius > 0.0


def test_dataset_average_hyperbolic_radius_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_entity("e1", "A")
    ds.add_entity("e2", "B")
    ds.graph.graph.nodes["e1"]["poincare_embedding"] = [0.1, 0.0]
    ds.graph.graph.nodes["e2"]["poincare_embedding"] = [0.0, 0.2]
    r = ds.average_hyperbolic_radius()
    assert r > 0.0
    assert ds.events[-1].operation == "average_hyperbolic_radius"
