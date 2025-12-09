import networkx as nx
import pytest

from datacreek.analysis.tpl import tpl_correct_graph
from datacreek.core.dataset import DatasetBuilder, DatasetType


def test_tpl_correct_graph_function():
    g1 = nx.path_graph(4)
    g2 = nx.cycle_graph(4)
    if (
        tpl_correct_graph.__module__ == "datacreek.analysis.tpl"
        and getattr(__import__("datacreek.analysis.fractal", fromlist=["gd"]), "gd")
        is None
    ):
        pytest.skip("gudhi not available")
    res = tpl_correct_graph(g1, g2, epsilon=0.0, max_iter=2)
    assert set(res) == {"distance_before", "distance_after", "corrected"}


def test_tpl_correct_graph_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "a")
    ds.add_chunk("d", "c2", "b")
    target = ds.graph.graph.to_undirected()
    try:
        res = ds.tpl_correct_graph(target, epsilon=0.0, max_iter=1)
    except RuntimeError:
        pytest.skip("gudhi not available")
    assert "distance_after" in res
    assert any(e.operation == "tpl_correct_graph" for e in ds.events)
