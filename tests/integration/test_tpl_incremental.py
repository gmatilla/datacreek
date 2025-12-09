import networkx as nx
import numpy as np
import pytest

from datacreek.analysis.tpl_incremental import tpl_incremental
from datacreek.core.dataset import DatasetBuilder, DatasetType


def _gudhi_available():
    try:
        import gudhi  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _gudhi_available(), reason="gudhi required")
def test_tpl_incremental_updates(monkeypatch):
    g = nx.path_graph(4)
    diags = tpl_incremental(g, radius=1)
    assert set(diags) == set(g.nodes())
    hashes = {n: g.nodes[n]["tpl_hash"] for n in g.nodes()}
    diags2 = tpl_incremental(g, radius=1)
    assert diags2 and {n: g.nodes[n]["tpl_hash"] for n in g.nodes()} == hashes
    g.add_edge(0, 3)
    diags3 = tpl_incremental(g, radius=1)
    assert any(g.nodes[n]["tpl_hash"] != hashes.get(n) for n in (0, 3))
    assert diags3


@pytest.mark.skipif(not _gudhi_available(), reason="gudhi required")
def test_tpl_incremental_wrapper(monkeypatch):
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "a")
    ds.add_chunk("d", "c2", "b")
    diags = ds.tpl_incremental(radius=1)
    assert diags
    assert any(e.operation == "tpl_incremental" for e in ds.events)


@pytest.mark.skipif(not _gudhi_available(), reason="gudhi required")
def test_tpl_incremental_global_diag():
    g = nx.cycle_graph(4)
    tpl_incremental(g, radius=1)
    assert "tpl_global" in g.graph
    full = []
    for n in g.nodes():
        full.extend(g.nodes[n]["tpl_diag"])
    assert np.allclose(
        np.asarray(full).reshape(-1, 2), np.asarray(g.graph["tpl_global"])
    )


@pytest.mark.skipif(not _gudhi_available(), reason="gudhi required")
def test_tpl_incremental_skip_unchanged(monkeypatch):
    import datacreek.analysis.tpl_incremental as tpli

    g = nx.path_graph(6)
    calls = 0

    def fake_persistence(*args, **kwargs):
        nonlocal calls
        calls += 1
        return np.array([[0.0, 1.0]])

    monkeypatch.setattr(tpli, "_local_persistence", fake_persistence)
    tpl_incremental(g, radius=1)
    assert calls == g.number_of_nodes()
    calls = 0
    g.add_edge(0, 5)
    tpl_incremental(g, radius=1)
    assert 0 < calls < g.number_of_nodes()
