import importlib
import sys
import types
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from datacreek.analysis.tpl import tpl_correct_graph


@pytest.mark.heavy
def test_node2vec_autotune_stub(monkeypatch, tmp_path):
    """`autotune_node2vec` should run with stubbed optimizer."""
    skopt_stub = types.SimpleNamespace(
        Optimizer=lambda *a, **k: types.SimpleNamespace(
            ask=lambda: [1.0, 1.0], tell=lambda *a, **k: None
        )
    )
    sys.modules["skopt"] = skopt_stub
    node2vec_tuning = importlib.import_module("datacreek.analysis.node2vec_tuning")

    class DummyKG:
        def __init__(self):
            self.graph = nx.path_graph(2)
            self.graph.graph = {}

        def compute_node2vec_embeddings(self, *, p: float = 1.0, q: float = 1.0):
            for n in self.graph.nodes:
                self.graph.nodes[n]["embedding"] = [p, q]

    kg = DummyKG()
    monkeypatch.setattr(node2vec_tuning, "recall10", lambda *a, **k: 0.5)
    monkeypatch.setattr(node2vec_tuning, "BEST_PQ_PATH", tmp_path / "best.json")
    best = node2vec_tuning.autotune_node2vec(
        kg, [0], {0: [1]}, max_evals=2, max_minutes=0.01
    )
    assert len(best) == 2
    assert Path(tmp_path / "best.json").exists()


@pytest.mark.heavy
def test_nprobe_autotune_stub(monkeypatch):
    """`autotune_nprobe` should adjust `nprobe`."""
    nprobe_tuning = importlib.import_module("datacreek.analysis.nprobe_tuning")

    class DummyIndex:
        def __init__(self):
            self.nprobe = 32

        def search(self, x, k):
            return np.zeros((x.shape[0], k), dtype=int), np.zeros((x.shape[0], k))

    dummy = DummyIndex()
    monkeypatch.setattr(
        nprobe_tuning,
        "faiss",
        types.SimpleNamespace(
            IndexFlatIP=lambda d: types.SimpleNamespace(
                add=lambda x: None,
                search=lambda q, k: (
                    np.zeros((q.shape[0], k), dtype=int),
                    np.zeros((q.shape[0], k)),
                ),
            ),
            IndexIVFPQ=lambda *a, **k: dummy,
        ),
    )
    monkeypatch.setattr(
        nprobe_tuning,
        "Optimizer",
        lambda *a, **k: types.SimpleNamespace(
            ask=lambda: [64], tell=lambda *a, **k: None
        ),
    )
    monkeypatch.setattr(nprobe_tuning, "_compute_recall", lambda *a, **k: 0.6)
    best = nprobe_tuning.autotune_nprobe(
        dummy,
        np.eye(2, dtype=np.float32),
        np.eye(2, dtype=np.float32),
        k=1,
        target=0.5,
        max_evals=2,
    )
    assert best == 64
    assert dummy.nprobe == 64


@pytest.mark.heavy
def test_tpl_incremental_stub(monkeypatch):
    """`tpl_incremental` should annotate nodes with persistence."""
    tpl_incremental = importlib.import_module("datacreek.analysis.tpl_incremental")

    g = nx.path_graph(3)
    monkeypatch.setattr(
        tpl_incremental, "_local_persistence", lambda *a, **k: np.array([[0.0, 1.0]])
    )
    diags = tpl_incremental.tpl_incremental(g, radius=1)
    assert diags and set(diags) == set(g.nodes())


@pytest.mark.heavy
def test_tpl_correct_graph_stub(monkeypatch):
    """`tpl_correct_graph` should compute distances."""
    g1 = nx.path_graph(3)
    g2 = nx.cycle_graph(3)
    tpl_module = importlib.import_module("datacreek.analysis.tpl")
    monkeypatch.setattr(
        tpl_module, "_diagram", lambda g, dimension=1: np.array([[0.0, 1.0]])
    )
    res = tpl_module.tpl_correct_graph(g1, g2, epsilon=0.0, max_iter=1)
    assert res["distance_after"] <= res["distance_before"]


@pytest.mark.heavy
def test_compute_recall_basic():
    """`_compute_recall` should average recall across queries."""
    mod = importlib.import_module("datacreek.analysis.nprobe_tuning")
    res = np.array([[1, 2, 3], [3, 4, 5]])
    gt = [[1, 4, 3], [4, 3, 6]]
    assert mod._compute_recall(res, gt) == pytest.approx(4 / 6)


@pytest.mark.heavy
def test_profile_nprobe_stub(monkeypatch, tmp_path):
    """`profile_nprobe` should record recall and latency."""
    # provide stub Optimizer before importing the module
    sys.modules["skopt"] = types.SimpleNamespace(
        Optimizer=lambda *a, **k: types.SimpleNamespace(
            ask=lambda: [32], tell=lambda *a, **k: None
        )
    )
    mod = importlib.import_module("datacreek.analysis.nprobe_tuning")

    class Flat:
        def __init__(self, d):
            self.d = d

        def add(self, xb):
            self.xb = xb

        def search(self, q, k):
            # deterministic nearest neighbour ids
            return None, np.tile(np.arange(k), (q.shape[0], 1))

    class IVFPQ:
        def __init__(self):
            self.nprobe = 32

        def search(self, q, k):
            return None, np.tile(np.arange(k), (q.shape[0], 1))

    monkeypatch.setattr(
        mod,
        "faiss",
        types.SimpleNamespace(IndexFlatIP=Flat, IndexIVFPQ=lambda *a, **k: IVFPQ()),
    )
    times = iter([0.0, 0.1, 0.2, 0.3])
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(times))

    idx = IVFPQ()
    out = mod.profile_nprobe(
        idx,
        np.eye(2, dtype=np.float32),
        np.eye(2, dtype=np.float32),
        k=1,
        nprobes=[32, 64],
        path=tmp_path / "p.pkl",
    )
    assert out["nprobe"] == [32, 64]
    assert out["recall"] == [1.0, 1.0]
    assert out["latency"] == pytest.approx([0.1, 0.1])
    assert (tmp_path / "p.pkl").exists()


@pytest.mark.heavy
def test_sinkhorn_diagram_and_local(monkeypatch):
    """`sinkhorn_w1` and diagram helpers should compute basic distances."""
    tpl = importlib.import_module("datacreek.analysis.tpl")
    tpli = importlib.import_module("datacreek.analysis.tpl_incremental")

    class SimplexTree:
        def __init__(self):
            self.simplices = []

        def insert(self, simplex, filtration=0.0):
            self.simplices.append(simplex)

        def compute_persistence(self, persistence_dim_max=True):
            pass

        def persistence_intervals_in_dimension(self, dim):
            if dim == 1 and any(len(s) == 2 for s in self.simplices):
                return np.array([[0.0, 1.0]])
            return np.empty((0, 2))

    gd_stub = types.SimpleNamespace(SimplexTree=SimplexTree)
    monkeypatch.setattr(tpl, "gd", gd_stub)
    monkeypatch.setattr(tpli, "gd", gd_stub)

    g = nx.cycle_graph(3)
    diag = tpl._diagram(g)
    assert diag.shape == (1, 2)

    dist = tpl.sinkhorn_w1(diag, diag)
    assert dist == pytest.approx(0.0)

    diag_local = tpli._local_persistence(g, 0, radius=1)
    assert diag_local.shape == (1, 2)

    h1 = tpli._local_hash(g, 0, radius=1)
    g.add_edge(0, 2, timestamp=1.0)
    h2 = tpli._local_hash(g, 0, radius=1)
    assert h1 != h2

    diags = tpli.tpl_incremental(g, radius=1)
    assert g.graph["tpl_global"] == [[0.0, 1.0]] * len(g.nodes())
    assert set(diags) == set(g.nodes())


@pytest.mark.heavy
def test_tpl_correct_graph_branch(monkeypatch):
    """Branch in ``tpl_correct_graph`` should apply corrections."""
    mod = importlib.import_module("datacreek.analysis.tpl")

    monkeypatch.setattr(
        mod, "_diagram", lambda g, dimension=1: np.array([[0.0, float(len(g))]])
    )
    monkeypatch.setattr(mod, "generate_graph_rnn_like", lambda n, e: nx.path_graph(n))
    monkeypatch.setattr(mod, "resolve_sheaf_obstruction", lambda *a, **k: None)

    g = nx.path_graph(3)
    target = nx.cycle_graph(4)
    res = mod.tpl_correct_graph(g, target, epsilon=0.5, max_iter=1)
    assert res["corrected"]
    assert res["distance_after"] <= res["distance_before"]


@pytest.mark.heavy
def test_sinkhorn_and_diagram_edge_cases(monkeypatch):
    """Edge cases for Sinkhorn and diagram utilities."""
    tpl = importlib.import_module("datacreek.analysis.tpl")

    # empty diagrams
    assert tpl.sinkhorn_w1(np.empty((0, 2)), np.empty((0, 2))) == 0.0

    # diagram requires gudhi
    monkeypatch.setattr(tpl, "gd", None)
    try:
        tpl._diagram(nx.Graph())
    except RuntimeError:
        pass
    else:
        pytest.fail("RuntimeError not raised")

    # stub gudhi returning no intervals
    class SimplexTree:
        def insert(self, *a, **k):
            pass

        def compute_persistence(self, *a, **k):
            pass

        def persistence_intervals_in_dimension(self, d):
            return []

    gd_stub = types.SimpleNamespace(SimplexTree=SimplexTree)
    monkeypatch.setattr(tpl, "gd", gd_stub)
    out = tpl._diagram(nx.Graph())
    assert out.size == 0
