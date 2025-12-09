import importlib.abc
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from types import ModuleType

import networkx as nx

analysis_pkg = ModuleType("datacreek.analysis")
analysis_pkg.__path__ = [
    str(Path(__file__).resolve().parents[1] / "datacreek" / "analysis")
]
sys.modules["datacreek.analysis"] = analysis_pkg
index_stub = ModuleType("datacreek.analysis.index")
index_stub.recall10 = lambda *a, **k: 0.5
sys.modules["datacreek.analysis.index"] = index_stub

spec = importlib.util.spec_from_file_location(
    "datacreek.analysis.node2vec_tuning",
    Path(__file__).resolve().parents[1]
    / "datacreek"
    / "analysis"
    / "node2vec_tuning.py",
)
assert isinstance(spec.loader, importlib.abc.Loader)
node2vec_tuning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node2vec_tuning)


class DummyKG:
    def __init__(self):
        self.graph = nx.path_graph(2)
        self.graph.graph = {}

    def compute_node2vec_embeddings(self, *, p: float = 1.0, q: float = 1.0):
        for n in self.graph.nodes:
            self.graph.nodes[n]["embedding"] = [p + n, q + n]


def test_autotune_node2vec_runs(monkeypatch, tmp_path):
    kg = DummyKG()

    monkeypatch.setattr(node2vec_tuning, "BEST_PQ_PATH", tmp_path / "art.json")

    monkeypatch.setattr(
        node2vec_tuning, "recall10", lambda g, q, gt, gamma=0.5, eta=0.25: 0.5
    )
    best = node2vec_tuning.autotune_node2vec(
        kg, [0], {0: [1]}, max_evals=3, max_minutes=0.1
    )
    assert 0.1 <= best[0] <= 4.0
    assert 0.1 <= best[1] <= 4.0
    data = json.loads((tmp_path / "art.json").read_text())
    assert data["p"] == best[0]
    assert data["q"] == best[1]


def test_autotune_node2vec_early_stop(monkeypatch, tmp_path):
    kg = DummyKG()
    monkeypatch.setattr(node2vec_tuning, "BEST_PQ_PATH", tmp_path / "art.json")
    calls = []

    def _compute(p: float = 1.0, q: float = 1.0):
        calls.append((p, q))
        for n in kg.graph.nodes:
            kg.graph.nodes[n]["embedding"] = [1.0, 1.0]

    monkeypatch.setattr(kg, "compute_node2vec_embeddings", _compute)
    monkeypatch.setattr(
        node2vec_tuning, "recall10", lambda g, q, gt, gamma=0.5, eta=0.25: 0.6
    )
    node2vec_tuning.autotune_node2vec(
        kg, [0], {0: [1]}, var_threshold=0.05, max_evals=5
    )
    assert len(calls) == 2  # one for trial, one for final selection
    assert (tmp_path / "art.json").exists()


def test_autotune_node2vec_time_budget(monkeypatch, tmp_path):
    kg = DummyKG()
    monkeypatch.setattr(node2vec_tuning, "BEST_PQ_PATH", tmp_path / "art.json")
    times = [0.0, 70.0]

    def fake_time():
        return times.pop(0)

    monkeypatch.setattr(node2vec_tuning.time, "monotonic", fake_time)
    monkeypatch.setattr(
        node2vec_tuning, "recall10", lambda g, q, gt, gamma=0.5, eta=0.25: 0.5
    )
    node2vec_tuning.autotune_node2vec(kg, [0], {0: [1]}, max_minutes=0.01)
    assert (tmp_path / "art.json").exists()


def test_autotune_node2vec_timeout_two_iters(monkeypatch, tmp_path):
    kg = DummyKG()
    monkeypatch.setattr(node2vec_tuning, "BEST_PQ_PATH", tmp_path / "best.json")

    times = [0.0, 3.0, 7.0]

    def fake_time():
        return times.pop(0)

    calls = []

    def _compute(p: float = 1.0, q: float = 1.0):
        calls.append((p, q))
        for n in kg.graph.nodes:
            kg.graph.nodes[n]["embedding"] = [p, q]

    monkeypatch.setattr(node2vec_tuning.time, "monotonic", fake_time)
    monkeypatch.setattr(kg, "compute_node2vec_embeddings", _compute)
    monkeypatch.setattr(
        node2vec_tuning,
        "recall10",
        lambda g, q, gt, gamma=0.5, eta=0.25: 0.5,
    )
    monkeypatch.setattr(node2vec_tuning, "_var_norm", lambda g: 1.0)

    node2vec_tuning.autotune_node2vec(kg, [0], {0: [1]}, max_minutes=0.1)
    # two iterations plus final call
    assert len(calls) - 1 == 2
    assert (tmp_path / "best.json").exists()


def test_autotune_node2vec_artifact_hash(monkeypatch, tmp_path):
    kg = DummyKG()
    monkeypatch.setattr(node2vec_tuning, "BEST_PQ_PATH", tmp_path / "best.json")
    monkeypatch.setattr(
        node2vec_tuning, "recall10", lambda g, q, gt, gamma=0.5, eta=0.25: 0.5
    )
    node2vec_tuning.autotune_node2vec(kg, [0], {0: [1]}, max_evals=2, max_minutes=0.01)
    data = json.loads((tmp_path / "best.json").read_text())
    assert data["dataset"] == node2vec_tuning._dataset_hash(kg)
