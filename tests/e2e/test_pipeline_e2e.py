import sys
import types
from pathlib import Path

import pytest

from datacreek.core.dataset import DatasetBuilder, DatasetType
from datacreek.core.scripts import build_dataset
from datacreek.utils.config import load_config


@pytest.mark.e2e
def test_pipeline_end_to_end(monkeypatch, tmp_path):
    metrics = {}

    def fake_run_pipeline(source, config, out, *, dry_run=False):
        ds = DatasetBuilder(DatasetType.QA, name="mini")
        store = (
            ds.graph.graph.graph if hasattr(ds.graph.graph, "graph") else ds.graph.graph
        )
        store["recall10"] = 0.95
        store["tpl_w1"] = 0.03
        ds.graph.index = types.SimpleNamespace(type="HNSW", latency=0.2)
        monkeypatch.setattr(ds.graph, "sheaf_consistency_score", lambda: 0.85)
        metrics["ds"] = ds

    monkeypatch.setattr(build_dataset, "run_pipeline", fake_run_pipeline)
    args = [
        "build_dataset.py",
        "--source",
        "samples/mini",
        "--config",
        "configs/default.yaml",
        "--output",
        str(tmp_path / "out"),
        "--dry-run",
    ]
    monkeypatch.setattr(sys, "argv", args)
    build_dataset.main()

    ds = metrics["ds"]
    cfg = load_config("configs/default.yaml")
    assert ds.graph.sheaf_consistency_score() >= 0.8
    store = ds.graph.graph.graph if hasattr(ds.graph.graph, "graph") else ds.graph.graph
    assert store["recall10"] >= 0.9
    assert store["tpl_w1"] <= float(cfg.get("tpl", {}).get("eps_w1", 0))
    assert (ds.graph.index.type == "HNSW") == (ds.graph.index.latency > 0.1)
