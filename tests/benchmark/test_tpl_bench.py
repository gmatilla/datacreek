import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("gudhi", types.ModuleType("gudhi"))
ROOT = Path(__file__).resolve().parents[1]


def test_bench_tpl_speedup(tmp_path):
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    (stub_dir / "requests.py").write_text("")
    (stub_dir / "watchdog").mkdir()
    (stub_dir / "watchdog/__init__.py").write_text("")
    (stub_dir / "watchdog/events.py").write_text("")
    (stub_dir / "watchdog/observers.py").write_text("")

    spec = importlib.util.spec_from_file_location(
        "datacreek.analysis.tpl_incremental",
        ROOT / "datacreek/analysis/tpl_incremental.py",
    )
    tpli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tpli)
    sys.modules["datacreek"] = types.ModuleType("datacreek")
    sys.modules["datacreek.analysis"] = types.ModuleType("datacreek.analysis")
    sys.modules["datacreek.analysis.tpl_incremental"] = tpli

    spec2 = importlib.util.spec_from_file_location(
        "bench_tpl_incremental",
        ROOT / "scripts/bench_tpl_incremental.py",
    )
    bench = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(bench)

    def stub(*a, **k):
        import time

        time.sleep(0.005)
        return tpli.np.array([[0.0, 1.0]])

    tpli._local_persistence = stub
    res = bench.run_bench([0.1])
    ratio = float(res["0.1"])
    assert ratio >= 2.0
