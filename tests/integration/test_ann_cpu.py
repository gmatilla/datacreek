"""Tests for CPU multi-probing parameters."""

import importlib


def test_choose_nprobe_multi_formula():
    mod = importlib.import_module("datacreek.analysis.hybrid_ann")
    n_cells = 1000
    tables = 16
    target = 0.9
    nprobe = mod.choose_nprobe_multi(n_cells, tables, target=target)
    recall = mod.expected_recall(nprobe, n_cells, tables)
    assert recall >= target
