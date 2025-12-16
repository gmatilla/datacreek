"""Utility to check availability of heavy dependencies before auditing."""

from __future__ import annotations

import importlib
from typing import Iterable, Sequence


MODULES: Sequence[str] = [
    # Core dependencies
    "torch",
    "faiss",
    "gudhi",
    "scipy",
    # Analyses flagged in AGENTS #10
    "datacreek.analysis.graphwave_bandwidth",
    "datacreek.analysis.graphwave_cuda",
    "datacreek.analysis.hyper_sagnn_cuda",
    "datacreek.analysis.chebyshev_diag",
    "datacreek.analysis.mapper",
    "datacreek.analysis.tda_fast",
    "datacreek.analysis.tpl",
    "datacreek.analysis.tpl_incremental",
    "datacreek.analysis.information",
    "datacreek.analysis.multiview",
    "datacreek.analysis.symmetry",
    "datacreek.analysis.compression",
    "datacreek.analysis.governance",
    "datacreek.analysis.rollback",
    "datacreek.analysis.ingestion",
    "datacreek.analysis.privacy",
    "datacreek.analysis.explain_viz",
]


def _check(module_name: str) -> str:
    try:
        importlib.import_module(module_name)
        return f"[OK]      {module_name}"
    except Exception as exc:
        return f"[MISSING] {module_name} ({exc.__class__.__name__})"


def main(mods: Iterable[str] | None = None) -> int:
    mods = mods or MODULES
    for module in mods:
        print(_check(module))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
