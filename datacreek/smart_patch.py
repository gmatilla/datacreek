from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from uuid import uuid4

from datacreek.utils.deps import optional_import

pywraplp = optional_import("ortools.linear_solver.pywraplp")

logger = logging.getLogger(__name__)

_STORAGE_DIR = Path(
    os.getenv("SMART_PATCH_STORAGE", Path(__file__).resolve().parent / ".." / "cache")
).resolve()
_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
_PATCHSET_FILE = _STORAGE_DIR / "patchsets.json"
_PATCHSET_LOCK = threading.RLock()


def _load_patchsets() -> Dict[str, List[str]]:
    if not _PATCHSET_FILE.exists():
        return {}
    try:
        return json.loads(_PATCHSET_FILE.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load patchsets from %s", _PATCHSET_FILE)
        return {}


def _persist_patchsets(data: Dict[str, List[str]]) -> None:
    try:
        _PATCHSET_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        logger.exception("Failed to persist patchsets to %s", _PATCHSET_FILE)


# Global registry storing chosen patch sets by opaque identifier.
PATCHSET_REGISTRY: Dict[str, List[str]] = _load_patchsets()


def _register_patchset(patchset_id: str, selected: List[str]) -> None:
    with _PATCHSET_LOCK:
        PATCHSET_REGISTRY[patchset_id] = selected
        _persist_patchsets(PATCHSET_REGISTRY)


@dataclass(frozen=True)
class PatchCandidate:
    r"""Candidate edge that may be patched.

    Parameters
    ----------
    edge_id:
        Identifier for the edge.
    benefit:
        Consistency gain :math:`c_e` obtained if the patch is applied.
    risk:
        Associated risk weight :math:`w_e` used in the budget constraint.
    """

    edge_id: str
    benefit: float
    risk: float


def solve_patch_ilp(
    candidates: Sequence[PatchCandidate], budget: float
) -> Tuple[str, List[str]]:
    r"""Solve the smart-patch ILP and store the resulting patch set.

    The optimisation problem is

    .. math::

        \max_x \sum_e c_e x_e \quad\text{s.t.}\quad \sum_e w_e x_e \le B,\;x_e\in\{0,1\}

    where :math:`c_e` is the expected consistency improvement for edge ``e`` and
    :math:`w_e` is its risk. ``B`` is the total risk budget. Selected edge ids are
    registered in :data:`PATCHSET_REGISTRY` under a generated patch set id.

    Parameters
    ----------
    candidates:
        Iterable of patch candidates.
    budget:
        Maximum total risk :math:`B` allowed.

    Returns
    -------
    tuple[str, list[str]]
        The patch set identifier and the list of selected edge ids.
    """

    if pywraplp is None:
        raise RuntimeError("ortools.linear_solver.pywraplp is unavailable")
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:  # pragma: no cover - solver missing in environment
        raise RuntimeError("SCIP solver is not available")

    edge_vars = {
        cand.edge_id: solver.BoolVar(f"x_{cand.edge_id}") for cand in candidates
    }

    # Risk budget constraint: sum_e w_e x_e <= B
    solver.Add(
        sum(cand.risk * edge_vars[cand.edge_id] for cand in candidates) <= budget
    )

    # Objective: maximise total consistency gain.
    objective = solver.Objective()
    for cand in candidates:
        objective.SetCoefficient(edge_vars[cand.edge_id], cand.benefit)
    objective.SetMaximization()

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("ILP solver did not find optimal solution")

    selected = [
        cand.edge_id
        for cand in candidates
        if edge_vars[cand.edge_id].solution_value() > 0.5
    ]
    patchset_id = uuid4().hex
    _register_patchset(patchset_id, selected)
    return patchset_id, selected
