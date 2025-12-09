from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
from uuid import uuid4

from ortools.linear_solver import pywraplp

# Global registry storing chosen patch sets by opaque identifier.
PATCHSET_REGISTRY: dict[str, List[str]] = {}


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
    PATCHSET_REGISTRY[patchset_id] = selected
    return patchset_id, selected
