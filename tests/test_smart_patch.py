from datacreek.smart_patch import PATCHSET_REGISTRY, PatchCandidate, solve_patch_ilp


def test_solve_patch_ilp_selects_optimal_edges_and_stores_patchset():
    candidates = [
        PatchCandidate("e1", benefit=5, risk=4),
        PatchCandidate("e2", benefit=4, risk=3),
        PatchCandidate("e3", benefit=3, risk=2),
    ]
    patchset_id, selected = solve_patch_ilp(candidates, budget=5)
    assert set(selected) == {"e2", "e3"}
    assert PATCHSET_REGISTRY[patchset_id] == selected
