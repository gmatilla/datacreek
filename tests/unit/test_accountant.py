import math

import pytest

from datacreek.dp import accountant as acc


def test_renyi_epsilon_default_and_custom():
    # single epsilon should equal itself regardless of alpha
    assert acc.renyi_epsilon([0.5]) == pytest.approx(0.5)
    # custom alphas calculation
    alphas = [2, 3]
    eps = acc.renyi_epsilon([0.5, 0.2], alphas=alphas)
    manual = min(
        math.log(sum(math.exp((a - 1) * e) for e in [0.5, 0.2])) / (a - 1)
        for a in alphas
    )
    assert eps == pytest.approx(manual)


def test_compute_and_allow_request():
    eps_list = [0.2, 0.3]
    eps = acc.compute_epsilon(eps_list, alphas=[2])
    assert eps == pytest.approx(acc.renyi_epsilon(eps_list, alphas=[2]))
    assert acc.allow_request(eps_list, eps + 0.1, alphas=[2])
    assert not acc.allow_request(eps_list, eps - 0.1, alphas=[2])
