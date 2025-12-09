import math

from datacreek.dp import allow_request, compute_epsilon, renyi_epsilon


def test_renyi_epsilon_basic():
    eps = [0.1, 0.2, 0.05]
    result = renyi_epsilon(eps, alphas=[2])
    expected = math.log(sum(math.exp((2 - 1) * e) for e in eps)) / (2 - 1)
    assert abs(result - expected) < 1e-9


def test_allow_request_gate():
    eps = [0.3, 0.4]
    assert not allow_request(eps, 1.0, alphas=[2])
    assert allow_request(eps, 1.1, alphas=[2])


def test_compute_epsilon_alias():
    values = [0.1, 0.2]
    assert compute_epsilon(values, alphas=[3]) == renyi_epsilon(values, alphas=[3])
