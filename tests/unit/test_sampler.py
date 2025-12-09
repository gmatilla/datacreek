import random

from dataset.sampler import StratifiedReservoirSampler


def test_stratified_sampler_distribution():
    """Sampler should keep class proportions within ±2 % of targets."""

    ratios = {"a": 0.5, "b": 0.3, "c": 0.2}
    k = 10_000
    sampler = StratifiedReservoirSampler(k=k, class_ratios=ratios, random_state=0)

    random.seed(0)
    classes = list(ratios.keys())
    weights = list(ratios.values())
    for i in range(100_000):
        label = random.choices(classes, weights)[0]
        sampler.add((i, label), label)

    samples = sampler.samples()
    assert len(samples) == k
    stats = sampler.stats()
    for cls, ratio in ratios.items():
        expected = ratio * k
        assert abs(stats[cls] - expected) / k <= 0.02
