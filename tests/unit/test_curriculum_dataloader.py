import random
from statistics import mean

from training.curriculum_dataloader import CurriculumDataLoader, compute_difficulty


def _make_sample(h: int) -> dict:
    """Utility to build a synthetic sample with identical metrics."""
    return {"hops": h, "prompt": "x" * h, "centrality": h}


def test_curriculum_sorting() -> None:
    """Samples should be yielded from easiest to hardest."""
    dataset = [_make_sample(h) for h in [3, 1, 4, 2, 5]]
    loader = CurriculumDataLoader(dataset, batch_size=1)
    difficulties = [compute_difficulty(batch[0]) for batch in loader]
    assert difficulties == sorted(difficulties)


def test_curriculum_improves_perplexity() -> None:
    """Curriculum ordering lowers early average difficulty by ≥10%.

    This acts as a proxy for validation perplexity decreasing when the
    dataloader feeds easier examples first.
    """

    dataset = [_make_sample(h) for h in range(1, 11)]
    rng = random.Random(0)
    shuffled = dataset.copy()
    rng.shuffle(shuffled)
    baseline_avg = mean(compute_difficulty(s) for s in shuffled[:3])

    first_batches: list[dict] = []
    for _, batch in zip(range(3), CurriculumDataLoader(dataset, batch_size=1)):
        first_batches.extend(batch)
    curriculum_avg = mean(compute_difficulty(s) for s in first_batches)

    assert curriculum_avg <= 0.9 * baseline_avg
