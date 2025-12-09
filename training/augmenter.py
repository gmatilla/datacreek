"""Active-learning augmentation for high-loss samples.

This module identifies samples with high training loss and generates
paraphrased variants using a hypergraph of synonyms. The augmented
samples are re-inserted into the training pool every ``interval`` epochs
so that the model focuses on difficult examples.
"""

from __future__ import annotations

import math
from typing import List, Mapping, Sequence

try:  # optional dependency
    from transformers import pipeline
except Exception:  # pragma: no cover - dependency missing
    pipeline = None  # type: ignore

__all__ = ["ActiveLearningAugmenter"]


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Compute the percentile of a sequence of numbers.

    A simple linear interpolation implementation is used instead of
    relying on NumPy so the function has no external dependencies.
    """

    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * percentile / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


# Hugging Face model for natural language inference (contradiction detection)
_NLI_MODEL_ID = "facebook/bart-large-mnli"


def _load_nli_pipeline() -> None:
    """Load the Hugging Face NLI pipeline if available."""

    global _NLI_PIPE
    if "_NLI_PIPE" not in globals():
        globals()["_NLI_PIPE"] = None
    if globals()["_NLI_PIPE"] is None and pipeline is not None:
        globals()["_NLI_PIPE"] = pipeline("text-classification", model=_NLI_MODEL_ID)


def _is_contradiction(original: str, paraphrase: str) -> bool:
    """Return ``True`` if the paraphrase contradicts the original sentence."""

    _load_nli_pipeline()
    pipe = globals().get("_NLI_PIPE")
    if pipe is None:  # pragma: no cover - dependency missing
        return False
    try:
        res = pipe({"text": original, "text_pair": paraphrase})[0]
    except Exception:  # pragma: no cover - inference failure
        return False
    label = str(res.get("label", "")).lower()
    return "contradiction" in label


def _generate_variants(
    text: str, synonym_map: Mapping[str, Sequence[str]], k: int
) -> List[str]:
    """Create ``k`` paraphrased variants by substituting synonyms.

    Contradictory paraphrases are removed using a natural language
    inference model (:mod:`facebook/bart-large-mnli`).

    Parameters
    ----------
    text:
        The original sample.
    synonym_map:
        Mapping from token to a list of synonyms. The first synonym is
        used for deterministic behaviour.
    k:
        Number of variants to generate.
    """

    tokens = text.split()
    variants: List[str] = []
    for i in range(k):
        new_tokens = []
        for token in tokens:
            synonyms = synonym_map.get(token)
            if synonyms:
                new_tokens.append(synonyms[i % len(synonyms)])
            else:
                new_tokens.append(token)
        candidate = " ".join(new_tokens)
        if not _is_contradiction(text, candidate):
            variants.append(candidate)
    return variants


class ActiveLearningAugmenter:
    """Augment difficult samples at a fixed epoch interval.

    Generated paraphrases are checked with a natural language inference
    model and any that contradict the source sentence are discarded.

    Parameters
    ----------
    synonym_map:
        Mapping of tokens to their synonyms extracted from the
        hypergraph.
    k:
        Number of variants to create for each selected sample.
    percentile:
        Percentile threshold above which a sample is considered
        difficult. Defaults to 95 (p95).
    interval:
        Augmentation interval in epochs. Augmentation occurs only when
        ``epoch % interval == 0``.
    """

    def __init__(
        self,
        synonym_map: Mapping[str, Sequence[str]],
        *,
        k: int = 1,
        percentile: float = 95.0,
        interval: int = 2,
    ) -> None:
        self.synonym_map = synonym_map
        self.k = k
        self.percentile = percentile
        self.interval = interval

    def augment(
        self,
        samples: Sequence[str],
        losses: Sequence[float],
        epoch: int,
    ) -> List[str]:
        """Return the dataset with augmented variants if at the right epoch.

        Parameters
        ----------
        samples:
            Original training samples.
        losses:
            Loss value for each sample.
        epoch:
            Current epoch number.
        """

        if len(samples) != len(losses):
            raise ValueError("samples and losses must have the same length")
        if epoch % self.interval != 0:
            return list(samples)

        threshold = _percentile(losses, self.percentile)
        augmented = list(samples)
        for sample, loss in zip(samples, losses):
            if loss > threshold:
                augmented.extend(_generate_variants(sample, self.synonym_map, self.k))
        return augmented
