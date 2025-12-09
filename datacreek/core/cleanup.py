from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from datacreek.models.results import KGCleanupStats

from .knowledge_graph import KnowledgeGraph

if TYPE_CHECKING:  # pragma: no cover - avoid circular import at runtime
    from .dataset import DatasetBuilder


def cleanup_knowledge_graph(
    kg: KnowledgeGraph,
    *,
    dataset_builder: Optional[DatasetBuilder] = None,
    resolve_threshold: float = 0.8,
    resolve_aliases: dict[str, list[str]] | None = None,
    dedup_similarity: float = 1.0,
    normalize_dates: bool = True,
    mark_conflicts: bool = False,
    validate: bool = False,
) -> KGCleanupStats:
    """Run standard cleanup operations on ``kg``.

    Parameters
    ----------
    dataset_builder:
        Optional :class:`DatasetBuilder` to log events on.
    resolve_threshold:
        Similarity used when merging entities.
    resolve_aliases:
        Mapping of canonical labels to aliases used during entity resolution.
    dedup_similarity:
        Similarity threshold for deduplicating chunk text.
    """
    from .knowledge_graph import verify_thresholds

    verify_thresholds()

    if dataset_builder is not None:
        removed, cleaned = dataset_builder.cleanup_graph(
            resolve_threshold=resolve_threshold,
            resolve_aliases=resolve_aliases,
            dedup_similarity=dedup_similarity,
            normalize_dates=normalize_dates,
            mark_conflicts=mark_conflicts,
            validate=validate,
        )
    else:
        removed = kg.deduplicate_chunks(dedup_similarity)
        cleaned = kg.clean_chunk_texts()
        if normalize_dates:
            kg.normalize_date_fields()
        kg.resolve_entities(threshold=resolve_threshold, aliases=resolve_aliases)
        if mark_conflicts:
            kg.mark_conflicting_facts()
        if validate:
            kg.validate_coherence()
    return KGCleanupStats(removed=removed, cleaned=cleaned)
