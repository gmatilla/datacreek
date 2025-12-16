"""Base KnowledgeGraph class composed of mixins."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ...utils.retrieval import EmbeddingIndex
from .analysis import AnalysisMixin
from .elements import ElementMixin
from .persistence import PersistenceMixin
from .search import SearchMixin

logger = logging.getLogger(__name__)

try:  # optional dependency for graph operations
    import networkx as nx
except Exception:  # pragma: no cover - minimal stub when networkx missing
    import types

    class _GraphStub:
        def __init__(self, *a, **k) -> None:
            pass

    nx = types.SimpleNamespace(DiGraph=_GraphStub, Graph=_GraphStub)  # type: ignore


@dataclass
class KnowledgeGraph(ElementMixin, SearchMixin, AnalysisMixin, PersistenceMixin):
    """Simple wrapper storing documents and chunks with source info."""

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    use_hnsw: bool = False
    index: EmbeddingIndex = field(init=False)
    faiss_index: object | None = field(init=False, default=None, repr=False)
    faiss_ids: list[str] | None = field(init=False, default=None, repr=False)
    faiss_index_type: str | None = field(init=False, default=None, repr=False)
    faiss_node_attr: str | None = field(init=False, default=None, repr=False)
    _mapper_cache: Dict[int, tuple[Any, list[set[str]]]] = field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:  # pragma: no cover - heavy
        """Initialize the internal embedding index."""

        self.index = EmbeddingIndex(use_hnsw=self.use_hnsw)
        self.faiss_index = None
        self.faiss_ids = None
        self.faiss_index_type = None
        self.faiss_node_attr = None
        self._mapper_cache = {}

    def has_label(self, label: str) -> bool:  # pragma: no cover - heavy
        """Return ``True`` if any node has ``label`` as a string label."""

        return any(
            data.get("label") == label or label in data.get("labels", [])
            for _, data in self.graph.nodes(data=True)
        )
