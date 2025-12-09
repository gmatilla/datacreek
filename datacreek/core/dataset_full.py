# pydocstyle: ignore=D100,D101,D102,D103,D107,D401
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

try:  # optional networkx dependency
    import networkx as nx
except Exception:  # pragma: no cover - optional dependency missing
    nx = None  # type: ignore

try:  # optional numpy dependency
    import numpy as np
except Exception:  # pragma: no cover - optional dependency missing
    np = None  # type: ignore

from ..analysis.monitoring import update_metric
from ..utils import push_metrics

if TYPE_CHECKING:
    from .ingest import IngestOptions

try:  # optional Redis dependency
    import redis
except Exception:  # pragma: no cover - optional dependency missing

    class _RedisStub:
        Redis = object

    redis = _RedisStub()  # type: ignore

try:  # optional Neo4j dependency
    from neo4j import Driver
except Exception:  # pragma: no cover - optional dependency missing
    Driver = object  # type: ignore

# Base directory for cached artifacts
CACHE_ROOT = os.environ.get("DATACREEK_CACHE", "./cache")
from datacreek.utils.config import load_config

from ..backends import get_redis_graph

try:  # optional redisgraph dependency
    from redisgraph import Edge as RGEdge
    from redisgraph import Graph as RGGraph
    from redisgraph import Node as RGNode
except Exception:  # pragma: no cover - optional
    RGGraph = None
    RGNode = None
    RGEdge = None

from ..models import LLMService
from ..models.stage import DatasetStage
from ..pipelines import DatasetType, PipelineStep
from ..security.dp_budget import DPBudgetManager
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 64
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass
class HistoryEvent:
    """Record a dataset modification for auditing."""

    operation: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    params: Optional[Dict[str, Any]] | None = None


@dataclass
class Atom:
    """Minimal logical atom ``(d, m)`` used during ingestion."""

    content: str
    id: str
    lang: str | None = None
    media: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class InvariantPolicy:
    r"""Thresholds used by :meth:`monitor_and_remediate`.

    Parameters
    ----------
    entropy_max:
        Upper bound on graph entropy :math:`H = -\sum_i p_i \log p_i`.
    gap_min:
        Lower bound on the normalized Laplacian spectral gap :math:`\lambda_1`.
    fractal_range:
        Acceptable interval for the box-counting dimension
        :math:`d_B = -\partial_{\log l_B} \log N_B(l_B)`.
    loops:
        Maximum number of remediation iterations.
    """

    entropy_max: float = 8.0
    gap_min: float = 0.05
    fractal_range: tuple[float, float] = (1.0, 4.0)
    loops: int = 3


@dataclass
class DatasetBuilder:
    """Manage a dataset under construction with its own knowledge graph."""

    dataset_type: DatasetType
    id: str = field(default_factory=lambda: secrets.token_hex(8))
    name: Optional[str] = None
    graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    use_hnsw: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    owner_id: int | None = None
    history: List[str] = field(default_factory=list)
    events: List[HistoryEvent] = field(default_factory=list)
    feedback: List[Dict[str, Any]] = field(default_factory=list)
    # track how often specific edges are traversed during generation
    edge_usage: Dict[str, int] = field(default_factory=dict)
    versions: List[Dict[str, Any]] = field(default_factory=list)
    ingested_docs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stage: DatasetStage = DatasetStage.CREATED
    redis_client: redis.Redis | None = field(default=None, repr=False)
    neo4j_driver: Driver | None = field(default=None, repr=False, compare=False)
    redis_key: str | None = field(default=None, repr=False)
    policy: InvariantPolicy = field(default_factory=InvariantPolicy)
    llm_service: LLMService | None = field(default=None, repr=False)
    auto_monitor: bool = False
    dp_budgets: DPBudgetManager = field(default_factory=DPBudgetManager)
    # stop signal used by the asynchronous policy monitor
    _policy_event: asyncio.Event | None = field(default=None, repr=False, compare=False)
    # background thread running the policy monitor
    _policy_thread: threading.Thread | None = field(
        default=None, repr=False, compare=False
    )

    def __del__(self) -> None:
        """Ensure background monitoring is terminated when the builder is collected."""
        try:
            self.stop_policy_monitor_thread()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    # ------------------------------------------------------------------
    # Name validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_name(name: str) -> str:
        """Validate that ``name`` is a safe identifier."""
        if len(name) > MAX_NAME_LENGTH or not NAME_PATTERN.match(name):
            raise ValueError(f"Invalid dataset name: {name}")
        return name

    def __post_init__(self) -> None:
        """Ensure persistence backends are configured when required."""
        if self.name:
            self.validate_name(self.name)
        if self.graph.use_hnsw != self.use_hnsw:
            self.graph.use_hnsw = self.use_hnsw
            self.graph.__post_init__()
        require = os.getenv("DATACREEK_REQUIRE_PERSISTENCE", "1") != "0"
        if require and (self.redis_client is None or self.neo4j_driver is None):
            raise ValueError("Redis and Neo4j must be configured")
        if self.auto_monitor:
            try:
                self.start_policy_monitor_thread([1])
            except Exception:  # pragma: no cover - background thread failure
                logger.exception("Failed to start policy monitor thread")

    def configure_llm_service(
        self,
        *,
        provider: str = "vllm",
        profile: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
    ) -> None:
        """Instantiate and store an :class:`LLMService`."""

        self.llm_service = LLMService(
            provider=provider,
            profile=profile,
            api_base=api_base,
            model=model,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """Persist the dataset state to configured backends."""

        if self.redis_client and self.name:
            key = self.redis_key or f"dataset:{self.name}"
            try:
                pipe = self.redis_client.pipeline()
                self.to_redis(pipe, key)
                pipe.sadd("datasets", self.name)
                if self.owner_id is not None:
                    pipe.sadd(f"user:{self.owner_id}:datasets", self.name)
                pipe.execute()
            except Exception:  # pragma: no cover - persistence failure
                logger.exception("Failed to persist dataset %s", self.name)
        if self.neo4j_driver and self.name:
            try:
                self.graph.to_neo4j(
                    self.neo4j_driver,
                    dataset=self.name,
                )
            except Exception:  # pragma: no cover - Neo4j unavailable
                logger.exception("Failed to persist graph %s to Neo4j", self.name)
        graph = get_redis_graph(self.name)
        if graph is not None:
            try:
                DatasetBuilder.save_redis_graph.__wrapped__(self, graph)
            except Exception:  # pragma: no cover - RedisGraph unavailable
                logger.exception("Failed to persist graph %s to RedisGraph", self.name)

    def _touch(self) -> None:
        """Update the ``accessed_at`` timestamp in Redis."""

        if self.redis_client and self.redis_key:
            self.accessed_at = datetime.now(timezone.utc)
            try:
                self.redis_client.set(self.redis_key, json.dumps(self.to_dict()))
            except Exception:  # pragma: no cover - Redis failure
                logger.exception("Failed to update access time for %s", self.name)

    # ------------------------------------------------------------------
    # Decorators
    # ------------------------------------------------------------------

    @staticmethod
    def persist_after(func):
        """Decorator ensuring dataset state is persisted after ``func``."""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self._persist()
            return result

        return wrapper

    @staticmethod
    def monitor_after(radii: Iterable[int] | None = None):
        """Decorator running :meth:`_enforce_policy` after the wrapped call."""

        if radii is None:
            radii = [1]

        def decorator(func: Callable[[Any], Any]):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                result = func(self, *args, **kwargs)
                if self.policy.loops > 0:
                    try:
                        self._enforce_policy(radii)
                    except Exception:
                        logger.exception("Automatic policy enforcement failed")
                return result

            return wrapper

        return decorator

    def _record_event(self, operation: str, message: str, **params: Any) -> None:
        """Store a history event and log it."""

        event = HistoryEvent(operation, message, params=params or None)
        self.events.append(event)
        self.history.append(message)
        logger.info(message)
        if self.redis_client and self.name:
            key = f"dataset:{self.name}"
            data = {
                "operation": event.operation,
                "message": event.message,
                "timestamp": event.timestamp.isoformat(),
                "params": event.params,
            }
            try:
                self.redis_client.rpush(f"{key}:events", json.dumps(data))
                log = data | {"dataset": self.name}
                self.redis_client.rpush("dataset:events", json.dumps(log))
            except Exception:
                logger.exception("Failed to persist event %s", self.name)

        self._persist()

    def log_cycle_metrics(self) -> None:
        """Emit debug metrics and push to Prometheus."""

        metrics = {
            "sigma_db": float(self.graph.graph.get("fractal_sigma", 0.0)),
            "coverage_frac": float(self.graph.fractal_coverage()),
            "H_wave": float(self.graph.graph.get("gw_entropy", 0.0)),
            "sheaf_score": float(self.graph.sheaf_consistency_score()),
            "recall10": float(self.graph.graph.get("recall10", 0.0)),
            "tpl_w1": float(self.graph.graph.get("tpl_w1", 0.0)),
            "j_cost": float(self.graph.graph.get("j_cost", 0.0)),
        }
        metrics["fractal_sigma"] = metrics["sigma_db"]
        logger.debug(
            "sigma_dB=%.4f coverage_frac=%.3f H_wave=%.4f sheaf_score=%.4f recall10=%.3f tpl_w1=%.4f j_cost=%.4f",
            metrics["sigma_db"],
            metrics["coverage_frac"],
            metrics["H_wave"],
            metrics["sheaf_score"],
            metrics["recall10"],
            metrics["tpl_w1"],
            metrics["j_cost"],
        )
        for k, v in metrics.items():
            try:
                update_metric(k, float(v))
            except Exception:
                pass

    def generate_model_card(
        self,
        prune_ratio: float,
        cca_sha: str,
        dp_eps: int = 2,
        code_commit: str | None = None,
    ) -> Dict[str, float]:
        """Return model card metrics as a dictionary.

        Parameters
        ----------
        prune_ratio:
            Ratio of pruned edges.
        cca_sha:
            Canonical commit identifier of CCA data.
        dp_eps:
            Differential privacy budget.
        code_commit:
            Git SHA of the current code revision.
        """

        meta = self.graph.graph.graph
        card = {
            "sigma_db": float(meta.get("fractal_sigma", 0.0)),
            "H_wave": float(meta.get("gw_entropy", 0.0)),
            "bias_wasserstein": float(meta.get("bias_W", 0.0)),
            "dp_eps": float(dp_eps),
            "prune_ratio": float(prune_ratio),
            "cca_sha": cca_sha,
        }
        card["code_commit"] = code_commit or os.getenv("GIT_SHA", "unknown")
        self._record_event("model_card", "Model card generated", **card)
        return card

    @staticmethod
    def model_card_html(card: Dict[str, float]) -> str:
        """Return HTML representation using Jinja2 and Chart.js."""

        try:
            from jinja2 import Template
        except Exception:  # pragma: no cover - optional dependency
            return json.dumps(card, indent=2)

        tmpl = Template(
            """
            <html>
            <head>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            </head>
            <body>
            <h1>Model Card</h1>
            <ul>
            {% for k, v in card.items() %}
              <li>{{ k }}: {{ v }}</li>
            {% endfor %}
            </ul>
            <canvas id="bias" width="200" height="100"></canvas>
            <script>
            new Chart(document.getElementById('bias'), {
              type: 'bar',
              data: {labels:['bias'], datasets:[{data:[{{ card.bias_wasserstein }}]}]}
            });
            </script>
            </body>
            </html>
            """
        )
        return tmpl.render(card=card)

    @monitor_after()
    def add_document(
        self,
        doc_id: str,
        source: str,
        *,
        text: str | None = None,
        author: str | None = None,
        organization: str | None = None,
        checksum: str | None = None,
        uid: str | None = None,
    ) -> None:
        """Insert a document node in the dataset graph."""

        self.graph.add_document(
            doc_id,
            source,
            text=text,
            author=author,
            organization=organization,
            checksum=checksum,
            uid=uid,
        )
        self._record_event(
            "add_document",
            f"Added document {doc_id}",
            source=source,
        )

    @monitor_after()
    def add_section(
        self,
        doc_id: str,
        section_id: str,
        title: str | None = None,
        source: Optional[str] = None,
        *,
        page: int | None = None,
    ) -> None:
        """Insert a section node for ``doc_id``."""

        self.graph.add_section(
            doc_id, section_id, title=title, source=source, page=page
        )
        self._record_event(
            "add_section",
            f"Added section {section_id} to {doc_id}",
            source=source,
        )

    @monitor_after()
    def add_chunk(
        self,
        doc_id: str,
        chunk_id: str,
        text: str,
        source: Optional[str] = None,
        *,
        section_id: str | None = None,
        page: int | None = None,
        emotion: str | None = None,
        modality: str | None = None,
        entities: list[str] | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Insert a chunk node in the dataset graph.

        Parameters
        ----------
        chunk_overlap:
            Overlap used when splitting the source text. Stored for
            provenance during later processing steps.
        """

        self.graph.add_chunk(
            doc_id,
            chunk_id,
            text,
            source,
            section_id=section_id,
            page=page,
            emotion=emotion,
            modality=modality,
            entities=entities,
            chunk_overlap=chunk_overlap,
        )
        self._record_event(
            "add_chunk",
            f"Added chunk {chunk_id} to {doc_id}",
            source=source,
        )

    @monitor_after()
    def add_image(
        self,
        doc_id: str,
        image_id: str,
        path: str,
        *,
        page: int | None = None,
        alt_text: str | None = None,
    ) -> None:
        """Insert an image node in the dataset graph."""

        self.graph.add_image(doc_id, image_id, path, page=page, alt_text=alt_text)
        self._record_event("add_image", f"Added image {image_id} to {doc_id}")

    @monitor_after()
    def add_audio(
        self,
        doc_id: str,
        audio_id: str,
        path: str,
        *,
        page: int | None = None,
        lang: str | None = None,
    ) -> None:
        """Insert an audio node in the dataset graph."""

        self.graph.add_audio(doc_id, audio_id, path, page=page, lang=lang)
        self._record_event("add_audio", f"Added audio {audio_id} to {doc_id}")

    def ingest_text_atoms(
        self, path: str, doc_id: str
    ) -> list[str]:  # pragma: no cover
        """Parse ``path`` into textual atoms and add them under ``doc_id``."""

        from ..analysis.ingestion import partition_files_to_atoms

        atoms = []
        for idx, text in enumerate(partition_files_to_atoms(path)):
            atom_id = f"{doc_id}_a{idx}"
            self.add_atom(doc_id, atom_id, text, "text")
            atoms.append(atom_id)
        return atoms

    def ingest_code_atoms(
        self, path: str, doc_id: str
    ) -> list[str]:  # pragma: no cover
        """Parse Python code into atoms (functions/classes) under ``doc_id``."""

        from ..analysis.ingestion import parse_code_to_atoms

        atoms = []
        for idx, text in enumerate(parse_code_to_atoms(path)):
            atom_id = f"{doc_id}_c{idx}"
            self.add_atom(doc_id, atom_id, text, "code")
            atoms.append(atom_id)
        return atoms

    def add_atom(  # pragma: no cover
        self,
        doc_id: str,
        atom_id: str,
        text: str,
        element_type: str,
        source: Optional[str] = None,
        *,
        page: int | None = None,
        lang: str | None = None,
        timestamp: datetime | None = None,
        emotion: str | None = None,
        modality: str | None = None,
        entities: list[str] | None = None,
    ) -> None:
        """Insert an atom node with optional metadata.

        Parameters
        ----------
        doc_id:
            ID of the parent document.
        atom_id:
            Identifier of the new atom.
        text:
            Raw textual content :math:`d`.
        element_type:
            Media or element type stored in ``m``.
        page:
            Page number if extracted from paginated media.
        lang:
            Language code for the content.
        timestamp:
            Timestamp when the atom was created. Defaults to ``now``.
        emotion, modality, entities:
            Optional semantic annotations.
        """

        self.graph.add_atom(
            doc_id,
            atom_id,
            text,
            element_type,
            source,
            page=page,
            lang=lang,
            timestamp=timestamp,
            emotion=emotion,
            modality=modality,
            entities=entities,
        )
        self._record_event(
            "add_atom",
            f"Added atom {atom_id} to {doc_id}",
            source=source,
        )

    def add_molecule(  # pragma: no cover
        self,
        doc_id: str,
        molecule_id: str,
        atom_ids: Iterable[str],
        source: Optional[str] = None,
    ) -> None:
        """Insert a molecule node made of existing atoms."""

        self.graph.add_molecule(doc_id, molecule_id, atom_ids, source=source)
        self._record_event(
            "add_molecule",
            f"Added molecule {molecule_id} to {doc_id}",
            source=source,
        )

    def add_hyperedge(  # pragma: no cover
        self,
        edge_id: str,
        node_ids: Iterable[str],
        *,
        relation: str = "member",
        source: str | None = None,
    ) -> None:
        """Insert a hyperedge node connecting ``node_ids``."""

        self.graph.add_hyperedge(
            edge_id,
            node_ids,
            relation=relation,
            source=source,
        )
        self._record_event(
            "add_hyperedge",
            f"Added hyperedge {edge_id}",
            relation=relation,
            source=source,
        )

    def add_simplex(  # pragma: no cover
        self,
        simplex_id: str,
        node_ids: Iterable[str],
        *,
        source: str | None = None,
    ) -> None:
        """Insert a simplex node made of existing vertices."""

        self.graph.add_simplex(simplex_id, node_ids, source=source)
        self._record_event(
            "add_simplex",
            f"Added simplex {simplex_id}",
            source=source,
        )

    def add_entity(  # pragma: no cover
        self, entity_id: str, text: str, source: Optional[str] = None
    ) -> None:
        """Insert an entity node."""

        self.graph.add_entity(entity_id, text, source)
        self._record_event(
            "add_entity",
            f"Added entity {entity_id}",
            source=source,
        )

    def link_entity(  # pragma: no cover
        self,
        node_id: str,
        entity_id: str,
        relation: str = "mentions",
        *,
        provenance: Optional[str] = None,
    ) -> None:
        """Link an entity to another node."""

        self.graph.link_entity(node_id, entity_id, relation, provenance=provenance)
        self._record_event(
            "link_entity",
            f"Linked {node_id} to {entity_id}",
            relation=relation,
            provenance=provenance,
        )

    def search_chunks(self, query: str) -> list[str]:  # pragma: no cover
        """Return chunk IDs whose text matches ``query``."""

        return self.graph.search_chunks(query)

    def chunks_by_emotion(self, emotion: str) -> list[str]:  # pragma: no cover
        """Return chunk IDs with a specific emotion label."""

        ids = self.graph.chunks_by_emotion(emotion)
        self._record_event(
            "chunks_by_emotion", "Chunks filtered by emotion", emotion=emotion
        )
        return ids

    def chunks_by_modality(self, modality: str) -> list[str]:  # pragma: no cover
        """Return chunk IDs with a specific modality label."""

        ids = self.graph.chunks_by_modality(modality)
        self._record_event(
            "chunks_by_modality", "Chunks filtered by modality", modality=modality
        )
        return ids

    def search(
        self, query: str, node_type: str = "chunk"
    ) -> list[str]:  # pragma: no cover
        """Return node IDs of ``node_type`` matching ``query``."""

        return self.graph.search(query, node_type=node_type)

    def search_documents(self, query: str) -> list[str]:  # pragma: no cover
        """Return document IDs with an ID or source matching ``query``."""

        return self.graph.search_documents(query)

    def search_entities(self, query: str) -> list[str]:  # pragma: no cover
        """Return entity IDs matching ``query``."""

        return self.graph.search(query, node_type="entity")

    def search_sections(self, query: str) -> list[str]:  # pragma: no cover
        """Return section IDs whose title or ID matches ``query``."""

        return self.graph.search(query, node_type="section")

    def search_facts(self, query: str) -> list[str]:  # pragma: no cover
        """Return fact IDs whose subject, predicate or object matches the query."""

        return self.graph.search(query, node_type="fact")

    def search_embeddings(  # pragma: no cover
        self,
        query: str,
        k: int = 3,
        fetch_neighbors: bool = True,
        *,
        node_type: str = "chunk",
    ) -> list[str]:
        """Wrapper for :meth:`KnowledgeGraph.search_embeddings`."""

        return self.graph.search_embeddings(
            query,
            k=k,
            fetch_neighbors=fetch_neighbors,
            node_type=node_type,
        )

    def search_hybrid(  # pragma: no cover
        self, query: str, k: int = 5, *, node_type: str = "chunk"
    ) -> list[str]:
        """Wrapper for :meth:`KnowledgeGraph.search_hybrid`."""

        return self.graph.search_hybrid(query, k=k, node_type=node_type)

    def cypher_ann_query(  # pragma: no cover
        self,
        driver: Driver,
        query: str,
        cypher: str,
        *,
        k: int = 5,
        node_type: str = "chunk",
    ) -> List[Dict[str, Any]]:
        """Wrapper for :meth:`KnowledgeGraph.cypher_ann_query`.

        Parameters mirror :meth:`KnowledgeGraph.cypher_ann_query`:

        - ``driver``: Neo4j driver instance
        - ``query``: text passed to the ANN search
        - ``cypher``: query string using ``$ids``
        - ``k``: number of ANN candidates
        - ``node_type``: restrict matches to this type

        The FAISS index retrieves up to ``k`` candidate IDs from ``query``.
        These IDs are bound to the ``ids`` parameter in ``cypher`` so you can
        execute arbitrary graph queries seeded by the ANN search.
        """

        return self.graph.cypher_ann_query(
            driver,
            query,
            cypher,
            k=k,
            node_type=node_type,
        )

    def search_with_links(  # pragma: no cover
        self, query: str, k: int = 5, hops: int = 1, *, fractal_level: int | None = None
    ) -> list[str]:
        """Wrapper for :meth:`KnowledgeGraph.search_with_links`."""

        return self.graph.search_with_links(
            query, k=k, hops=hops, fractal_level=fractal_level
        )

    def search_with_links_data(  # pragma: no cover
        self, query: str, k: int = 5, hops: int = 1, *, fractal_level: int | None = None
    ) -> List[Dict[str, Any]]:
        """Wrapper for :meth:`KnowledgeGraph.search_with_links_data`.

        Returns detailed chunk information, hop depth and traversal path.
        """

        results = self.graph.search_with_links_data(
            query, k=k, hops=hops, fractal_level=fractal_level
        )
        for item in results:
            path = item.get("path")
            if not path:
                continue
            self._update_edge_usage(path)
        return results

    def link_similar_chunks(self, k: int = 3) -> None:  # pragma: no cover
        """Create similarity edges between chunks using embeddings."""

        self.graph.link_similar_chunks(k)
        self._record_event(
            "link_similar_chunks",
            f"Linked similar chunks (k={k})",
            k=k,
        )

    def link_similar_sections(self, k: int = 3) -> None:  # pragma: no cover
        """Create similarity edges between section titles."""

        self.graph.link_similar_sections(k)
        self._record_event(
            "link_similar_sections",
            f"Linked similar sections (k={k})",
            k=k,
        )

    def link_similar_documents(self, k: int = 3) -> None:  # pragma: no cover
        """Create similarity edges between document texts."""

        self.graph.link_similar_documents(k)
        self._record_event(
            "link_similar_documents",
            f"Linked similar documents (k={k})",
            k=k,
        )

    def link_chunks_by_entity(self) -> int:  # pragma: no cover
        """Connect chunks that mention the same entity."""

        added = self.graph.link_chunks_by_entity()
        if added:
            msg = f"Added {added} co-mention links"
            self._record_event(
                "link_chunks_by_entity",
                msg,
                added=added,
            )
        return added

    def link_documents_by_entity(self) -> int:  # pragma: no cover
        """Connect documents that mention the same entity."""

        added = self.graph.link_documents_by_entity()
        if added:
            msg = f"Linked {added} co-mentioned documents"
            self._record_event(
                "link_documents_by_entity",
                msg,
                added=added,
            )
        return added

    def link_sections_by_entity(self) -> int:  # pragma: no cover
        """Connect sections that mention the same entity."""

        added = self.graph.link_sections_by_entity()
        if added:
            msg = f"Linked {added} co-mentioned sections"
            self._record_event(
                "link_sections_by_entity",
                msg,
                added=added,
            )
        return added

    def link_authors_organizations(self) -> int:  # pragma: no cover
        """Create affiliation links between authors and organizations."""

        added = self.graph.link_authors_organizations()
        if added:
            msg = f"Linked {added} authors to organizations"
            self._record_event(
                "link_authors_organizations",
                msg,
                added=added,
            )
        return added

    def get_similar_chunks(
        self, chunk_id: str, k: int = 3
    ) -> list[str]:  # pragma: no cover
        """Return up to ``k`` chunk IDs most similar to ``chunk_id``."""

        return self.graph.get_similar_chunks(chunk_id, k=k)

    def get_similar_chunks_data(  # pragma: no cover
        self, chunk_id: str, k: int = 3
    ) -> List[Dict[str, Any]]:
        """Return up to ``k`` similar chunk infos for ``chunk_id``."""

        return self.graph.get_similar_chunks_data(chunk_id, k=k)

    def get_chunk_neighbors(
        self, k: int = 3
    ) -> Dict[str, List[str]]:  # pragma: no cover
        """Return the ``k`` nearest neighbors for each chunk."""

        return self.graph.get_chunk_neighbors(k=k)

    def get_chunk_neighbors_data(
        self, k: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:  # pragma: no cover
        """Return neighbor information for every chunk."""

        return self.graph.get_chunk_neighbors_data(k=k)

    def get_similar_sections(
        self, section_id: str, k: int = 3
    ) -> list[str]:  # pragma: no cover
        """Return up to ``k`` section IDs most similar to ``section_id``."""

        return self.graph.get_similar_sections(section_id, k=k)

    def get_similar_documents(
        self, doc_id: str, k: int = 3
    ) -> list[str]:  # pragma: no cover
        """Return up to ``k`` document IDs most similar to ``doc_id``."""

        return self.graph.get_similar_documents(doc_id, k=k)

    def hybrid_score(  # pragma: no cover
        self,
        src: str,
        tgt: str,
        *,
        n2v_attr: str = "embedding",
        gw_attr: str = "graphwave_embedding",
        hyper_attr: str = "poincare_embedding",
        gamma: float = 0.5,
        eta: float = 0.25,
    ) -> float:
        r"""Wrapper for :meth:`KnowledgeGraph.hybrid_score`.

        Parameters mirror :meth:`KnowledgeGraph.hybrid_score` and the
        returned value is the multi-view similarity

        .. math::

            S = \gamma \cos(\text{n2v}) + \eta (1 - d_{\mathbb{B}})
            + (1-\gamma-\eta)(1 - \cos(\text{gw}))

        where ``n2v_attr``, ``gw_attr`` and ``hyper_attr`` name the node
        properties holding the Node2Vec, GraphWave and Poincar\xe9
        embeddings. ``gamma`` controls the weight of the Euclidean term
        while ``eta`` controls the hyperbolic part.
        """

        return self.graph.hybrid_score(
            src,
            tgt,
            n2v_attr=n2v_attr,
            gw_attr=gw_attr,
            hyper_attr=hyper_attr,
            gamma=gamma,
            eta=eta,
        )

    def similar_by_hybrid(  # pragma: no cover
        self,
        node_id: str,
        *,
        k: int = 5,
        node_type: str = "chunk",
        n2v_attr: str = "embedding",
        gw_attr: str = "graphwave_embedding",
        hyper_attr: str = "poincare_embedding",
        gamma: float = 0.5,
        eta: float = 0.25,
    ) -> List[Tuple[str, float]]:
        r"""Return nodes ranked by multi-view similarity to ``node_id``.

        The method delegates to :meth:`KnowledgeGraph.similar_by_hybrid` and
        computes the same hybrid score

        .. math::

            S = \gamma \cos(\text{n2v}) + \eta (1 - d_{\mathbb{B}}) +
            (1-\gamma-\eta)(1 - \cos(\text{gw}))

        where the Node2Vec, Poincar\xe9 and GraphWave embeddings are read from
        ``n2v_attr``, ``hyper_attr`` and ``gw_attr``. ``gamma`` and ``eta``
        control the Euclidean and hyperbolic contribution respectively.
        """

        return self.graph.similar_by_hybrid(
            node_id,
            k=k,
            node_type=node_type,
            n2v_attr=n2v_attr,
            gw_attr=gw_attr,
            hyper_attr=hyper_attr,
            gamma=gamma,
            eta=eta,
        )

    def ann_hybrid_search(  # pragma: no cover
        self,
        q_n2v: Sequence[float],
        q_gw: Sequence[float],
        q_hyp: Sequence[float],
        *,
        k: int = 5,
        ann_k: int = 2000,
        node_type: str = "chunk",
        n2v_attr: str = "embedding",
        gw_attr: str = "graphwave_embedding",
        hyper_attr: str = "poincare_embedding",
        gamma: float = 0.5,
        eta: float = 0.25,
    ) -> List[Tuple[str, float]]:
        r"""Wrapper for :meth:`KnowledgeGraph.ann_hybrid_search`.

        ``q_n2v``, ``q_gw`` and ``q_hyp`` are the query embeddings in
        Node2Vec, GraphWave and Poincar\xe9 space. Candidates are fetched
        from the FAISS index on ``n2v_attr`` and scored with

        .. math::

            S = \gamma \cos(\text{n2v}) + \eta (1 - d_{\mathbb{B}}) +
            (1-\gamma-\eta)(1 - \cos(\text{gw}))

        where ``gamma`` and ``eta`` weight the Euclidean and hyperbolic parts.
        ``ann_k`` controls the number of FAISS candidates considered before
        returning the top ``k`` results.
        """

        return self.graph.ann_hybrid_search(
            q_n2v,
            q_gw,
            q_hyp,
            k=k,
            ann_k=ann_k,
            node_type=node_type,
            n2v_attr=n2v_attr,
            gw_attr=gw_attr,
            hyper_attr=hyper_attr,
            gamma=gamma,
            eta=eta,
        )

    def governance_metrics(  # pragma: no cover
        self,
        *,
        n2v_attr: str = "embedding",
        gw_attr: str = "graphwave_embedding",
        hyper_attr: str = "poincare_embedding",
    ) -> Dict[str, float]:
        """Wrapper for :meth:`KnowledgeGraph.governance_metrics`."""

        metrics = self.graph.governance_metrics(
            n2v_attr=n2v_attr,
            gw_attr=gw_attr,
            hyp_attr=hyper_attr,
        )
        self._record_event("governance_metrics", str(metrics))
        return metrics

    def mitigate_bias_wasserstein(  # pragma: no cover
        self,
        groups: Dict[str, str],
        *,
        attr: str = "embedding",
    ) -> Dict[str, np.ndarray]:
        """Wrapper for :meth:`KnowledgeGraph.mitigate_bias_wasserstein`."""

        res = self.graph.mitigate_bias_wasserstein(groups, attr=attr)
        self._record_event("mitigate_bias_wasserstein", f"groups={len(groups)}")
        return res

    def average_hyperbolic_radius(
        self, *, attr: str = "poincare_embedding"
    ) -> float:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.average_hyperbolic_radius`."""

        radius = self.graph.average_hyperbolic_radius(attr=attr)
        self._record_event("average_hyperbolic_radius", f"radius={radius:.4f}")
        return radius

    def apply_k_out_privacy(
        self, ids: List[str], k: int = 2
    ) -> List[str]:  # pragma: no cover
        """Return ``ids`` after applying k-out randomized response."""

        from ..analysis.privacy import k_out_randomized_response

        priv = k_out_randomized_response(ids, k=k)
        self._record_event("k_out_privacy", f"k={k}")
        return priv

    # ------------------------------------------------------------------
    # Differential privacy budget helpers
    # ------------------------------------------------------------------

    def add_privacy_budget(self, user: str, epsilon: float) -> None:  # pragma: no cover
        """Register ``user`` with a daily budget ``epsilon``."""

        self.dp_budgets.add_user(user, epsilon)
        self._record_event("add_privacy_budget", f"user={user} eps={epsilon}")

    def consume_privacy_budget(
        self, user: str, amount: float
    ) -> bool:  # pragma: no cover
        """Consume ``amount`` from ``user``'s privacy budget."""

        ok = self.dp_budgets.consume(user, amount)
        self._record_event("consume_privacy_budget", f"user={user} amount={amount}")
        return ok

    def get_chunk_context(  # pragma: no cover
        self, chunk_id: str, before: int = 1, after: int = 1
    ) -> list[str]:
        """Return chunk IDs surrounding ``chunk_id`` including itself."""

        return self.graph.get_chunk_context(chunk_id, before=before, after=after)

    def deduplicate_chunks(self, similarity: float = 1.0) -> int:  # pragma: no cover
        """Remove duplicate chunk nodes from the graph.

        Parameters
        ----------
        similarity:
            Similarity threshold between 0 and 1 for detecting duplicates. A
            value of 1.0 removes only exact matches.
        """

        removed = self.graph.deduplicate_chunks(similarity)
        if removed:
            msg = f"Removed {removed} duplicate chunks"
            self._record_event("deduplicate_chunks", msg)
        return removed

    def clean_chunks(self) -> int:  # pragma: no cover
        """Normalize chunk text to remove markup and extra whitespace."""

        cleaned = self.graph.clean_chunk_texts()
        if cleaned:
            msg = f"Cleaned {cleaned} chunks"
            self._record_event("clean_chunks", msg)
        return cleaned

    @persist_after
    def cleanup_graph(  # pragma: no cover
        self,
        *,
        resolve_threshold: float = 0.8,
        resolve_aliases: dict[str, list[str]] | None = None,
        dedup_similarity: float = 1.0,
        normalize_dates: bool = True,
        mark_conflicts: bool = False,
        validate: bool = False,
    ) -> tuple[int, int]:
        """Run standard deduplication and cleaning steps on the graph.

        This removes duplicate chunks, normalizes their text and resolves
        entities so later pipeline stages work with a clean knowledge graph.

        Parameters
        ----------
        resolve_threshold:
            Minimum similarity score when merging entities.
        resolve_aliases:
            Optional mapping of canonical labels to lists of aliases used during
            entity resolution.
        dedup_similarity:
            Threshold used when removing duplicate chunks. Values below 1.0
            allow near-duplicate detection.

        Returns
        -------
        tuple[int, int]
            Number of chunks removed and cleaned respectively.
        """

        removed = self.deduplicate_chunks(similarity=dedup_similarity)
        cleaned = self.clean_chunks()
        if normalize_dates:
            self.normalize_dates()
        self.resolve_entities(threshold=resolve_threshold, aliases=resolve_aliases)
        if mark_conflicts:
            self.mark_conflicting_facts()
        if validate:
            self.validate_coherence()
        self._record_event(
            "cleanup_graph",
            f"Graph cleaned (removed={removed}, cleaned={cleaned})",
            resolve_threshold=resolve_threshold,
            dedup_similarity=dedup_similarity,
            removed=removed,
            cleaned=cleaned,
        )
        self.stage = max(self.stage, DatasetStage.CURATED)
        return removed, cleaned

    def normalize_dates(self) -> int:  # pragma: no cover
        """Standardize date attributes on nodes to ISO format."""

        changed = self.graph.normalize_date_fields()
        if changed:
            msg = f"Normalized {changed} date fields"
            self._record_event("normalize_dates", msg)
        return changed

    def prune_sources(self, sources: List[str]) -> int:  # pragma: no cover
        """Remove nodes and edges associated with ``sources`` from the graph."""

        removed = self.graph.prune_sources(sources)
        if removed:
            joined = ", ".join(sources)
            msg = f"Pruned {removed} nodes from {joined}"
            self._record_event("prune_sources", msg, sources=sources)
        return removed

    def resolve_entities(  # pragma: no cover
        self,
        threshold: float = 0.8,
        aliases: dict[str, list[str]] | None = None,
    ) -> int:
        """Merge entity nodes that likely refer to the same real world entity."""

        merged = self.graph.resolve_entities(threshold=threshold, aliases=aliases)
        if merged:
            msg = f"Merged {merged} entities"
            self._record_event(
                "resolve_entities",
                msg,
                threshold=threshold,
                aliases=list(aliases.keys()) if aliases else None,
            )
        return merged

    def predict_links(  # pragma: no cover
        self, threshold: float = 0.8, *, use_graph_embeddings: bool = False
    ) -> None:
        """Infer missing relations between entities based on similarity."""

        self.graph.predict_links(
            threshold=threshold, use_graph_embeddings=use_graph_embeddings
        )
        self._record_event(
            "predict_links",
            "Predicted entity links",
            threshold=threshold,
            use_graph_embeddings=use_graph_embeddings,
        )

    def enrich_entity(self, entity_id: str) -> None:  # pragma: no cover
        """Enrich an entity node using external sources like Wikidata."""

        self.graph.enrich_entity_wikidata(entity_id)
        self._record_event("enrich_entity", f"Entity {entity_id} enriched")

    def enrich_entity_dbpedia(self, entity_id: str) -> None:  # pragma: no cover
        """Enrich an entity node using DBpedia."""

        self.graph.enrich_entity_dbpedia(entity_id)
        self._record_event(
            "enrich_entity_dbpedia",
            f"Entity {entity_id} enriched from DBpedia",
        )

    def consolidate_schema(self) -> None:  # pragma: no cover
        """Normalize labels in the underlying knowledge graph."""

        self.graph.consolidate_schema()
        self._record_event("consolidate_schema", "Schema consolidated")

    def detect_communities(self, n_clusters: int = 3) -> None:  # pragma: no cover
        """Cluster chunks into communities."""

        self.graph.cluster_chunks(n_clusters=n_clusters)
        self._record_event(
            "detect_communities",
            "Communities detected",
            n_clusters=n_clusters,
        )

    def detect_entity_groups(self, n_clusters: int = 3) -> None:  # pragma: no cover
        """Cluster entity nodes into groups."""

        self.graph.cluster_entities(n_clusters=n_clusters)
        self._record_event(
            "detect_entity_groups",
            "Entity groups detected",
            n_clusters=n_clusters,
        )

    def summarize_entity_groups(self) -> None:  # pragma: no cover
        self.graph.summarize_entity_groups()
        self._record_event("summarize_entity_groups", "Entity groups summarized")

    def summarize_communities(self) -> None:  # pragma: no cover
        self.graph.summarize_communities()
        self._record_event("summarize_communities", "Communities summarized")

    def score_trust(self) -> None:  # pragma: no cover
        self.graph.score_trust()
        self._record_event("score_trust", "Computed trust scores")

    def compute_centrality(  # pragma: no cover
        self, node_type: str = "entity", metric: str = "degree"
    ) -> None:
        """Compute centrality metrics for graph nodes."""
        self.graph.compute_centrality(node_type=node_type, metric=metric)
        self._record_event(
            "compute_centrality",
            f"Centrality ({metric}) computed for {node_type} nodes",
            node_type=node_type,
            metric=metric,
        )

    def compute_graph_embeddings(  # pragma: no cover
        self,
        dimensions: int | None = None,
        walk_length: int = 10,
        num_walks: int = 50,
        seed: int = 0,
        workers: int = 1,
        *,
        p: float | None = None,
        q: float | None = None,
    ) -> None:
        """Generate Node2Vec embeddings for all nodes."""
        cfg = load_config()
        embed_cfg = cfg.get("embeddings", {}).get("node2vec", {})
        dim_val = dimensions or int(embed_cfg.get("dimension", 64))
        p_val = p if p is not None else float(embed_cfg.get("p", 1.0))
        q_val = q if q is not None else float(embed_cfg.get("q", 1.0))

        self.graph.compute_node2vec_embeddings(
            dimensions=dim_val,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            seed=seed,
            p=p_val,
            q=q_val,
        )
        self._record_event(
            "compute_graph_embeddings",
            "Graph embeddings computed",
            dimensions=dim_val,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            seed=seed,
        )

    def compute_node2vec_gds(  # pragma: no cover
        self,
        driver: "Driver",
        *,
        dimensions: int | None = None,
        walk_length: int = 40,
        walks_per_node: int = 10,
        p: float | None = None,
        q: float | None = None,
        dataset: str | None = None,
        write_property: str = "embedding",
    ) -> None:
        """Run Neo4j GDS Node2Vec and store embeddings on the graph."""

        cfg = load_config()
        embed_cfg = cfg.get("embeddings", {}).get("node2vec", {})
        dim_val = dimensions or int(embed_cfg.get("dimension", 128))
        p_val = p if p is not None else float(embed_cfg.get("p", 1.0))
        q_val = q if q is not None else float(embed_cfg.get("q", 1.0))

        self.graph.compute_node2vec_gds(
            driver,
            dimensions=dim_val,
            walk_length=walk_length,
            walks_per_node=walks_per_node,
            p=p_val,
            q=q_val,
            dataset=dataset,
            write_property=write_property,
        )
        self._record_event(
            "compute_node2vec_gds",
            "Neo4j Node2Vec embeddings computed",
            dimensions=dim_val,
            walk_length=walk_length,
            walks_per_node=walks_per_node,
            p=p_val,
            q=q_val,
        )

    def compute_graphwave_embeddings(  # pragma: no cover
        self,
        scales: Iterable[float],
        num_points: int = 10,
        *,
        chebyshev_order: int | None = None,
        gpu: bool = False,
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.compute_graphwave_embeddings`."""

        self.graph.compute_graphwave_embeddings(
            scales=scales,
            num_points=num_points,
            chebyshev_order=chebyshev_order,
            gpu=gpu,
        )
        self._record_event(
            "compute_graphwave_embeddings",
            "GraphWave embeddings computed",
            scales=list(scales),
            num_points=num_points,
            chebyshev_order=chebyshev_order,
            gpu=gpu,
        )

    def graphwave_entropy(self) -> float:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.graphwave_entropy`."""

        val = self.graph.graphwave_entropy()
        self._record_event("graphwave_entropy", "GraphWave entropy computed")
        return val

    def ensure_graphwave_entropy(  # pragma: no cover
        self,
        threshold: float,
        *,
        scales: Iterable[float] = (0.5, 1.0),
        num_points: int = 10,
    ) -> float:
        """Wrapper for :meth:`KnowledgeGraph.ensure_graphwave_entropy`."""

        val = self.graph.ensure_graphwave_entropy(
            threshold,
            scales=scales,
            num_points=num_points,
        )
        self._record_event(
            "ensure_graphwave_entropy",
            "GraphWave entropy enforced",
            threshold=threshold,
        )
        return val

    def embedding_entropy(
        self, node_attr: str = "embedding"
    ) -> float:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.embedding_entropy`."""

        val = self.graph.embedding_entropy(node_attr=node_attr)
        self._record_event(
            "embedding_entropy",
            "Embedding entropy computed",
            attribute=node_attr,
        )
        return val

    def embedding_box_counting_dimension(  # pragma: no cover
        self, node_attr: str, radii: Iterable[float]
    ) -> tuple[float, list[tuple[float, int]]]:
        """Wrapper for :meth:`KnowledgeGraph.embedding_box_counting_dimension`."""

        dim, counts = self.graph.embedding_box_counting_dimension(node_attr, radii)
        self._record_event(
            "embedding_box_counting_dimension",
            "Embedding fractal dimension computed",
            attribute=node_attr,
            radii=list(radii),
            dimension=dim,
        )
        return dim, counts

    def compute_poincare_embeddings(  # pragma: no cover
        self,
        dim: int = 2,
        negative: int = 5,
        epochs: int = 50,
        learning_rate: float = 0.1,
        burn_in: int = 10,
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.compute_poincare_embeddings`."""

        self.graph.compute_poincare_embeddings(
            dim=dim,
            negative=negative,
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
        )
        self._record_event(
            "compute_poincare_embeddings",
            "Poincaré embeddings computed",
            dim=dim,
            negative=negative,
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
        )

    def compute_hyperbolic_hypergraph_embeddings(  # pragma: no cover
        self,
        dim: int = 2,
        negative: int = 5,
        epochs: int = 50,
        learning_rate: float = 0.1,
        burn_in: int = 10,
    ) -> Dict[str, list[float]]:
        """Wrapper for :meth:`KnowledgeGraph.compute_hyperbolic_hypergraph_embeddings`."""

        result = self.graph.compute_hyperbolic_hypergraph_embeddings(
            dim=dim,
            negative=negative,
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
        )
        self._record_event(
            "compute_hyperbolic_hypergraph_embeddings",
            "Hyperbolic hypergraph embeddings computed",
            dim=dim,
            negative=negative,
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
        )
        return result

    def compute_learned_hyperbolic_embeddings(  # pragma: no cover
        self,
        *,
        feature_attr: str = "embedding",
        write_property: str = "hyperbolic_learned",
        dim: int = 2,
        lr: float = 0.01,
        epochs: int = 50,
        geo_weight: float = 1.0,
        frac_weight: float = 0.1,
        target_dim: float | None = None,
    ) -> Dict[str, list[float]]:
        """Wrapper for :meth:`KnowledgeGraph.compute_learned_hyperbolic_embeddings`."""

        result = self.graph.compute_learned_hyperbolic_embeddings(
            feature_attr=feature_attr,
            write_property=write_property,
            dim=dim,
            lr=lr,
            epochs=epochs,
            geo_weight=geo_weight,
            frac_weight=frac_weight,
            target_dim=target_dim,
        )
        self._record_event(
            "compute_learned_hyperbolic_embeddings",
            "Hyperbolic embeddings learned with fractal regularization",
            dim=dim,
            lr=lr,
            epochs=epochs,
        )
        return result

    def haa_link_score(
        self, driver: "Driver", a_id: str, b_id: str
    ) -> float | None:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.haa_link_score`."""

        result = self.graph.haa_link_score(driver, a_id, b_id)
        self._record_event("haa_link_score", "Hyper-AA pair score fetched")
        return result

    def compute_hyper_sagnn_embeddings(  # pragma: no cover
        self,
        *,
        node_attr: str = "embedding",
        edge_attr: str = "hyper_sagnn_embedding",
        embed_dim: int | None = None,
        seed: int | None = None,
    ) -> Dict[str, list[float]]:
        """Wrapper for :meth:`KnowledgeGraph.compute_hyper_sagnn_embeddings`."""

        result = self.graph.compute_hyper_sagnn_embeddings(
            node_attr=node_attr,
            edge_attr=edge_attr,
            embed_dim=embed_dim,
            seed=seed,
        )
        self._record_event(
            "compute_hyper_sagnn_embeddings",
            "Hyper-SAGNN embeddings computed",
            embed_dim=embed_dim,
        )
        return result

    def compute_hyper_sagnn_head_drop_embeddings(  # pragma: no cover
        self,
        *,
        node_attr: str = "embedding",
        edge_attr: str = "hyper_sagnn_hd_embedding",
        num_heads: int = 4,
        threshold: float = 0.1,
        seed: int | None = None,
    ) -> Dict[str, list[float]]:
        """Wrapper for :meth:`KnowledgeGraph.compute_hyper_sagnn_head_drop_embeddings`."""

        result = self.graph.compute_hyper_sagnn_head_drop_embeddings(
            node_attr=node_attr,
            edge_attr=edge_attr,
            num_heads=num_heads,
            threshold=threshold,
            seed=seed,
        )
        self._record_event(
            "compute_hyper_sagnn_head_drop_embeddings",
            "Hyper-SAGNN embeddings computed with HEAD-Drop",
            num_heads=num_heads,
            threshold=threshold,
        )
        return result

    def compute_hgcn_sagnn_embeddings(  # pragma: no cover
        self,
        *,
        node_attr: str = "embedding",
        edge_attr: str = "hgcn_sagnn_embedding",
        K: int = 2,
        embed_dim: int | None = None,
        seed: int | None = None,
    ) -> Dict[str, list[float]]:
        """Wrapper for :meth:`KnowledgeGraph.compute_hgcn_sagnn_embeddings`."""

        result = self.graph.compute_hgcn_sagnn_embeddings(
            node_attr=node_attr,
            edge_attr=edge_attr,
            K=K,
            embed_dim=embed_dim,
            seed=seed,
        )
        self._record_event(
            "compute_hgcn_sagnn_embeddings",
            "Hyper-SAGNN embeddings computed with HGCN spectral features",
            K=K,
            embed_dim=embed_dim,
        )
        return result

    def hyper_adamic_adar_scores(
        self,
    ) -> Dict[tuple[str, str], float]:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.hyper_adamic_adar_scores`."""

        result = self.graph.hyper_adamic_adar_scores()
        self._record_event(
            "hyper_adamic_adar_scores",
            "Hyper-Adamic–Adar scores computed",
        )
        return result

    def edge_attention_scores(  # pragma: no cover
        self,
        *,
        node_attr: str = "embedding",
        seed: int | None = None,
    ) -> Dict[str, float]:
        """Wrapper for :meth:`KnowledgeGraph.edge_attention_scores`."""

        result = self.graph.edge_attention_scores(node_attr=node_attr, seed=seed)
        self._record_event(
            "edge_attention_scores",
            "Edge attention scores computed",
        )
        return result

    def explain_node(  # pragma: no cover
        self, node_id: str, *, hops: int = 3, node_attr: str = "embedding"
    ) -> Dict[str, Any]:
        """Wrapper for :meth:`KnowledgeGraph.explain_node`."""

        result = self.graph.explain_node(node_id, hops=hops, node_attr=node_attr)
        self._record_event(
            "explain_node",
            f"Explained node {node_id}",
            node=node_id,
            hops=hops,
        )
        return result

    def recall_at_k(  # pragma: no cover
        self,
        queries: Sequence[str],
        ground_truth: Dict[str, Sequence[str]],
        *,
        k: int = 10,
        gamma: float = 0.5,
        eta: float = 0.25,
    ) -> float:
        """Wrapper for :meth:`KnowledgeGraph.recall_at_k`."""

        score = self.graph.recall_at_k(
            queries,
            ground_truth,
            k=k,
            gamma=gamma,
            eta=eta,
        )
        if k == 10:
            self.graph.graph["recall10"] = score
        self._record_event(
            "recall_at_k",
            "Recall@k computed",
            k=k,
        )
        return score

    def compute_graphsage_embeddings(  # pragma: no cover
        self,
        *,
        dimensions: int = 64,
        num_layers: int = 2,
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.compute_graphsage_embeddings`."""

        self.graph.compute_graphsage_embeddings(
            dimensions=dimensions, num_layers=num_layers
        )
        self._record_event(
            "compute_graphsage_embeddings",
            "GraphSAGE embeddings computed",
            dimensions=dimensions,
            num_layers=num_layers,
        )

    def compute_transe_embeddings(  # pragma: no cover
        self,
        *,
        dimensions: int = 64,
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.compute_transe_embeddings`."""

        self.graph.compute_transe_embeddings(dimensions=dimensions)
        self._record_event(
            "compute_transe_embeddings",
            "TransE relation embeddings computed",
            dimensions=dimensions,
        )

    def build_faiss_index(  # pragma: no cover
        self, node_attr: str = "embedding", *, method: str | None = None
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.build_faiss_index`."""

        cfg = load_config()
        method_val = method or cfg.get("ann", {}).get("backend", "flat")
        self.graph.build_faiss_index(node_attr=node_attr, method=method_val)
        self._record_event(
            "build_faiss_index",
            "FAISS index built",
            node_attr=node_attr,
            method=method_val,
        )

    def search_faiss(  # pragma: no cover
        self,
        vector: Iterable[float],
        k: int = 5,
        *,
        adaptive: bool = False,
        latency_threshold: float = 0.1,
    ) -> list[str]:
        """Wrapper for :meth:`KnowledgeGraph.search_faiss`."""

        return self.graph.search_faiss(
            vector,
            k=k,
            adaptive=adaptive,
            latency_threshold=latency_threshold,
        )

    def compute_distmult_embeddings(  # pragma: no cover
        self,
        *,
        dimensions: int = 64,
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.compute_distmult_embeddings`."""

        self.graph.compute_distmult_embeddings(dimensions=dimensions)
        self._record_event(
            "compute_distmult_embeddings",
            "DistMult relation embeddings computed",
            dimensions=dimensions,
        )

    def compute_multigeometric_embeddings(  # pragma: no cover
        self,
        *,
        node2vec_dim: int = 64,
        graphwave_scales: Iterable[float] | None = None,
        graphwave_points: int = 10,
        poincare_dim: int = 2,
        negative: int = 5,
        epochs: int = 50,
        learning_rate: float = 0.1,
        burn_in: int = 10,
        node2vec_p: float = 1.0,
        node2vec_q: float = 1.0,
    ) -> None:
        """Compute Node2Vec, GraphWave, Poincar\u00e9 and GraphSAGE embeddings."""

        self.compute_graph_embeddings(
            dimensions=node2vec_dim,
            walk_length=10,
            num_walks=50,
            workers=1,
            seed=0,
            p=node2vec_p,
            q=node2vec_q,
        )
        self.compute_graphwave_embeddings(
            scales=graphwave_scales or [0.5, 1.0],
            num_points=graphwave_points,
        )
        self.compute_poincare_embeddings(
            dim=poincare_dim,
            negative=min(negative, max(1, len(self.graph.graph.nodes) - 2)),
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
        )
        self.compute_graphsage_embeddings(dimensions=node2vec_dim, num_layers=2)
        self._record_event(
            "compute_multigeometric_embeddings",
            "Multi-geometry embeddings computed",
            node2vec_dim=node2vec_dim,
            graphwave_scales=list(graphwave_scales or [0.5, 1.0]),
            graphwave_points=graphwave_points,
            poincare_dim=poincare_dim,
            negative=negative,
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
            node2vec_p=node2vec_p,
            node2vec_q=node2vec_q,
        )

    def compute_product_manifold_embeddings(  # pragma: no cover
        self,
        *,
        hyperbolic_attr: str = "poincare_embedding",
        euclidean_attr: str = "embedding",
        write_property: str = "product_embedding",
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.compute_product_manifold_embeddings`."""

        self.graph.compute_product_manifold_embeddings(
            hyperbolic_attr=hyperbolic_attr,
            euclidean_attr=euclidean_attr,
            write_property=write_property,
        )
        self._record_event(
            "compute_product_manifold_embeddings",
            "Product-manifold embeddings computed",
            hyperbolic_attr=hyperbolic_attr,
            euclidean_attr=euclidean_attr,
        )

    def train_product_manifold_embeddings(  # pragma: no cover
        self,
        contexts: Iterable[tuple[str, str]],
        *,
        hyperbolic_attr: str = "poincare_embedding",
        euclidean_attr: str = "embedding",
        alpha: float = 0.5,
        lr: float = 0.01,
        epochs: int = 1,
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.train_product_manifold_embeddings`."""

        self.graph.train_product_manifold_embeddings(
            contexts,
            hyperbolic_attr=hyperbolic_attr,
            euclidean_attr=euclidean_attr,
            alpha=alpha,
            lr=lr,
            epochs=epochs,
        )
        self._record_event(
            "train_product_manifold_embeddings",
            "Product-manifold embeddings trained",
            alpha=alpha,
            epochs=epochs,
        )

    def compute_aligned_cca_embeddings(  # pragma: no cover
        self,
        *,
        n_components: int = 32,
        n2v_attr: str = "embedding",
        gw_attr: str = "graphwave_embedding",
        write_property: str = "acca_embedding",
        path: str = os.path.join(CACHE_ROOT, "cca.pkl"),
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.compute_aligned_cca_embeddings`."""

        self.graph.compute_aligned_cca_embeddings(
            n_components=n_components,
            n2v_attr=n2v_attr,
            gw_attr=gw_attr,
            write_property=write_property,
            path=path,
        )
        self._record_event(
            "compute_aligned_cca_embeddings",
            "Aligned CCA embeddings computed",
            n_components=n_components,
        )

    def multiview_contrastive_loss(  # pragma: no cover
        self,
        *,
        n2v_attr: str = "embedding",
        gw_attr: str = "graphwave_embedding",
        hyper_attr: str = "poincare_embedding",
        tau: float = 0.1,
    ) -> float:
        """Wrapper for :meth:`KnowledgeGraph.multiview_contrastive_loss`."""

        loss = self.graph.multiview_contrastive_loss(
            n2v_attr=n2v_attr,
            gw_attr=gw_attr,
            hyper_attr=hyper_attr,
            tau=tau,
        )
        self._record_event(
            "multiview_contrastive_loss",
            "Computed multi-view contrastive loss",
            tau=tau,
        )
        return loss

    def compute_meta_embeddings(  # pragma: no cover
        self,
        *,
        n2v_attr: str = "embedding",
        gw_attr: str = "graphwave_embedding",
        hyper_attr: str = "poincare_embedding",
        bottleneck: int = 64,
        write_property: str = "meta_embedding",
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.compute_meta_embeddings`."""

        self.graph.compute_meta_embeddings(
            n2v_attr=n2v_attr,
            gw_attr=gw_attr,
            hyper_attr=hyper_attr,
            bottleneck=bottleneck,
            write_property=write_property,
        )
        self._record_event(
            "compute_meta_embeddings",
            "Meta-embeddings computed",
            bottleneck=bottleneck,
        )

    def prune_embeddings(
        self, *, tol: float = 1e-3
    ) -> Dict[str, int]:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.prune_embeddings`."""

        mapping = self.graph.prune_embeddings(tol=tol)
        self._record_event(
            "prune_embeddings",
            "Embeddings pruned via fractal Net",
            clusters=len(set(mapping.values())),
        )
        return mapping

    def fractalnet_compress(  # pragma: no cover
        self, node_attr: str = "embedding"
    ) -> Dict[int, np.ndarray]:
        """Wrapper for :meth:`KnowledgeGraph.fractalnet_compress`."""

        comp = self.graph.fractalnet_compress(node_attr=node_attr)
        self._record_event(
            "fractalnet_compress",
            "Embeddings compressed by fractal level",
            node_attr=node_attr,
            levels=len(comp),
        )
        return comp

    def prune_fractalnet_weights(  # pragma: no cover
        self, weights: np.ndarray, *, ratio: float = 0.5
    ) -> np.ndarray:
        """Prune model weights via :func:`prune_fractalnet`.

        Parameters
        ----------
        weights:
            Weight array to prune.
        ratio:
            Fraction of weights to keep.

        Returns
        -------
        numpy.ndarray
            Pruned weight array.
        """

        from ..analysis.compression import prune_fractalnet as _pf

        pruned = _pf(weights, ratio=ratio)
        self._record_event(
            "prune_fractalnet_weights",
            "Weights pruned via FractalNet rule",
            ratio=ratio,
            kept=int(np.count_nonzero(pruned)),
        )
        return pruned

    def fractal_dimension(  # pragma: no cover
        self, radii: Iterable[int]
    ) -> tuple[float, list[tuple[int, int]]]:
        """Wrapper for :meth:`KnowledgeGraph.box_counting_dimension`."""

        dim, counts = self.graph.box_counting_dimension(radii)
        self._record_event(
            "fractal_dimension",
            "Fractal dimension computed",
            radii=list(radii),
        )
        return dim, counts

    def colour_box_dimension(  # pragma: no cover
        self, radii: Iterable[int]
    ) -> tuple[float, list[tuple[int, int]]]:
        """Wrapper for :meth:`KnowledgeGraph.colour_box_dimension`."""

        dim, counts = self.graph.colour_box_dimension(radii)
        self._record_event(
            "colour_box_dimension",
            "COLOUR-box fractal dimension computed",
            radii=list(radii),
        )
        return dim, counts

    def compute_fractal_features(  # pragma: no cover
        self, radii: Iterable[int], *, max_dim: int = 1
    ) -> Dict[str, Any]:
        """Compute fractal metrics and record the event."""

        features = self.graph.compute_fractal_features(radii, max_dim=max_dim)
        self._record_event(
            "compute_fractal_features",
            "Fractal features computed",
            radii=list(radii),
            dimension=features["dimension"],
            radius=features["radius"],
        )
        return features

    def fractal_information_metrics(  # pragma: no cover
        self, radii: Iterable[int], *, max_dim: int = 1
    ) -> Dict[str, Any]:
        """Wrapper for :meth:`KnowledgeGraph.fractal_information_metrics`."""

        metrics = self.graph.fractal_information_metrics(radii, max_dim=max_dim)
        self._record_event(
            "fractal_information_metrics",
            "Fractal information metrics computed",
            radii=list(radii),
        )
        return metrics

    def dimension_distortion(self, radii: Iterable[int]) -> float:  # pragma: no cover
        """Return |D_graph - D_embedding| for stored Poincaré embeddings.

        This method forwards ``radii`` to
        :meth:`KnowledgeGraph.dimension_distortion`, which estimates the fractal
        dimensions of the graph and its Poincaré embedding using box counting.
        The returned value is the absolute difference between these dimensions
        and indicates how faithfully the embedding preserves the graph
        geometry.
        """

        dist = self.graph.dimension_distortion(radii)
        self._record_event(
            "dimension_distortion",
            "Fractal dimension distortion computed",
            radii=list(radii),
            distortion=dist,
        )
        return dist

    def detect_automorphisms(
        self, max_count: int = 10
    ) -> List[Dict[str, str]]:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.detect_automorphisms`."""

        autos = self.graph.detect_automorphisms(max_count=max_count)
        self._record_event(
            "detect_automorphisms",
            "Automorphisms detected",
            count=len(autos),
        )
        return autos

    def automorphism_group_order(self, max_count: int = 100) -> int:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.automorphism_group_order`."""

        order = self.graph.automorphism_group_order(max_count=max_count)
        self._record_event(
            "automorphism_group_order",
            "Automorphism group order estimated",
            count=order,
        )
        return order

    def quotient_by_symmetry(  # pragma: no cover
        self, *, max_count: int = 10
    ) -> tuple[nx.Graph, Dict[str, int]]:
        """Wrapper for :meth:`KnowledgeGraph.quotient_by_symmetry`."""

        q, mapping = self.graph.quotient_by_symmetry(max_count=max_count)
        self._record_event(
            "quotient_by_symmetry",
            "Graph quotient computed",
            classes=len(set(mapping.values())),
        )
        return q, mapping

    def mapper_nerve(
        self, radius: int
    ) -> tuple[nx.Graph, list[set[str]]]:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.mapper_nerve`."""

        nerve, cover = self.graph.mapper_nerve(radius)
        self._record_event(
            "mapper_nerve",
            "Mapper nerve computed",
            radius=radius,
            clusters=len(cover),
        )
        return nerve, cover

    def clear_mapper_cache(self) -> None:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.clear_mapper_cache`."""

        self.graph.clear_mapper_cache()
        self._record_event("clear_mapper_cache", "Mapper cache cleared")

    def rollback_gremlin_diff(
        self, output: str = "rollback.diff"
    ) -> str:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.rollback_gremlin_diff`."""

        path = self.graph.rollback_gremlin_diff(output)
        self._record_event("rollback_gremlin_diff", "Rollback diff written", path=path)
        return path

    def sheaf_checker_sla(self, failures: Iterable[float]) -> float:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.sheaf_checker_sla`."""

        mttr = self.graph.sheaf_checker_sla(failures)
        self._record_event("sheaf_checker_sla", "SLA evaluated", mttr=mttr)
        return mttr

    def inverse_mapper(  # pragma: no cover
        self, nerve: nx.Graph, cover: Iterable[Iterable[str]]
    ) -> nx.Graph:
        """Wrapper for :meth:`KnowledgeGraph.inverse_mapper`."""

        g = self.graph.inverse_mapper(nerve, cover)
        self._record_event("inverse_mapper", "Graph reconstructed from nerve")
        return g

    # --------------------------------------------------------------
    # Graph generation wrappers
    # --------------------------------------------------------------

    def generate_graph_rnn_like(
        self, num_nodes: int, num_edges: int
    ) -> nx.Graph:  # pragma: no cover
        """Return a random graph mimicking GraphRNN output."""

        g = self.graph.generate_graph_rnn_like(num_nodes, num_edges)
        self._record_event(
            "generate_graph_rnn_like",
            "Random GraphRNN-like graph generated",
            nodes=num_nodes,
            edges=num_edges,
        )
        return g

    def filter_semantic_cycles(  # pragma: no cover
        self,
        *,
        attr: str = "text",
        stopwords: Iterable[str] | None = None,
        max_len: int = 4,
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.filter_semantic_cycles`."""

        self.graph.filter_semantic_cycles(
            attr=attr, stopwords=stopwords, max_len=max_len
        )
        self._record_event(
            "filter_semantic_cycles",
            "Trivial cycles removed",
            max_len=max_len,
        )

    def generate_graph_rnn(  # pragma: no cover
        self, num_nodes: int, num_edges: int, *, p: float = 0.5, directed: bool = False
    ) -> nx.Graph:
        """Return a simple sequential GraphRNN-style graph."""

        g = self.graph.generate_graph_rnn(num_nodes, num_edges, p=p, directed=directed)
        self._record_event(
            "generate_graph_rnn",
            "GraphRNN sequential graph generated",
            nodes=num_nodes,
            edges=num_edges,
            p=p,
            directed=directed,
        )
        return g

    def generate_graph_rnn_stateful(  # pragma: no cover
        self,
        num_nodes: int,
        num_edges: int,
        *,
        hidden_dim: int = 8,
        seed: int | None = None,
    ) -> nx.DiGraph:
        """Return a directed graph from a tiny stateful RNN generator."""

        g = self.graph.generate_graph_rnn_stateful(
            num_nodes, num_edges, hidden_dim=hidden_dim, seed=seed
        )
        self._record_event(
            "generate_graph_rnn_stateful",
            "Stateful GraphRNN graph generated",
            nodes=num_nodes,
            edges=num_edges,
            hidden_dim=hidden_dim,
        )
        return g

    def generate_graph_rnn_sequential(  # pragma: no cover
        self,
        num_nodes: int,
        num_edges: int,
        *,
        hidden_dim: int = 8,
        seed: int | None = None,
        directed: bool = True,
    ) -> nx.Graph:
        """Return a graph using a sequential RNN-style generator."""

        g = self.graph.generate_graph_rnn_sequential(
            num_nodes,
            num_edges,
            hidden_dim=hidden_dim,
            seed=seed,
            directed=directed,
        )
        self._record_event(
            "generate_graph_rnn_sequential",
            "Sequential GraphRNN graph generated",
            nodes=num_nodes,
            edges=num_edges,
            directed=directed,
        )
        return g

    def optimize_topology_constrained(  # pragma: no cover
        self,
        target: nx.Graph,
        radii: Iterable[int],
        *,
        dimension: int = 1,
        epsilon: float = 0.0,
        delta: float = 0.1,
        max_iter: int = 100,
        seed: int | None = None,
        use_generator: bool = False,
        use_netgan: bool = False,
    ) -> Tuple[float, float]:
        """Wrapper for :meth:`KnowledgeGraph.optimize_topology_constrained`."""

        dist, diff = self.graph.optimize_topology_constrained(
            target,
            radii,
            dimension=dimension,
            epsilon=epsilon,
            delta=delta,
            max_iter=max_iter,
            seed=seed,
            use_generator=use_generator,
            use_netgan=use_netgan,
        )
        self._record_event(
            "optimize_topology_constrained",
            "Topology adjusted with fractal constraint",
            dimension=dimension,
            epsilon=epsilon,
            delta=delta,
            max_iter=max_iter,
            use_generator=use_generator,
            use_netgan=use_netgan,
            dimension_diff=diff,
        )
        return dist, diff

    def validate_topology(  # pragma: no cover
        self,
        target: nx.Graph,
        radii: Iterable[int],
        *,
        dimension: int = 1,
    ) -> Tuple[float, float]:
        """Wrapper for :meth:`KnowledgeGraph.validate_topology`."""

        dist, diff = self.graph.validate_topology(target, radii, dimension=dimension)
        self._record_event(
            "validate_topology",
            "Topology compared to target",
            dimension=dimension,
            distance=dist,
            dimension_diff=diff,
        )
        return dist, diff

    def fractal_information_density(  # pragma: no cover
        self, radii: Iterable[int], *, max_dim: int = 1
    ) -> float:
        """Wrapper for :meth:`KnowledgeGraph.fractal_information_density`."""

        val = self.graph.fractal_information_density(radii, max_dim=max_dim)
        self._record_event(
            "fractal_information_density",
            "Fractal information density computed",
            radii=list(radii),
        )
        return val

    def fractal_coverage(self) -> float:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.fractal_coverage`."""

        val = self.graph.fractal_coverage()
        self._record_event(
            "fractal_coverage",
            "Fractal coverage computed",
        )
        return val

    def ensure_fractal_coverage(  # pragma: no cover
        self,
        threshold: float,
        radii: Iterable[int],
        *,
        max_levels: int = 5,
    ) -> float:
        """Wrapper for :meth:`KnowledgeGraph.ensure_fractal_coverage`."""

        val = self.graph.ensure_fractal_coverage(
            threshold, radii, max_levels=max_levels
        )
        self._record_event(
            "ensure_fractal_coverage",
            "Fractal coverage enforced",
            threshold=threshold,
        )
        return val

    def diversification_score(  # pragma: no cover
        self,
        nodes: Iterable,
        radii: Iterable[int],
        *,
        max_dim: int = 1,
        dimension: int = 0,
    ) -> float:
        """Wrapper for :meth:`KnowledgeGraph.diversification_score`."""

        val = self.graph.diversification_score(
            nodes, radii, max_dim=max_dim, dimension=dimension
        )
        self._record_event(
            "diversification_score",
            "Diversification score computed",
            nodes=list(nodes),
        )
        return val

    def select_diverse_nodes(  # pragma: no cover
        self, candidates: Iterable[str], count: int, radii: Iterable[int]
    ) -> list[str]:
        """Wrapper for :meth:`KnowledgeGraph.select_diverse_nodes`."""

        selected = self.graph.select_diverse_nodes(candidates, count, radii)
        self._record_event(
            "select_diverse_nodes",
            "Diversification filter applied",
            count=count,
            selected=selected,
        )
        return selected

    def sample_diverse_chunks(
        self, count: int, radii: Iterable[int]
    ) -> list[str]:  # pragma: no cover
        """Return ``count`` chunk IDs that best cover unexplored graph regions.

        The helper computes diversification scores using ``radii`` and
        returns chunk IDs that maximize the score so subsequent prompts
        sample from novel subgraphs.
        """

        candidates = [
            n for n, d in self.graph.graph.nodes(data=True) if d.get("type") == "chunk"
        ]
        selected = self.select_diverse_nodes(candidates, count, radii)
        self._record_event(
            "sample_diverse_chunks",
            "Diverse chunks selected",
            count=count,
            selected=selected,
        )
        return selected

    def hyperbolic_neighbors(
        self, node_id: str, k: int = 5
    ) -> List[tuple[str, float]]:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.hyperbolic_neighbors`."""

        neighs = self.graph.hyperbolic_neighbors(node_id, k=k)
        self._record_event(
            "hyperbolic_neighbors",
            "Hyperbolic nearest neighbors computed",
            node=node_id,
            k=k,
        )
        return neighs

    def hyperbolic_reasoning(  # pragma: no cover
        self, start: str, goal: str, *, max_steps: int = 5
    ) -> List[str]:
        """Wrapper for :meth:`KnowledgeGraph.hyperbolic_reasoning`."""

        path = self.graph.hyperbolic_reasoning(start, goal, max_steps=max_steps)
        self._record_event(
            "hyperbolic_reasoning",
            "Hyperbolic reasoning path computed",
            start=start,
            goal=goal,
            max_steps=max_steps,
        )
        return path

    def hyperbolic_hypergraph_reasoning(  # pragma: no cover
        self,
        start: str,
        goal: str,
        *,
        penalty: float = 1.0,
        max_steps: int = 5,
    ) -> List[str]:
        """Wrapper for :meth:`KnowledgeGraph.hyperbolic_hypergraph_reasoning`."""

        path = self.graph.hyperbolic_hypergraph_reasoning(
            start,
            goal,
            penalty=penalty,
            max_steps=max_steps,
        )
        self._record_event(
            "hyperbolic_hypergraph_reasoning",
            "Hyperbolic hypergraph path computed",
            start=start,
            goal=goal,
            penalty=penalty,
            max_steps=max_steps,
        )
        return path

    def hyperbolic_multi_curvature_reasoning(  # pragma: no cover
        self,
        start: str,
        goal: str,
        *,
        curvatures: Iterable[float],
        weights: Optional[Dict[float, float]] = None,
        max_steps: int = 5,
    ) -> List[str]:
        """Wrapper for :meth:`KnowledgeGraph.hyperbolic_multi_curvature_reasoning`."""

        path = self.graph.hyperbolic_multi_curvature_reasoning(
            start,
            goal,
            curvatures=curvatures,
            weights=weights,
            max_steps=max_steps,
        )
        self._record_event(
            "hyperbolic_multi_curvature_reasoning",
            "Multi-curvature hyperbolic path computed",
            start=start,
            goal=goal,
            curvatures=list(curvatures),
            max_steps=max_steps,
        )
        return path

    def spectral_dimension(  # pragma: no cover
        self, times: Iterable[float]
    ) -> tuple[float, list[tuple[float, float]]]:
        """Wrapper for :meth:`KnowledgeGraph.spectral_dimension`."""

        dim, traces = self.graph.spectral_dimension(times)
        self._record_event(
            "spectral_dimension",
            "Spectral dimension computed",
            times=list(times),
        )
        return dim, traces

    def spectral_entropy(self, normed: bool = True) -> float:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.spectral_entropy`."""

        ent = self.graph.spectral_entropy(normed=normed)
        self._record_event(
            "spectral_entropy",
            "Spectral entropy computed",
            normed=normed,
        )
        return ent

    def spectral_gap(self, normed: bool = True) -> float:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.spectral_gap`."""

        gap = self.graph.spectral_gap(normed=normed)
        self._record_event(
            "spectral_gap",
            "Spectral gap computed",
            normed=normed,
        )
        return gap

    def laplacian_energy(self, normed: bool = True) -> float:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.laplacian_energy`."""

        energy = self.graph.laplacian_energy(normed=normed)
        self._record_event(
            "laplacian_energy",
            "Laplacian energy computed",
            normed=normed,
        )
        return energy

    def lacunarity(self, radius: int = 1) -> float:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.lacunarity`."""

        lac = self.graph.lacunarity(radius=radius)
        self._record_event(
            "lacunarity",
            "Graph lacunarity computed",
            radius=radius,
        )
        return lac

    def sheaf_laplacian(
        self, edge_attr: str = "sheaf_sign"
    ) -> np.ndarray:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.sheaf_laplacian`."""

        L = self.graph.sheaf_laplacian(edge_attr=edge_attr)
        self._record_event(
            "sheaf_laplacian",
            "Sheaf Laplacian computed",
            edge_attr=edge_attr,
        )
        return L

    def sheaf_convolution(  # pragma: no cover
        self,
        features: Dict[str, Iterable[float]],
        *,
        edge_attr: str = "sheaf_sign",
        alpha: float = 0.1,
    ) -> Dict[str, list[float]]:
        """Wrapper for :meth:`KnowledgeGraph.sheaf_convolution`."""

        result = self.graph.sheaf_convolution(
            features, edge_attr=edge_attr, alpha=alpha
        )
        self._record_event(
            "sheaf_convolution",
            "Sheaf convolution applied",
            edge_attr=edge_attr,
            alpha=alpha,
        )
        return result

    def sheaf_neural_network(  # pragma: no cover
        self,
        features: Dict[str, Iterable[float]],
        *,
        layers: int = 2,
        alpha: float = 0.1,
        edge_attr: str = "sheaf_sign",
    ) -> Dict[str, list[float]]:
        """Wrapper for :meth:`KnowledgeGraph.sheaf_neural_network`."""

        result = self.graph.sheaf_neural_network(
            features,
            layers=layers,
            alpha=alpha,
            edge_attr=edge_attr,
        )
        self._record_event(
            "sheaf_neural_network",
            "Sheaf neural network applied",
            layers=layers,
            alpha=alpha,
            edge_attr=edge_attr,
        )
        return result

    def sheaf_cohomology(  # pragma: no cover
        self, *, edge_attr: str = "sheaf_sign", tol: float = 1e-5
    ) -> int:
        """Wrapper for :meth:`KnowledgeGraph.sheaf_cohomology`."""

        val = self.graph.sheaf_cohomology(edge_attr=edge_attr, tol=tol)
        self._record_event(
            "sheaf_cohomology",
            "Sheaf cohomology computed",
            h1=val,
        )
        return val

    def sheaf_cohomology_blocksmith(  # pragma: no cover
        self,
        *,
        edge_attr: str = "sheaf_sign",
        block_size: int = 40000,
        tol: float = 1e-5,
    ) -> int:
        """Wrapper for :meth:`KnowledgeGraph.sheaf_cohomology_blocksmith`."""

        val = self.graph.sheaf_cohomology_blocksmith(
            edge_attr=edge_attr,
            block_size=block_size,
            tol=tol,
        )
        self._record_event(
            "sheaf_cohomology_blocksmith",
            "Sheaf cohomology approximated via Block-Smith",
            h1=val,
        )
        return val

    def resolve_sheaf_obstruction(  # pragma: no cover
        self, *, edge_attr: str = "sheaf_sign", max_iter: int = 10
    ) -> int:
        """Wrapper for :meth:`KnowledgeGraph.resolve_sheaf_obstruction`."""

        val = self.graph.resolve_sheaf_obstruction(
            edge_attr=edge_attr, max_iter=max_iter
        )
        self._record_event(
            "resolve_sheaf_obstruction",
            "Sheaf obstruction resolved",
            h1=val,
        )
        return val

    def sheaf_consistency_score(
        self, *, edge_attr: str = "sheaf_sign"
    ) -> float:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.sheaf_consistency_score`."""

        val = self.graph.sheaf_consistency_score(edge_attr=edge_attr)
        self._record_event(
            "sheaf_consistency_score",
            "Computed sheaf consistency score",
            score=val,
        )
        return val

    def sheaf_consistency_score_batched(  # pragma: no cover
        self,
        batches: Iterable[Iterable[str]],
        *,
        edge_attr: str = "sheaf_sign",
    ) -> list[float]:
        """Wrapper for :meth:`KnowledgeGraph.sheaf_consistency_score_batched`."""

        scores = self.graph.sheaf_consistency_score_batched(
            batches, edge_attr=edge_attr
        )
        self._record_event(
            "sheaf_consistency_score_batched",
            "Computed batched sheaf consistency scores",
        )
        return scores

    def spectral_bound_exceeded(  # pragma: no cover
        self, k: int, tau: float, *, edge_attr: str = "sheaf_sign"
    ) -> bool:
        """Wrapper for :meth:`KnowledgeGraph.spectral_bound_exceeded`."""

        flag = self.graph.spectral_bound_exceeded(k, tau, edge_attr=edge_attr)
        self._record_event(
            "spectral_bound_exceeded",
            "Checked spectral bound",
            k=k,
            tau=tau,
            exceeded=flag,
        )
        return flag

    def path_to_text(self, path: Iterable) -> str:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.path_to_text`."""

        text = self.graph.path_to_text(path)
        self._record_event(
            "path_to_text",
            "Converted graph path to text",
            path=list(path),
        )
        return text

    def neighborhood_to_sentence(self, path: Iterable) -> str:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.neighborhood_to_sentence`."""

        text = self.graph.neighborhood_to_sentence(path)
        self._record_event(
            "neighborhood_to_sentence",
            "Converted graph path to text",
            path=list(path),
        )
        return text

    def subgraph_to_text(self, nodes: Iterable) -> str:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.subgraph_to_text`."""

        text = self.graph.subgraph_to_text(nodes)
        self._record_event(
            "subgraph_to_text",
            "Converted graph subgraph to text",
            nodes=list(nodes),
        )
        return text

    def graph_to_text(self) -> str:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.graph_to_text`."""

        text = self.graph.graph_to_text()
        self._record_event(
            "graph_to_text",
            "Converted entire graph to text",
        )
        return text

    def graph_information_bottleneck(  # pragma: no cover
        self,
        labels: Dict[str, int],
        *,
        beta: float = 1.0,
    ) -> float:
        """Wrapper for :meth:`KnowledgeGraph.graph_information_bottleneck`."""

        loss = self.graph.graph_information_bottleneck(labels, beta=beta)
        self._record_event(
            "graph_information_bottleneck",
            "Information bottleneck computed",
            beta=beta,
        )
        return loss

    def graph_entropy(self, *, base: float = 2.0) -> float:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.graph_entropy`."""

        val = self.graph.graph_entropy(base=base)
        self._record_event("graph_entropy", "Graph entropy computed", base=base)
        return val

    def subgraph_entropy(
        self, nodes: Iterable, *, base: float = 2.0
    ) -> float:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.subgraph_entropy`."""

        val = self.graph.subgraph_entropy(nodes, base=base)
        self._record_event(
            "subgraph_entropy",
            "Subgraph entropy computed",
            base=base,
            nodes=list(nodes),
        )
        return val

    def structural_entropy(
        self, tau: int, *, base: float = 2.0
    ) -> float:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.structural_entropy`."""

        val = self.graph.structural_entropy(tau, base=base)
        self._record_event(
            "structural_entropy",
            "Structural entropy computed",
            base=base,
            tau=tau,
        )
        return val

    def adaptive_triangle_threshold(  # pragma: no cover
        self,
        *,
        weight: str = "weight",
        base: float = 2.0,
        scale: float = 10.0,
    ) -> int:
        """Wrapper for :meth:`KnowledgeGraph.adaptive_triangle_threshold`."""

        val = self.graph.adaptive_triangle_threshold(
            weight=weight, base=base, scale=scale
        )
        self._record_event(
            "adaptive_triangle_threshold",
            "Entropy-based triangle threshold computed",
            weight=weight,
            base=base,
            scale=scale,
            value=val,
        )
        return val

    def autotune_step(  # pragma: no cover
        self,
        labels: Dict[str, int],
        motifs: Iterable[nx.Graph],
        state: "AutoTuneState",
        *,
        node_attr: str = "embedding",
        weights: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0),
        lr: float = 0.1,
        penalty_cfg: Optional[Dict[str, float]] = None,
        latency: float = 0.0,
    ) -> Dict[str, Any]:
        """Wrapper for :meth:`KnowledgeGraph.autotune_step`."""

        res = self.graph.autotune_step(
            labels,
            motifs,
            state,
            node_attr=node_attr,
            weights=weights,
            lr=lr,
            penalty_cfg=penalty_cfg,
            latency=latency,
        )
        from ..analysis.autotune import update_theta as _update_theta

        _update_theta(state, res)
        self.graph.graph["j_cost"] = float(res["cost"])
        update_metric("j_cost", float(res["cost"]))
        update_metric("autotune_cost", float(res["cost"]))
        self._record_event(
            "autotuning_layer",
            "Autotuning iteration executed",
            cost=res["cost"],
            tau=res["tau"],
            eps=res["eps"],
        )
        try:
            push_metrics(
                {
                    "autotune_cost": float(res["cost"]),
                    "autotune_tau": float(res["tau"]),
                    "autotune_eps": float(res["eps"]),
                }
            )
        except Exception:
            pass
        try:
            self.log_cycle_metrics()
        except Exception:
            pass
        return res

    def svgp_ei_propose(  # pragma: no cover
        self,
        history: Sequence[tuple[Sequence[float], float]],
        bounds: Sequence[tuple[float, float]],
        *,
        m: int = 100,
        n_samples: int = 256,
    ) -> list[float]:
        """Wrapper for :meth:`KnowledgeGraph.svgp_ei_propose`."""

        vec = self.graph.svgp_ei_propose(
            history,
            bounds,
            m=m,
            n_samples=n_samples,
        )
        self._record_event(
            "svgp_ei_propose",
            "SVGP-EI hyperparameter proposal",
        )
        return vec

    def kw_gradient(  # pragma: no cover
        self, f: Callable[[float], float], x: float, *, h: float = 1.0, n: int = 4
    ) -> float:
        """Wrapper for :meth:`KnowledgeGraph.kw_gradient`."""

        val = self.graph.kw_gradient(f, x, h=h, n=n)
        self._record_event("kw_gradient", "KW gradient estimate")
        return val

    def prototype_subgraph(  # pragma: no cover
        self,
        labels: Dict[str, int],
        class_id: int,
        *,
        radius: int = 1,
    ) -> nx.Graph:
        """Wrapper for :meth:`KnowledgeGraph.prototype_subgraph`."""

        sub = self.graph.prototype_subgraph(labels, class_id, radius=radius)
        self._record_event(
            "prototype_subgraph",
            "Prototype subgraph extracted",
            class_id=class_id,
            radius=radius,
        )
        return sub

    def select_mdl_motifs(
        self, motifs: Iterable[nx.Graph]
    ) -> List[nx.Graph]:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.select_mdl_motifs`."""

        selected = self.graph.select_mdl_motifs(motifs)
        self._record_event(
            "select_mdl_motifs",
            "Motifs selected via MDL",
            count=len(selected),
        )
        return selected

    def laplacian_spectrum(self, normed: bool = True) -> np.ndarray:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.laplacian_spectrum`."""

        evals = self.graph.laplacian_spectrum(normed=normed)
        self._record_event(
            "laplacian_spectrum",
            "Laplacian spectrum computed",
            normed=normed,
        )
        return evals

    def spectral_density(  # pragma: no cover
        self, bins: int = 50, *, normed: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Wrapper for :meth:`KnowledgeGraph.spectral_density`."""

        hist, edges = self.graph.spectral_density(bins=bins, normed=normed)
        self._record_event(
            "spectral_density",
            "Spectral density computed",
            bins=bins,
            normed=normed,
        )
        return hist, edges

    def graph_fourier_transform(  # pragma: no cover
        self, signal: Dict[str, float] | np.ndarray, *, normed: bool = True
    ) -> np.ndarray:
        """Wrapper for :meth:`KnowledgeGraph.graph_fourier_transform`."""

        coeffs = self.graph.graph_fourier_transform(signal, normed=normed)
        self._record_event(
            "graph_fourier_transform",
            "Graph Fourier transform computed",
            normed=normed,
        )
        return coeffs

    def inverse_graph_fourier_transform(  # pragma: no cover
        self, coeffs: np.ndarray, *, normed: bool = True
    ) -> np.ndarray:
        """Wrapper for :meth:`KnowledgeGraph.inverse_graph_fourier_transform`."""

        signal = self.graph.inverse_graph_fourier_transform(coeffs, normed=normed)
        self._record_event(
            "inverse_graph_fourier_transform",
            "Inverse graph Fourier transform computed",
            normed=normed,
        )
        return signal

    def persistence_entropy(self, dimension: int = 0) -> float:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.persistence_entropy`."""

        ent = self.graph.persistence_entropy(dimension)
        self._record_event(
            "persistence_entropy",
            "Persistence entropy computed",
            dimension=dimension,
        )
        return ent

    def persistence_diagrams(
        self, max_dim: int = 2
    ) -> Dict[int, np.ndarray]:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.persistence_diagrams`."""

        diags = self.graph.persistence_diagrams(max_dim)
        self._record_event(
            "persistence_diagrams",
            "Persistence diagrams computed",
            max_dim=max_dim,
        )
        return diags

    def persistence_wasserstein_distance(  # pragma: no cover
        self, other: nx.Graph, *, dimension: int = 0, order: int = 1
    ) -> float:
        """Wrapper for :meth:`KnowledgeGraph.persistence_wasserstein_distance`."""

        dist = self.graph.persistence_wasserstein_distance(
            other, dimension=dimension, order=order
        )
        self._record_event(
            "persistence_wasserstein_distance",
            f"distance={dist}",
            dimension=dimension,
            order=order,
        )
        return dist

    def topological_signature(
        self, max_dim: int = 1
    ) -> Dict[str, Any]:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.topological_signature`."""

        sig = self.graph.topological_signature(max_dim=max_dim)
        self._record_event(
            "topological_signature",
            "Topological signature computed",
            max_dim=max_dim,
        )
        return sig

    def topological_signature_hash(self, max_dim: int = 1) -> str:  # pragma: no cover
        """Return an MD5 digest of the graph's topological signature.

        This simply forwards to
        :meth:`KnowledgeGraph.topological_signature_hash` while logging the
        access as an event so downstream consumers can trace when the
        signature was computed.

        Parameters
        ----------
        max_dim:
            Maximum homology dimension considered when computing the
            persistence diagrams.
        """

        h = self.graph.topological_signature_hash(max_dim=max_dim)
        self._record_event(
            "topological_signature_hash",
            "Topological signature hashed",
            max_dim=max_dim,
        )
        return h

    def fractalize_level(
        self, radius: int
    ) -> tuple[nx.Graph, Dict[str, int]]:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.fractalize_level`."""

        coarse, mapping = self.graph.fractalize_level(radius)
        self._record_event(
            "fractalize_level",
            "Graph coarse-grained via box covering",
            radius=radius,
        )
        return coarse, mapping

    def fractalize_optimal(  # pragma: no cover
        self, radii: Iterable[int]
    ) -> tuple[nx.Graph, Dict[str, int], int]:
        """Wrapper for :meth:`KnowledgeGraph.fractalize_optimal`."""

        coarse, mapping, radius = self.graph.fractalize_optimal(radii)
        self._record_event(
            "fractalize_optimal",
            "Graph coarse-grained at optimal radius",
            radii=list(radii),
            chosen_radius=radius,
        )
        return coarse, mapping, radius

    def record_feedback(self, record_id: str, comment: str) -> None:  # pragma: no cover
        """Store user feedback linked to a dataset record."""

        entry = {
            "record_id": record_id,
            "comment": comment,
            "time": datetime.now(timezone.utc).isoformat(),
        }
        self.feedback.append(entry)
        self._record_event("record_feedback", "Feedback recorded", record_id=record_id)

    def verify_statements(  # pragma: no cover
        self, statements: Iterable[tuple[str, str, str]], *, max_hops: int = 3
    ) -> float:
        """Wrapper for :meth:`KnowledgeGraph.verify_statements`."""

        stmts = list(statements)
        score = self.graph.verify_statements(stmts, max_hops=max_hops)
        self._record_event(
            "verify_statements",
            "Statements verified",
            count=len(stmts),
            score=score,
        )
        return score

    def verify_answer(
        self, answer: str, *, max_hops: int = 3
    ) -> float:  # pragma: no cover
        """Return confidence score for ``answer`` based on graph facts.

        The text is parsed with :func:`extract_facts` to obtain triples which are
        then passed to :meth:`verify_statements`. The average confidence is
        returned and also recorded in the dataset history.
        """

        from ..utils.fact_extraction import extract_facts

        facts = extract_facts(answer)
        triples = [(f["subject"], f["predicate"], f["object"]) for f in facts]
        score = self.verify_statements(triples, max_hops=max_hops) if triples else 0.0
        self._record_event(
            "verify_answer",
            "Answer verified",
            score=score,
            facts=len(triples),
        )
        return score

    def verify_qa_pairs(  # pragma: no cover
        self, pairs: Iterable["QAPair"], *, max_hops: int = 3
    ) -> list["QAPair"]:
        """Annotate ``pairs`` with a confidence score using graph verification."""

        from datacreek.models.qa import QAPair

        verified: list[QAPair] = []
        for p in pairs:
            score = self.verify_answer(p.answer, max_hops=max_hops)
            p.confidence = score
            verified.append(p)
        self._record_event(
            "verify_qa_pairs",
            "QA pairs verified",
            count=len(verified),
        )
        return verified

    def _update_edge_usage(self, path: Iterable[str]) -> None:  # pragma: no cover
        """Increment counters for edges appearing in ``path``."""

        nodes = list(path)
        for u, v in zip(nodes[:-1], nodes[1:]):
            key = f"{u}->{v}"
            self.edge_usage[key] = self.edge_usage.get(key, 0) + 1

    def coverage_stats(self) -> Dict[str, float]:  # pragma: no cover
        """Return coverage statistics based on traversed edges.

        ``edge_coverage`` measures the ratio of unique edges encountered during
        generation versus the total number present in the graph. ``betti_coverage``
        applies the Betti-1 formula on the subgraph induced by those edges to
        indicate how much of the global cycle structure was explored.
        """

        total = self.graph.graph.number_of_edges()
        used_pairs = {tuple(k.split("->", 1)) for k in self.edge_usage}
        sub = nx.Graph()
        sub.add_nodes_from(self.graph.graph.nodes())
        sub.add_edges_from(used_pairs)
        betti_total = (
            self.graph.graph.number_of_edges()
            - self.graph.graph.number_of_nodes()
            + nx.number_connected_components(self.graph.graph)
        )
        betti_sub = (
            sub.number_of_edges()
            - sub.number_of_nodes()
            + nx.number_connected_components(sub)
        )
        edge_cov = len(used_pairs) / total if total else 0.0
        betti_cov = betti_sub / betti_total if betti_total else 0.0
        self._record_event(
            "coverage_stats",
            "Coverage computed",
            edge_coverage=edge_cov,
            betti_coverage=betti_cov,
        )
        return {"edge_coverage": edge_cov, "betti_coverage": betti_cov}

    def invariants_dashboard(  # pragma: no cover
        self,
        radii: Iterable[int],
        *,
        max_dim: int = 1,
        normed: bool = True,
    ) -> Dict[str, Any]:
        """Compute key graph invariants for monitoring.

        Parameters
        ----------
        radii:
            Box radii used when estimating the fractal dimension.
        max_dim:
            Maximum homology dimension for :func:`topological_signature`.
        normed:
            Whether to use the normalized Laplacian for the spectral gap.

        Returns
        -------
        dict
            ``entropy`` of the degree distribution, ``fractal_dim`` from
            box-counting, ``spectral_gap`` of the Laplacian and the persistence
            ``signature``.
        """

        entropy = self.graph_entropy()
        dim, _ = self.fractal_dimension(radii)
        gap = self.spectral_gap(normed=normed)
        signature = self.topological_signature(max_dim=max_dim)

        self._record_event(
            "invariants_dashboard",
            "Graph invariants computed",
            entropy=entropy,
            fractal_dim=dim,
            spectral_gap=gap,
        )

        return {
            "entropy": entropy,
            "fractal_dim": dim,
            "spectral_gap": gap,
            "signature": signature,
        }

    def monitor_and_remediate(  # pragma: no cover
        self,
        radii: Iterable[int],
        *,
        policy: InvariantPolicy | None = None,
    ) -> Dict[str, Any]:
        """Enforce invariants and rerun cleanup using ``policy``.

        The dashboard metrics are recomputed after each iteration until the
        entropy, spectral gap and fractal dimension fall within the limits set by
        ``policy``.
        """

        p = policy or self.policy

        metrics = self.invariants_dashboard(radii)
        for _ in range(p.loops):
            changed = False
            if metrics["entropy"] > p.entropy_max:
                self.run_quality_layer(use_neo4j=False)
                self.run_information_layer({}, [], beta=1.0)
                changed = True
            if metrics["spectral_gap"] < p.gap_min:
                self.run_topological_perception_layer(
                    self.graph.graph.to_undirected(), radii, loops=1, epsilon=0.0
                )
                changed = True
            if not (p.fractal_range[0] <= metrics["fractal_dim"] <= p.fractal_range[1]):
                self.run_fractal_layer(radii, max_levels=1)
                changed = True
            if not changed:
                break
            metrics = self.invariants_dashboard(radii)

        self._record_event(
            "monitor_and_remediate",
            "Invariant policy executed",
            entropy=metrics.get("entropy"),
            fractal_dim=metrics.get("fractal_dim"),
            spectral_gap=metrics.get("spectral_gap"),
        )
        return metrics

    def dynamic_reconfigure(
        self, radii: Iterable[int], *, loops: int = 1
    ) -> None:  # pragma: no cover
        """Refresh fractal levels and resolve cohomology.

        Parameters
        ----------
        radii:
            Box radii used when calling :meth:`annotate_mdl_levels`.
        loops:
            Number of iterations for fractal and sheaf updates. Each
            iteration inserts at most one new level and attempts to
            resolve the :math:`H^{1}` obstruction class.
        """

        self.annotate_mdl_levels(radii, max_levels=loops)
        if self.sheaf_cohomology() > 0:
            self.resolve_sheaf_obstruction(max_iter=loops)
        self.graph.compute_hyper_sagnn_embeddings()
        self._record_event(
            "dynamic_reconfigure",
            "Hierarchy and sheaf refreshed",
            loops=loops,
        )

    def update_hypergraph_structure(  # pragma: no cover
        self, *, k: int = 5, threshold: float = 0.8
    ) -> list[str]:
        """Predict and insert new hyperedges via embeddings.

        Parameters
        ----------
        k:
            Maximum number of suggestions to insert.
        threshold:
            Cosine similarity cutoff between Hyper-SAGNN embeddings for
            proposing a new hyperedge.
        """

        suggestions = self.graph.predict_hyperedges(k=k, threshold=threshold)
        for edge_id, nodes in suggestions:
            try:
                self.graph.add_hyperedge(
                    edge_id, nodes, relation="predicted", source="auto"
                )
            except ValueError:
                continue
        self._record_event(
            "hypergraph_update",
            "Hyperedges predicted",
            count=len(suggestions),
        )
        return [e for e, _ in suggestions]

    # ------------------------------------------------------------------
    # Helper utilities

    def _enforce_policy(self, radii: Iterable[int]) -> None:  # pragma: no cover
        """Apply dynamic reconfiguration and monitoring policy.

        The sequence ``dynamic_reconfigure`` → ``update_hypergraph_structure``
        → ``monitor_and_remediate`` refreshes fractal levels, resolves sheaf
        inconsistencies, proposes new hyperedges and cleans up until the
        invariant thresholds are satisfied.
        """

        self.dynamic_reconfigure(radii)
        self.update_hypergraph_structure()
        self.monitor_and_remediate(radii)

    async def start_policy_monitor(
        self,
        radii: Iterable[int],
        *,
        interval: float = 60.0,
    ) -> None:
        r"""Periodically enforce invariant limits in the background.

        The loop calls :meth:`_enforce_policy` every ``interval`` seconds so the
        entropy :math:`H`, fractal dimension :math:`d_B` and Laplacian gap
        :math:`\lambda_1` remain within the thresholds set by :attr:`policy`.
        """

        if self._policy_event is None:
            self._policy_event = asyncio.Event()
        while not self._policy_event.is_set():
            try:
                self._enforce_policy(radii)
            except Exception:
                logger.exception("Policy monitor iteration failed")
            await asyncio.sleep(interval)

    def stop_policy_monitor(self) -> None:  # pragma: no cover
        """Signal the asynchronous policy monitor to terminate."""

        if self._policy_event is not None:
            self._policy_event.set()

    def start_policy_monitor_thread(  # pragma: no cover
        self, radii: Iterable[int], *, interval: float = 60.0
    ) -> None:
        """Run :meth:`start_policy_monitor` in a background thread."""

        if self._policy_thread is not None and self._policy_thread.is_alive():
            return

        def runner() -> None:  # pragma: no cover
            asyncio.run(self.start_policy_monitor(radii, interval=interval))

        self._policy_thread = threading.Thread(target=runner, daemon=True)
        self._policy_thread.start()

    def stop_policy_monitor_thread(self) -> None:  # pragma: no cover
        """Terminate the background policy monitor thread."""

        self.stop_policy_monitor()
        if self._policy_thread is not None:
            self._policy_thread.join(timeout=1.0)
            self._policy_thread = None

    def build_fractal_hierarchy(  # pragma: no cover
        self, radii: Iterable[int], *, max_levels: int = 5
    ) -> list[tuple[nx.Graph, Dict[str, int], int]]:
        """Wrapper for :meth:`KnowledgeGraph.build_fractal_hierarchy`."""

        hierarchy = self.graph.build_fractal_hierarchy(radii, max_levels=max_levels)
        self._record_event(
            "build_fractal_hierarchy",
            "Fractal hierarchy constructed",
            radii=list(radii),
            max_levels=max_levels,
        )
        return hierarchy

    def build_mdl_hierarchy(  # pragma: no cover
        self, radii: Iterable[int], *, max_levels: int = 5
    ) -> list[tuple[nx.Graph, Dict[str, int], int]]:
        """Wrapper for :meth:`KnowledgeGraph.build_mdl_hierarchy`."""

        hierarchy = self.graph.build_mdl_hierarchy(radii, max_levels=max_levels)
        self._record_event(
            "build_mdl_hierarchy",
            "MDL hierarchy constructed",
            radii=list(radii),
            max_levels=max_levels,
        )
        return hierarchy

    def annotate_fractal_levels(  # pragma: no cover
        self, radii: Iterable[int], *, max_levels: int = 5
    ) -> None:
        """Wrapper for :meth:`KnowledgeGraph.annotate_fractal_levels`."""

        self.graph.annotate_fractal_levels(radii, max_levels=max_levels)
        self._record_event(
            "annotate_fractal_levels",
            "Fractal levels annotated",
            radii=list(radii),
            max_levels=max_levels,
        )

    def annotate_mdl_levels(
        self, radii: Iterable[int], *, max_levels: int = 5
    ) -> None:  # pragma: no cover
        """Wrapper for :meth:`KnowledgeGraph.annotate_mdl_levels`."""

        self.graph.annotate_mdl_levels(radii, max_levels=max_levels)
        self._record_event(
            "annotate_mdl_levels",
            "Fractal levels annotated with MDL stopping",
            radii=list(radii),
            max_levels=max_levels,
        )

    def optimize_topology(  # pragma: no cover
        self,
        target: nx.Graph,
        *,
        dimension: int = 1,
        epsilon: float = 0.0,
        max_iter: int = 100,
        seed: int | None = None,
        use_generator: bool = False,
        use_netgan: bool = False,
    ) -> float:
        """Wrapper for :meth:`KnowledgeGraph.optimize_topology`."""
        before = self.graph.topological_signature(max_dim=dimension)
        dist = self.graph.optimize_topology(
            target,
            dimension=dimension,
            epsilon=epsilon,
            max_iter=max_iter,
            seed=seed,
            use_generator=use_generator,
            use_netgan=use_netgan,
        )
        after = self.graph.topological_signature(max_dim=dimension)
        self._record_event(
            "optimize_topology",
            "Topology adjusted via bottleneck minimization",
            dimension=dimension,
            epsilon=epsilon,
            max_iter=max_iter,
            use_generator=use_generator,
            use_netgan=use_netgan,
            before_entropy=before.get("entropy"),
            after_entropy=after.get("entropy"),
        )
        return dist

    def optimize_topology_iterative(  # pragma: no cover
        self,
        target: nx.Graph,
        *,
        loops: int = 8,
        dimension: int = 1,
        epsilon: float = 0.0,
        max_iter: int = 100,
        seed: int | None = None,
    ) -> float:
        """Run :meth:`KnowledgeGraph.optimize_topology_iterative`."""

        before = self.graph.topological_signature(max_dim=dimension)
        dist = self.graph.optimize_topology_iterative(
            target,
            loops=loops,
            dimension=dimension,
            epsilon=epsilon,
            max_iter=max_iter,
            seed=seed,
        )
        after = self.graph.topological_signature(max_dim=dimension)
        self._record_event(
            "optimize_topology_iterative",
            "Topology iteratively optimized",
            loops=loops,
            dimension=dimension,
            epsilon=epsilon,
            max_iter=max_iter,
            before_entropy=before.get("entropy"),
            after_entropy=after.get("entropy"),
        )
        return dist

    def run_quality_layer(  # pragma: no cover
        self,
        *,
        use_neo4j: bool | None = None,
        min_component_size: int = 2,
        similarity_threshold: float = 0.95,
        triangle_threshold: int = 1,
        link_threshold: float = 0.0,
        freeze_version: bool = False,
    ) -> Dict[str, Any]:
        """Clean the graph via quality checks.

        When ``use_neo4j`` is ``True`` (or ``None`` with a configured Neo4j
        driver) the routine delegates to :meth:`gds_quality_check` so the Neo4j
        GDS library performs the heavy lifting. Otherwise a lightweight
        in-memory variant :meth:`quality_check` is used. The result dictionary is
        returned verbatim and an event is logged for traceability.

        Parameters
        ----------
        use_neo4j:
            Whether to rely on Neo4j. ``None`` means auto-detect based on
            ``neo4j_driver``.
        min_component_size:
            Components smaller than this size are removed. ``None`` loads the
            default from ``configs/default.yaml`` (``cleanup.k_min``) so
            autotuning can adjust the parameter.
        similarity_threshold:
            Duplicate nodes above this value are merged.
        triangle_threshold:
            Edges incident to nodes with fewer triangles are pruned.
        link_threshold:
            Minimum score for suggested links to be added.
        freeze_version:
            When ``True`` a ``_clean0`` version of the dataset is written back
            to Neo4j for auditing.

        Returns
        -------
        dict
            Summary metrics from the underlying quality check.
        """

        if use_neo4j is None:
            use_neo4j = self.neo4j_driver is not None

        if use_neo4j:
            result = self.gds_quality_check(
                driver=self.neo4j_driver,
                dataset=self.name,
                min_component_size=min_component_size,
                similarity_threshold=similarity_threshold,
                triangle_threshold=triangle_threshold,
                link_threshold=link_threshold,
                freeze_version=freeze_version,
            )
        else:
            result = self.quality_check(
                min_component_size=min_component_size,
                triangle_threshold=triangle_threshold,
                similarity=similarity_threshold,
                link_threshold=link_threshold,
            )

        self._record_event(
            "quality_layer",
            "Graph cleaned via quality layer",
            use_neo4j=use_neo4j,
            freeze_version=freeze_version,
            **result,
        )

        return result

    def run_fractal_layer(  # pragma: no cover
        self,
        radii: Iterable[int],
        *,
        max_levels: int = 5,
        delta: float = 0.01,
    ) -> Dict[str, Any]:
        """Fractalize the graph and annotate levels.

        The routine computes the fractal dimension using ``radii`` then builds
        a multi-scale hierarchy via :meth:`build_fractal_hierarchy`. Nodes are
        annotated with their ``fractal_level`` and the dimension is recomputed
        afterwards to quantify the distortion.

        Parameters
        ----------
        radii:
            Candidate box radii used for the box-covering algorithm.
        max_levels:
            Maximum hierarchy depth to construct.
        delta:
            Threshold on the absolute change in fractal dimension signalling
            convergence.

        Returns
        -------
        dict
            Dictionary with ``dimension_before``/``dimension_after`` and the
            number of hierarchy ``levels`` constructed.
        """

        dim_before, _ = self.fractal_dimension(radii)
        hierarchy = self.build_fractal_hierarchy(radii, max_levels=max_levels)
        self.annotate_mdl_levels(radii, max_levels=max_levels)
        dim_after, _ = self.fractal_dimension(radii)
        diff = abs(dim_after - dim_before)

        self._record_event(
            "fractal_layer",
            "Graph fractalized",
            radii=list(radii),
            max_levels=max_levels,
            dimension_before=dim_before,
            dimension_after=dim_after,
            diff=diff,
            converged=diff <= delta,
        )

        return {
            "dimension_before": dim_before,
            "dimension_after": dim_after,
            "diff": diff,
            "levels": len(hierarchy),
            "converged": diff <= delta,
        }

    def run_embedding_layer(  # pragma: no cover
        self,
        *,
        node2vec_dim: int = 64,
        graphwave_scales: Iterable[float] = (0.5, 1.0),
        graphwave_points: int = 10,
        poincare_dim: int = 2,
        negative: int = 5,
        epochs: int = 50,
        learning_rate: float = 0.1,
        burn_in: int = 10,
        node2vec_p: float = 1.0,
        node2vec_q: float = 1.0,
        entropy_threshold: float = 0.5,
        radii: Iterable[float] = (0.5, 1.0),
    ) -> Dict[str, Any]:
        """Generate multi-geometry embeddings and monitor entropy.

        The routine computes Node2Vec, GraphWave, Poincar\u00e9 and GraphSAGE
        embeddings via :meth:`compute_multigeometric_embeddings`. The resulting
        ``embedding`` vectors are analysed through their entropy and fractal
        dimension. GraphWave entropy is enforced using
        :meth:`ensure_graphwave_entropy` with ``entropy_threshold``.

        Parameters
        ----------
        node2vec_dim:
            Dimension of the Node2Vec and GraphSAGE embeddings.
        graphwave_scales:
            Heat kernel scales for the GraphWave embeddings.
        graphwave_points:
            Number of sample points used by GraphWave.
        poincare_dim:
            Dimension of the Poincar\u00e9 embeddings.
        negative:
            Negative samples for the hyperbolic optimizer.
        epochs:
            Training epochs for the Poincar\u00e9 model.
        learning_rate:
            Learning rate of the Poincar\u00e9 optimizer.
        burn_in:
            Burn-in period for the Poincar\u00e9 optimizer.
        node2vec_p, node2vec_q:
            Return and in-out parameters of the Node2Vec random walks.
        entropy_threshold:
            Minimum acceptable GraphWave entropy.
        radii:
            Radii used to estimate the fractal dimension of embeddings.

        Returns
        -------
        dict
            Dictionary containing ``entropy`` of the embeddings, ``gw_entropy``
            after enforcement and the fractal ``dimension``.
        """

        self.compute_multigeometric_embeddings(
            node2vec_dim=node2vec_dim,
            graphwave_scales=graphwave_scales,
            graphwave_points=graphwave_points,
            poincare_dim=poincare_dim,
            negative=negative,
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
            node2vec_p=node2vec_p,
            node2vec_q=node2vec_q,
        )

        gw_entropy = self.ensure_graphwave_entropy(
            entropy_threshold,
            scales=graphwave_scales,
            num_points=graphwave_points,
        )
        entropy = self.embedding_entropy()
        dim, _ = self.embedding_box_counting_dimension("embedding", radii)

        self._record_event(
            "embedding_layer",
            "Multi-geometry embeddings computed",
            node2vec_dim=node2vec_dim,
            graphwave_scales=list(graphwave_scales),
            graphwave_points=graphwave_points,
            poincare_dim=poincare_dim,
            negative=negative,
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
            node2vec_p=node2vec_p,
            node2vec_q=node2vec_q,
            entropy=entropy,
            gw_entropy=gw_entropy,
            dimension=dim,
        )

        return {"entropy": entropy, "gw_entropy": gw_entropy, "dimension": dim}

    def run_hypergraph_layer(  # pragma: no cover
        self,
        *,
        embed_dim: int = 64,
        k: int = 5,
        threshold: float = 0.8,
    ) -> list[str]:
        """Refresh embeddings and predict new hyperedges."""

        self.compute_hyper_sagnn_embeddings(embed_dim=embed_dim)
        edges = self.update_hypergraph_structure(k=k, threshold=threshold)
        self._record_event(
            "hypergraph_layer",
            "Hypergraph structure updated",
            embed_dim=embed_dim,
            k=k,
            threshold=threshold,
            count=len(edges),
        )
        return edges

    def run_topological_perception_layer(  # pragma: no cover
        self,
        target: nx.Graph,
        radii: Iterable[int],
        *,
        loops: int = 8,
        dimension: int = 1,
        epsilon: float = 0.0,
        max_iter: int = 100,
    ) -> Dict[str, Any]:
        """Apply the Topological Perception Layer corrections.

        The routine resolves sheaf cohomology obstructions and edits
        ``perception_link`` edges so that the persistence diagram of the
        resulting graph approaches ``target`` within ``epsilon``.

        Parameters
        ----------
        target:
            Reference graph providing the desired topology.
        radii:
            Candidate radii used to update fractal levels after optimization.
        loops:
            Maximum number of optimization rounds.
        dimension:
            Homology dimension for the bottleneck objective.
        epsilon:
            Stop once the bottleneck distance drops below this value.
        max_iter:
            Maximum iterations per optimization round.

        Returns
        -------
        dict
            Dictionary containing ``distance`` to ``target`` and the sheaf
            cohomology before and after correction.
        """

        before_sig = self.topological_signature(max_dim=dimension)
        h1_before = self.sheaf_cohomology()
        h1_after = h1_before
        if h1_before > 0:
            h1_after = self.resolve_sheaf_obstruction(max_iter=loops)

        dist = self.optimize_topology_iterative(
            target,
            loops=loops,
            dimension=dimension,
            epsilon=epsilon,
            max_iter=max_iter,
        )
        after_sig = self.topological_signature(max_dim=dimension)

        # Refresh fractal levels to keep the hierarchy consistent
        self.annotate_mdl_levels(radii, max_levels=loops)

        self._record_event(
            "topological_perception_layer",
            "Topology and cohomology adjusted",
            loops=loops,
            dimension=dimension,
            epsilon=epsilon,
            max_iter=max_iter,
            distance=dist,
            h1_before=h1_before,
            h1_after=h1_after,
        )
        return {
            "distance": dist,
            "h1_before": h1_before,
            "h1_after": h1_after,
            "before": before_sig,
            "after": after_sig,
        }

    def tpl_correct_graph(  # pragma: no cover
        self,
        target: nx.Graph,
        *,
        epsilon: float = 0.1,
        dimension: int = 1,
        order: int = 1,
        max_iter: int = 5,
    ) -> Dict[str, Any]:
        """Wrapper for :meth:`KnowledgeGraph.tpl_correct_graph`."""

        res = self.graph.tpl_correct_graph(
            target,
            epsilon=epsilon,
            dimension=dimension,
            order=order,
            max_iter=max_iter,
        )
        self.graph.graph["tpl_w1"] = float(res["distance_after"])
        update_metric("tpl_w1", float(res["distance_after"]))
        try:
            from datacreek.analysis.monitoring import tpl_w1_g as _tpl_w1_gauge

            if _tpl_w1_gauge is not None:
                _tpl_w1_gauge.set(float(res["distance_after"]))
        except Exception:
            pass
        self._record_event(
            "tpl_correct_graph",
            "Wasserstein-based topology correction",
            epsilon=epsilon,
            dimension=dimension,
            order=order,
            corrected=res["corrected"],
            distance_after=float(res["distance_after"]),
        )
        return res

    def tpl_incremental(  # pragma: no cover
        self,
        *,
        radius: int = 1,
        dimension: int = 1,
    ) -> Dict[int, np.ndarray]:
        """Wrapper for :meth:`KnowledgeGraph.tpl_incremental`."""

        diags = self.graph.tpl_incremental(radius=radius, dimension=dimension)
        self._record_event(
            "tpl_incremental",
            "Incremental persistence update",
            radius=radius,
            dimension=dimension,
        )
        return diags

    def run_information_layer(  # pragma: no cover
        self,
        labels: Dict[str, int],
        motifs: Iterable[nx.Graph],
        *,
        beta: float = 1.0,
    ) -> Dict[str, Any]:
        """Optimize semantic units via IB and MDL.

        The routine computes the Graph Information Bottleneck loss using the
        current node embeddings and ``labels``. It then selects a subset of
        ``motifs`` that minimize the MDL description length. The MDL values
        before and after selection, along with the IB loss, are returned.

        Parameters
        ----------
        labels:
            Mapping from node ID to integer class label.
        motifs:
            Candidate subgraphs describing higher order structure.
        beta:
            Weight of the information regularizer.

        Returns
        -------
        dict
            Dictionary containing ``loss`` and MDL metrics.
        """

        from ..analysis.information import mdl_description_length

        loss = self.graph_information_bottleneck(labels, beta=beta)

        mdl_before = mdl_description_length(self.graph.graph.to_undirected(), motifs)
        selected = self.select_mdl_motifs(motifs)
        mdl_after = mdl_description_length(self.graph.graph.to_undirected(), selected)

        self._record_event(
            "information_layer",
            "Information bottleneck and MDL optimization",
            beta=beta,
            loss=loss,
            mdl_before=mdl_before,
            mdl_after=mdl_after,
            selected=len(selected),
        )

        return {
            "loss": loss,
            "mdl_before": mdl_before,
            "mdl_after": mdl_after,
            "selected": len(selected),
        }

    def run_generation_layer(  # pragma: no cover
        self,
        llm_call: Callable[[str], str] | None = None,
        *,
        query: str,
        k: int = 5,
        hops: int = 1,
        template: str = "qa",
        retries: int = 3,
        insert_patterns: Iterable[tuple[str, str]] = (),
        max_path: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate prompt/response pairs from an informative subgraph.

        ``query`` seeds a neighborhood search via
        :meth:`search_with_links_data`. Each resulting path is converted to
        text with :meth:`neighborhood_to_sentence`.  Tool call markers from
        ``insert_patterns`` are inserted and the prompt is fed to
        ``llm_call`` using :func:`~datacreek.utils.self_instruct.generate_with_self_instruct`.
        Paths longer than ``max_path`` yield a lower confidence value.

        Parameters
        ----------
        llm_call:
            Callable sending a prompt to a language model.
        query:
            Text used to select starting chunks.
        k:
            Number of initial ANN results.
        hops:
            Number of link hops when expanding the search.
        template:
            Name of the validation template for :func:`generate_with_self_instruct`.
        retries:
            Attempts allowed when the output does not validate.
        insert_patterns:
            Optional ``(name, regex)`` pairs for tool call insertion.
        max_path:
            Path length above which the confidence score is reduced.

        Returns
        -------
        list of dict
            ``prompt``/``response`` pairs with a ``confidence`` metric.
        """

        from ..utils.self_instruct import generate_with_self_instruct

        if llm_call is None:
            if self.llm_service is None:
                raise ValueError("llm_call required when no llm_service configured")
            llm_call = self.llm_service

        paths = self.search_with_links_data(query, k=k, hops=hops)
        records: List[Dict[str, Any]] = []
        for item in paths:
            p = item.get("path", [])
            prompt = self.neighborhood_to_sentence(p)
            if insert_patterns:
                prompt = self.auto_tool_calls(prompt, insert_patterns)
            response = generate_with_self_instruct(
                llm_call, prompt, template=template, retries=retries
            )
            conf = 1.0 if len(p) <= max_path else 0.5
            records.append({"prompt": prompt, "response": response, "confidence": conf})

        self._record_event(
            "generation_layer",
            "LLM generation from graph context",
            query=query,
            k=k,
            hops=hops,
            count=len(records),
        )

        # refresh fractal hierarchy and enforce invariant limits
        self._enforce_policy([1])

        return records

    async def run_generation_layer_async(
        self,
        llm_call: Callable[[str], Awaitable[str]] | None = None,
        *,
        query: str,
        k: int = 5,
        hops: int = 1,
        template: str = "qa",
        retries: int = 3,
        insert_patterns: Iterable[tuple[str, str]] = (),
        max_path: int = 3,
    ) -> List[Dict[str, Any]]:
        """Asynchronous variant of :meth:`run_generation_layer`."""

        from ..utils.self_instruct import generate_with_self_instruct_async

        if llm_call is None:
            if self.llm_service is None:
                raise ValueError("llm_call required when no llm_service configured")

            async def default_call(prompt: str) -> str:
                return (await self.llm_service.acomplete([prompt]))[0]

            llm_call = default_call

        paths = self.search_with_links_data(query, k=k, hops=hops)
        records: List[Dict[str, Any]] = []

        async def process(item: Dict[str, Any]) -> Dict[str, Any]:
            p = item.get("path", [])
            prompt = self.neighborhood_to_sentence(p)
            if insert_patterns:
                prompt = self.auto_tool_calls(prompt, insert_patterns)
            response = await generate_with_self_instruct_async(
                llm_call, prompt, template=template, retries=retries
            )
            conf = 1.0 if len(p) <= max_path else 0.5
            return {"prompt": prompt, "response": response, "confidence": conf}

        tasks = [process(item) for item in paths]
        if tasks:
            records = await asyncio.gather(*tasks)

        self._record_event(
            "generation_layer_async",
            "Async LLM generation from graph context",
            query=query,
            k=k,
            hops=hops,
            count=len(records),
        )

        self._enforce_policy([1])

        return records

    def run_compression_layer(  # pragma: no cover
        self,
        *,
        tol: float = 1e-3,
        max_count: int = 10,
    ) -> Dict[str, int]:
        """Compress embeddings and quotient by symmetry.

        The method clusters node embeddings via :func:`fractal_net_prune` and
        then computes the graph quotient under detected automorphisms.  The
        automorphism group order gives a measure of global redundancy while the
        embedding clusters highlight local similarity.

        Parameters
        ----------
        tol:
            Distance threshold when merging embeddings.
        max_count:
            Maximum number of automorphisms explored when estimating the group
            order and the quotient mapping.

        Returns
        -------
        dict
            Dictionary with ``clusters`` from pruning, ``classes`` in the
            quotient graph and the estimated automorphism ``order``.
        """

        mapping = self.prune_embeddings(tol=tol)
        clusters = len(set(mapping.values())) if mapping else 0
        order = self.automorphism_group_order(max_count=max_count)
        _, sym_map = self.quotient_by_symmetry(max_count=max_count)
        classes = len(set(sym_map.values())) if sym_map else 0

        self._record_event(
            "compression_layer",
            "Embeddings pruned and graph quotiented",
            tol=tol,
            max_count=max_count,
            clusters=clusters,
            classes=classes,
            order=order,
        )
        return {"clusters": clusters, "classes": classes, "order": order}

    def run_export_layer(  # pragma: no cover
        self,
        fmt: str = "chatml",
        *,
        radii: Iterable[int] = (1, 2, 3),
        max_levels: int = 5,
        ib_beta: float = 1.0,
    ) -> str:
        """Serialize prompts with fractal metadata.

        The helper gathers prompt records via :meth:`export_prompts` then
        converts them to ``fmt`` using :mod:`datacreek.utils.format_converter`.

        Parameters
        ----------
        fmt:
            Output format name (``"chatml"``, ``"alpaca"`` or ``"jsonl"``).
        radii:
            Candidate box radii used when annotating fractal levels.
        max_levels:
            Maximum hierarchy depth for the MDL-based annotation.
        ib_beta:
            Information bottleneck weight stored in each record.

        Returns
        -------
        str
            Dataset content in the requested format.
        """

        records = self.export_prompts(
            auto_fractal=True,
            radii=radii,
            max_levels=max_levels,
            mdl_radii=radii,
            ib_beta=ib_beta,
        )

        qa_pairs = [{"question": r["prompt"], "answer": ""} for r in records]

        from ..utils.format_converter import to_alpaca, to_chatml, to_jsonl

        if fmt == "jsonl":
            # retain metadata for full traceability
            formatted = to_jsonl(records)
        else:
            dispatch = {
                "alpaca": to_alpaca,
                "chatml": to_chatml,
            }
            if fmt not in dispatch:
                raise ValueError(f"unknown format: {fmt}")

            formatted = dispatch[fmt](qa_pairs)

        self._record_event(
            "export_layer",
            "Prompts serialized",
            format=fmt,
            count=len(records),
        )
        return formatted

    @monitor_after()
    def apply_perception(  # pragma: no cover
        self,
        node_id: str,
        new_text: str,
        *,
        perception_id: str | None = None,
        strength: float | None = None,
        threshold: float = 0.95,
    ) -> None:
        """Update ``node_id`` with ``new_text`` and check for duplicates.

        The modified text is persisted when Neo4j is configured, then a
        ``gds.nodeSimilarity`` query validates that no near-duplicates exist.
        Any matches above ``threshold`` are logged via a ``node_similarity_check``
        event for auditing purposes.

        Parameters
        ----------
        node_id:
            Identifier of the node to update.
        new_text:
            Replacement text to store on the node.
        perception_id:
            Optional label describing the semantic transformation.
        strength:
            Optional magnitude of the transformation.
        threshold:
            Minimum similarity required to flag matches.
        """

        self.graph.apply_perception(
            node_id,
            new_text,
            perception_id=perception_id,
            strength=strength,
        )
        self._record_event(
            "apply_perception",
            "Node perception applied",
            node_id=node_id,
            perception_id=perception_id,
            strength=strength,
        )

        if self.neo4j_driver and self.name:
            matches = self.graph.node_similarity(
                self.neo4j_driver,
                node_id,
                dataset=self.name,
                threshold=threshold,
            )
            if matches:
                self._record_event(
                    "node_similarity_check",
                    "Similarity check after perception",
                    node_id=node_id,
                    matches=matches,
                )

    @monitor_after()
    def apply_perception_all_nodes(  # pragma: no cover
        self,
        transform: Callable[[str], str],
        *,
        perception_id: str | None = None,
        strength: float | None = None,
        threshold: float = 0.95,
    ) -> Dict[object, str]:
        """Apply ``transform`` to every node then verify uniqueness.

        The transformation function is executed for each node's text and the
        result persisted. After all updates a ``gds.nodeSimilarity`` query runs
        for every node to detect possible duplicates. Matches are aggregated and
        recorded in a ``node_similarity_check`` event.

        Parameters
        ----------
        transform:
            Callable that receives the current text and returns the modified
            version.
        perception_id:
            Optional semantic label for the transformation applied.
        strength:
            Optional magnitude of the semantic change.
        threshold:
            Minimum similarity required to record a match.
        """

        updated = self.graph.apply_perception_all(
            transform,
            perception_id=perception_id,
            strength=strength,
        )
        self._record_event(
            "apply_perception_all_nodes",
            "Perception applied to all nodes",
            perception_id=perception_id,
            strength=strength,
            count=len(updated),
        )

        if self.neo4j_driver and self.name:
            all_matches: Dict[str, List[tuple[str, float]]] = {}
            for nid in updated:
                matches = self.graph.node_similarity(
                    self.neo4j_driver,
                    nid,
                    dataset=self.name,
                    threshold=threshold,
                )
                if matches:
                    all_matches[nid] = matches
            if all_matches:
                self._record_event(
                    "node_similarity_check",
                    "Similarity check after bulk perception",
                    matches=all_matches,
                )
        return updated

    def node_similarity(  # pragma: no cover
        self,
        node_id: str,
        *,
        threshold: float = 0.95,
        driver: Driver | None = None,
    ) -> List[tuple[str, float]]:
        """List nodes similar to ``node_id`` via Neo4j GDS.

        Each call emits a ``node_similarity_query`` event so that similarity
        lookups can be audited.  Provide a custom ``driver`` to query a
        different Neo4j instance.

        Parameters
        ----------
        node_id:
            Identifier of the node to compare with others.
        threshold:
            Minimum similarity score for results.
        driver:
            Optional Neo4j driver. Defaults to ``self.neo4j_driver``.
        """

        driver = driver or self.neo4j_driver
        if not driver:
            raise ValueError("Neo4j driver required")

        matches = self.graph.node_similarity(
            driver,
            node_id,
            dataset=self.name,
            threshold=threshold,
        )
        self._record_event(
            "node_similarity_query",
            "Similar nodes retrieved",
            node_id=node_id,
            threshold=threshold,
            count=len(matches),
        )
        return matches

    def auto_tool_calls(
        self, text: str, tools: Iterable[tuple[str, str]]
    ) -> str:  # pragma: no cover
        """Insert simple tool call placeholders into ``text``."""

        from ..utils import insert_tool_calls

        out = insert_tool_calls(text, tools)
        self._record_event("auto_tool_calls", "Tool calls inserted", tools=list(tools))
        return out

    @monitor_after()
    def auto_tool_calls_node(  # pragma: no cover
        self, node_id: str, tools: Iterable[tuple[str, str]]
    ) -> str:
        """Insert tool call placeholders into a graph node's text."""

        updated = self.graph.auto_tool_calls(node_id, tools)
        self._record_event(
            "auto_tool_calls_node",
            "Tool calls inserted on node",
            node_id=node_id,
            tools=list(tools),
        )
        return updated

    @monitor_after()
    def auto_tool_calls_all_nodes(  # pragma: no cover
        self, tools: Iterable[tuple[str, str]]
    ) -> Dict[object, str]:
        """Insert tool calls into every node in the graph."""

        updated = self.graph.auto_tool_calls_all(tools)
        self._record_event(
            "auto_tool_calls_all_nodes",
            "Tool calls inserted on all nodes",
            tools=list(tools),
        )
        return updated

    def gds_quality_check(  # pragma: no cover
        self,
        *,
        min_component_size: int = 2,
        similarity_threshold: float = 0.95,
        triangle_threshold: int = 1,
        link_threshold: float = 0.0,
        freeze_version: bool = False,
        driver: Driver | None = None,
    ) -> Dict[str, Any]:
        """Run Neo4j GDS quality checks via :class:`KnowledgeGraph`.

        Parameters
        ----------
        min_component_size:
            Components smaller than this size are removed.
        similarity_threshold:
            Duplicate nodes above this similarity are reported.
        triangle_threshold:
            Edges incident to nodes with fewer triangles are removed.
        link_threshold:
            Minimum score for suggested links to be added to the graph.
        freeze_version:
            When ``True``, a snapshot of the cleaned graph is written back to
            Neo4j using the ``"_clean0"`` dataset suffix.
        driver:
            Optional Neo4j driver. If ``None`` the instance's driver is used.
        """

        driver = driver or self.neo4j_driver
        if not driver:
            raise ValueError("Neo4j driver required")

        cfg = load_config()
        cleanup_cfg = cfg.get("cleanup", {})
        min_component_size = cleanup_cfg.get("k_min", min_component_size)
        similarity_threshold = cleanup_cfg.get("sigma", similarity_threshold)
        triangle_threshold = cleanup_cfg.get("tau", triangle_threshold)

        result = self.graph.gds_quality_check(
            driver,
            dataset=self.name,
            min_component_size=min_component_size,
            similarity_threshold=similarity_threshold,
            triangle_threshold=triangle_threshold,
            link_threshold=link_threshold,
        )
        self._record_event(
            "gds_quality_check",
            "Neo4j GDS quality check performed",
            min_component_size=min_component_size,
            similarity_threshold=similarity_threshold,
            triangle_threshold=triangle_threshold,
            link_threshold=link_threshold,
            triangles_removed=result.get("triangles_removed", 0),
            freeze_version=freeze_version,
        )
        if freeze_version:
            self.graph.to_neo4j(driver, dataset=f"{self.name}_clean0")
        return result

    def quality_check(  # pragma: no cover
        self,
        *,
        min_component_size: int = 2,
        triangle_threshold: int = 1,
        similarity: float = 0.95,
        link_threshold: float = 0.0,
    ) -> Dict[str, int]:
        """Run lightweight quality checks without Neo4j."""

        result = self.graph.quality_check(
            min_component_size=min_component_size,
            triangle_threshold=triangle_threshold,
            similarity=similarity,
            link_threshold=link_threshold,
        )
        self._record_event(
            "quality_check",
            "Graph quality check performed",
            **result,
        )
        return result

    def update_embeddings(self, node_type: str = "chunk") -> None:  # pragma: no cover
        """Materialize embeddings for nodes of ``node_type``."""

        self.graph.update_embeddings(node_type=node_type)
        self._record_event(
            "update_embeddings", "Embeddings materialized", node_type=node_type
        )

    def extract_facts(
        self, client: Optional["LLMClient"] = None
    ) -> None:  # pragma: no cover
        """Run fact extraction on all chunk nodes."""

        from datacreek.utils.fact_extraction import extract_facts

        for cid, data in list(self.graph.graph.nodes(data=True)):
            if data.get("type") != "chunk":
                continue
            facts = extract_facts(data.get("text", ""), client)
            for i, fact in enumerate(facts):
                fid = f"{cid}_fact_{i}"
                self.graph.add_fact(
                    fact["subject"],
                    fact["predicate"],
                    fact["object"],
                    fact_id=fid,
                    source=data.get("source"),
                )
                self.graph.graph.add_edge(
                    cid, fid, relation="has_fact", provenance=data.get("source")
                )
        self._record_event("extract_facts", "Facts extracted")

    def extract_entities(
        self, model: str | None = "en_core_web_sm"
    ) -> None:  # pragma: no cover
        """Run named entity recognition on all chunks."""

        self.graph.extract_entities(model=model)
        self._record_event("extract_entities", "Entities extracted", model=model)

    def find_conflicting_facts(
        self,
    ) -> List[tuple[str, str, Dict[str, List[str]]]]:  # pragma: no cover
        """Return edges with the same subject/predicate but different objects."""

        conflicts: Dict[tuple[str, str], Dict[str, List[str]]] = {}
        for u, v, edata in self.graph.graph.edges(data=True):
            rel = edata.get("relation")
            if not rel or rel in {
                "has_chunk",
                "next_chunk",
                "subject",
                "object",
                "has_fact",
                "mentions",
            }:
                continue
            prov = edata.get("provenance")
            key = (u, rel)
            conflicts.setdefault(key, {})
            conflicts[key].setdefault(v, [])
            if prov:
                conflicts[key][v].append(prov)

        result = []
        for key, obj_map in conflicts.items():
            if len(obj_map) > 1:
                result.append((key[0], key[1], obj_map))
        return result

    def mark_conflicting_facts(self) -> int:  # pragma: no cover
        """Annotate edges that belong to conflicting fact groups."""

        marked = self.graph.mark_conflicting_facts()
        if marked:
            msg = f"Marked {marked} conflicting facts"
            self._record_event("mark_conflicts", msg)
        return marked

    def validate_coherence(self) -> int:  # pragma: no cover
        """Flag logically inconsistent edges in the underlying graph."""

        marked = self.graph.validate_coherence()
        if marked:
            msg = f"Marked {marked} inconsistent relations"
            self._record_event("validate_coherence", msg)
        return marked

    def get_chunks_for_document(self, doc_id: str) -> list[str]:  # pragma: no cover
        return self.graph.get_chunks_for_document(doc_id)

    def get_images_for_document(self, doc_id: str) -> list[str]:  # pragma: no cover
        return self.graph.get_images_for_document(doc_id)

    def get_captions_for_document(self, doc_id: str) -> list[str]:  # pragma: no cover
        return self.graph.get_captions_for_document(doc_id)

    def get_caption_for_image(self, image_id: str) -> str | None:  # pragma: no cover
        return self.graph.get_caption_for_image(image_id)

    def get_audios_for_document(self, doc_id: str) -> list[str]:  # pragma: no cover
        return self.graph.get_audios_for_document(doc_id)

    def get_atoms_for_document(self, doc_id: str) -> list[str]:  # pragma: no cover
        return self.graph.get_atoms_for_document(doc_id)

    def get_molecules_for_document(self, doc_id: str) -> list[str]:  # pragma: no cover
        return self.graph.get_molecules_for_document(doc_id)

    def get_atoms_for_molecule(self, molecule_id: str) -> list[str]:  # pragma: no cover
        """Return atom IDs contained in ``molecule_id``."""

        atoms = self.graph.get_atoms_for_molecule(molecule_id)
        self._record_event("get_atoms_for_molecule", molecule_id)
        return atoms

    def get_sections_for_document(self, doc_id: str) -> list[str]:  # pragma: no cover
        return self.graph.get_sections_for_document(doc_id)

    def get_document_for_section(
        self, section_id: str
    ) -> str | None:  # pragma: no cover
        return self.graph.get_document_for_section(section_id)

    def get_document_for_chunk(self, chunk_id: str) -> str | None:  # pragma: no cover
        return self.graph.get_document_for_chunk(chunk_id)

    def get_chunks_for_section(self, section_id: str) -> list[str]:  # pragma: no cover
        return self.graph.get_chunks_for_section(section_id)

    def get_section_for_chunk(self, chunk_id: str) -> str | None:  # pragma: no cover
        return self.graph.get_section_for_chunk(chunk_id)

    def get_next_chunk(self, chunk_id: str) -> str | None:  # pragma: no cover
        """Return the chunk following ``chunk_id`` if any."""

        return self.graph.get_next_chunk(chunk_id)

    def get_previous_chunk(self, chunk_id: str) -> str | None:  # pragma: no cover
        """Return the chunk preceding ``chunk_id`` if any."""

        return self.graph.get_previous_chunk(chunk_id)

    def get_page_for_chunk(self, chunk_id: str) -> int | None:  # pragma: no cover
        """Return the page number stored on ``chunk_id``."""

        return self.graph.get_page_for_chunk(chunk_id)

    def get_page_for_section(self, section_id: str) -> int | None:  # pragma: no cover
        """Return the page number recorded for ``section_id``."""

        return self.graph.get_page_for_section(section_id)

    def get_next_section(self, section_id: str) -> str | None:  # pragma: no cover
        """Return the section following ``section_id`` if any."""

        return self.graph.get_next_section(section_id)

    def get_previous_section(self, section_id: str) -> str | None:  # pragma: no cover
        """Return the section preceding ``section_id`` if any."""

        return self.graph.get_previous_section(section_id)

    def get_facts_for_chunk(self, chunk_id: str) -> list[str]:  # pragma: no cover
        """Return fact IDs attached to ``chunk_id``."""

        return self.graph.get_facts_for_chunk(chunk_id)

    def get_facts_for_document(self, doc_id: str) -> list[str]:  # pragma: no cover
        """Return fact IDs related to any chunk of ``doc_id``."""

        return self.graph.get_facts_for_document(doc_id)

    def get_chunks_for_fact(self, fact_id: str) -> list[str]:  # pragma: no cover
        """Return chunk IDs referencing ``fact_id``."""

        return self.graph.get_chunks_for_fact(fact_id)

    def get_entities_for_fact(self, fact_id: str) -> list[str]:  # pragma: no cover
        """Return entity IDs linked as subject or object of ``fact_id``."""

        return self.graph.get_entities_for_fact(fact_id)

    def get_sections_for_fact(self, fact_id: str) -> list[str]:  # pragma: no cover
        """Return section IDs referencing ``fact_id`` via a chunk."""

        return self.graph.get_sections_for_fact(fact_id)

    def get_documents_for_fact(self, fact_id: str) -> list[str]:  # pragma: no cover
        """Return document IDs referencing ``fact_id`` via a chunk."""

        return self.graph.get_documents_for_fact(fact_id)

    def get_pages_for_fact(self, fact_id: str) -> list[int]:  # pragma: no cover
        """Return page numbers referencing ``fact_id`` via chunks."""

        return self.graph.get_pages_for_fact(fact_id)

    def get_facts_for_entity(self, entity_id: str) -> list[str]:  # pragma: no cover
        """Return fact IDs connected to ``entity_id``."""

        return self.graph.get_facts_for_entity(entity_id)

    def get_chunks_for_entity(self, entity_id: str) -> list[str]:  # pragma: no cover
        """Return chunk IDs mentioning ``entity_id``."""

        return self.graph.get_chunks_for_entity(entity_id)

    def get_entities_for_chunk(self, chunk_id: str) -> list[str]:  # pragma: no cover
        """Return entity IDs mentioned in ``chunk_id``."""

        return self.graph.get_entities_for_chunk(chunk_id)

    def get_entities_for_document(self, doc_id: str) -> list[str]:  # pragma: no cover
        """Return entity IDs mentioned anywhere in ``doc_id``."""

        return self.graph.get_entities_for_document(doc_id)

    def find_facts(  # pragma: no cover
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
    ) -> list[str]:
        """Wrapper for :meth:`KnowledgeGraph.find_facts`."""

        return self.graph.find_facts(
            subject=subject, predicate=predicate, object=object
        )

    def get_documents_for_entity(self, entity_id: str) -> list[str]:  # pragma: no cover
        """Return document IDs where ``entity_id`` is mentioned."""

        return self.graph.get_documents_for_entity(entity_id)

    def get_pages_for_entity(self, entity_id: str) -> list[int]:  # pragma: no cover
        """Return page numbers where ``entity_id`` is mentioned."""

        return self.graph.get_pages_for_entity(entity_id)

    def remove_chunk(self, chunk_id: str) -> None:  # pragma: no cover
        """Remove a chunk node from the dataset graph."""
        if self.graph.graph.has_node(chunk_id):
            preds = list(self.graph.graph.predecessors(chunk_id))
            self.graph.remove_chunk(chunk_id)
            if preds:
                msg = f"Removed chunk {chunk_id} from {preds[0]}"
                self._record_event("remove_chunk", msg)

    def remove_document(self, doc_id: str) -> None:  # pragma: no cover
        """Remove a document and all its chunks."""
        if not self.graph.graph.has_node(doc_id):
            return
        self.graph.remove_document(doc_id)
        self._record_event("remove_document", f"Removed document {doc_id}")

    def clone(self, name: Optional[str] = None) -> "DatasetBuilder":  # pragma: no cover
        """Return a deep copy of this dataset with a new optional name."""
        if name is not None:
            self.validate_name(name)
        clone = DatasetBuilder(
            self.dataset_type,
            name=name,
            graph=deepcopy(self.graph),
            use_hnsw=self.use_hnsw,
        )
        clone.owner_id = self.owner_id
        clone.history = self.history.copy()
        clone.versions = deepcopy(self.versions)
        clone.stage = self.stage
        return clone

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover
        """Serialize this dataset to a Python dictionary."""

        return {
            "dataset_type": self.dataset_type.value,
            "id": self.id,
            "name": self.name,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "history": self.history,
            "versions": self.versions,
            "ingested_docs": self.ingested_docs,
            "events": [
                {**asdict(e), "timestamp": e.timestamp.isoformat()} for e in self.events
            ],
            "graph": self.graph.to_dict(),
            "stage": int(self.stage),
            "use_hnsw": self.use_hnsw,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetBuilder":  # pragma: no cover
        """Rebuild a dataset from serialized ``data``."""

        name = data.get("name")
        if name is not None:
            cls.validate_name(name)
        ds = cls(
            DatasetType(data["dataset_type"]),
            name=name,
            use_hnsw=bool(data.get("use_hnsw", False)),
        )
        if "id" in data:
            ds.id = data["id"]
        if ts := data.get("created_at"):
            ds.created_at = datetime.fromisoformat(ts)
        if ts := data.get("accessed_at"):
            ds.accessed_at = datetime.fromisoformat(ts)
        ds.owner_id = data.get("owner_id")
        ds.history = list(data.get("history", []))
        ds.versions = list(data.get("versions", []))
        ds.ingested_docs = dict(data.get("ingested_docs", {}))
        ds.events = [
            HistoryEvent(
                e.get("operation"),
                e.get("message"),
                timestamp=(
                    datetime.fromisoformat(e["timestamp"])
                    if "timestamp" in e
                    else datetime.now(timezone.utc)
                ),
                params=e.get("params"),
            )
            for e in data.get("events", [])
        ]
        ds.graph = KnowledgeGraph.from_dict(data.get("graph", {}))
        ds.stage = DatasetStage(int(data.get("stage", 0)))
        return ds

    # ------------------------------------------------------------------
    # Redis helpers
    # ------------------------------------------------------------------

    def to_redis(  # pragma: no cover
        self, client: redis.Redis | redis.client.Pipeline, key: str | None = None
    ) -> str:
        """Persist the dataset in Redis under ``key``."""

        key = key or (self.name or "dataset")
        self.redis_key = key
        if self.name:
            self.validate_name(self.name)
        pipe = (
            client.pipeline()
            if not isinstance(client, redis.client.Pipeline)
            else client
        )
        pipe.set(key, json.dumps(self.to_dict()))
        events_key = f"{key}:events"
        pipe.delete(events_key)
        if self.events:
            pipe.rpush(
                events_key,
                *[
                    json.dumps(
                        {
                            "operation": ev.operation,
                            "message": ev.message,
                            "timestamp": ev.timestamp.isoformat(),
                            "params": ev.params,
                        }
                    )
                    for ev in self.events
                ],
            )
        if pipe is not client:
            pipe.execute()
        return key

    @classmethod
    def from_redis(  # pragma: no cover
        cls, client: redis.Redis | None, key: str, driver: Driver | None = None
    ) -> "DatasetBuilder":
        """Load a dataset from Redis and optionally Neo4j."""

        if client is None:
            raise ValueError("Redis client required")
        data = client.get(key)
        if data is None:
            raise KeyError(key)
        ds = cls.from_dict(json.loads(data))
        if ds.name:
            cls.validate_name(ds.name)
        ds.redis_client = client
        ds.neo4j_driver = driver
        ds.redis_key = key
        require = os.getenv("DATACREEK_REQUIRE_PERSISTENCE", "1") != "0"
        if require and driver is None:
            raise ValueError("Neo4j driver required")
        graph = get_redis_graph(ds.name)
        if graph is not None:
            try:
                DatasetBuilder.load_redis_graph.__wrapped__(ds, graph)
            except Exception:
                logger.exception("Failed to load graph %s from RedisGraph", ds.name)
        events_key = f"{key}:events"
        raw_events = client.lrange(events_key, 0, -1)
        if raw_events:
            ds.events = []
            for e in raw_events:
                ev = json.loads(e)
                ts = ev.get("timestamp")
                if ts:
                    ev["timestamp"] = datetime.fromisoformat(ts)
                ds.events.append(HistoryEvent(**ev))
            ds.history = [ev.message for ev in ds.events]
        ds._touch()
        return ds

    @persist_after
    def ingest_file(  # pragma: no cover
        self,
        path: str,
        doc_id: str | None = None,
        *,
        config: dict[str, Any] | None = None,
        high_res: bool = False,
        ocr: bool = False,
        use_unstructured: bool | None = None,
        extract_entities: bool = False,
        extract_facts: bool = False,
        client: "LLMClient" | None = None,
        options: "IngestOptions" | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> str:
        """Parse ``path`` and ingest its content into the dataset."""

        from .ingest import ingest_into_dataset

        if options is not None:
            config = options.config
            high_res = options.high_res
            ocr = options.ocr
            use_unstructured = options.use_unstructured
            extract_entities = options.extract_entities
            extract_facts = options.extract_facts

        doc = ingest_into_dataset(
            path,
            self,
            doc_id=doc_id,
            config=config,
            high_res=high_res,
            ocr=ocr,
            use_unstructured=use_unstructured,
            extract_entities=extract_entities,
            extract_facts=extract_facts,
            client=client,
            options=None,
            progress_callback=progress_callback,
        )
        self.ingested_docs[doc] = {
            "path": path,
            "config": config,
            "high_res": high_res,
            "ocr": ocr,
            "use_unstructured": use_unstructured,
            "extract_entities": extract_entities,
            "extract_facts": extract_facts,
        }
        self.stage = max(self.stage, DatasetStage.INGESTED)
        self._record_event(
            "ingest_document",
            f"Ingested {doc}",
            path=path,
            high_res=high_res,
            ocr=ocr,
            use_unstructured=use_unstructured,
            extract_entities=extract_entities,
            extract_facts=extract_facts,
        )
        return doc

    async def ingest_file_async(
        self,
        path: str,
        doc_id: str | None = None,
        *,
        config: dict[str, Any] | None = None,
        high_res: bool = False,
        ocr: bool = False,
        use_unstructured: bool | None = None,
        extract_entities: bool = False,
        extract_facts: bool = False,
        client: "LLMClient" | None = None,
        options: "IngestOptions" | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> str:
        """Asynchronous counterpart to :meth:`ingest_file`."""

        from .ingest import ingest_into_dataset_async

        if options is not None:
            config = options.config
            high_res = options.high_res
            ocr = options.ocr
            use_unstructured = options.use_unstructured
            extract_entities = options.extract_entities
            extract_facts = options.extract_facts

        doc = await ingest_into_dataset_async(
            path,
            self,
            doc_id=doc_id,
            config=config,
            high_res=high_res,
            ocr=ocr,
            use_unstructured=use_unstructured,
            extract_entities=extract_entities,
            extract_facts=extract_facts,
            client=client,
            options=None,
            progress_callback=progress_callback,
        )
        self.ingested_docs[doc] = {
            "path": path,
            "config": config,
            "high_res": high_res,
            "ocr": ocr,
            "use_unstructured": use_unstructured,
            "extract_entities": extract_entities,
            "extract_facts": extract_facts,
        }
        self.stage = max(self.stage, DatasetStage.INGESTED)
        self._record_event(
            "ingest_document",
            f"Ingested {doc}",
            path=path,
            high_res=high_res,
            ocr=ocr,
            use_unstructured=use_unstructured,
            extract_entities=extract_entities,
            extract_facts=extract_facts,
        )
        self._persist()
        return doc

    def run_ingestion_layer(  # pragma: no cover
        self, paths: Iterable[str], *, options: "IngestOptions" | None = None
    ) -> List[str]:
        """Ingest many files at once.

        ``paths`` are processed in order and turned into atoms
        :math:`(d, m)` via :meth:`ingest_file`. The list of produced
        document IDs is returned and the operation is logged for
        auditing.
        """

        docs: List[str] = []
        for p in paths:
            docs.append(self.ingest_file(p, options=options))

        # immediately reconcile new nodes and enforce invariants
        self._enforce_policy([1])

        self._record_event(
            "ingestion_layer",
            "Documents ingested",
            count=len(docs),
        )
        return docs

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def get_raw_text(self) -> str:  # pragma: no cover
        """Return text representation of the underlying graph."""

        return self.graph.to_text()

    @persist_after
    def run_post_kg_pipeline(  # pragma: no cover
        self,
        *,
        config_path: Path | None = None,
        provider: str | None = None,
        profile: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        num_pairs: int | None = None,
        threshold: float | None = None,
        fmt: str | None = None,
        overrides: Dict[str, Any] | None = None,
        verbose: bool = False,
        async_mode: bool = False,
        multi_answer: bool = False,
        batch_size: int | None = None,
        inference_batch: int | None = None,
        start_step: PipelineStep | None = None,
        pipeline_config_path: Path | None = None,
        dedup_similarity: float = 1.0,
        keep_ratings: bool = False,
        redis_client: redis.Redis | None = None,
    ) -> Any:
        """Run generation steps after the knowledge graph stage.

        When ``async_mode`` is ``True`` the underlying LLM calls may be
        executed concurrently. ``multi_answer`` forwards to the knowledge graph
        generator to produce several answers per fact. ``pipeline_config_path``
        can override the default pipeline definition. ``dedup_similarity``
        adjusts near-duplicate detection during cleanup and ``keep_ratings``
        controls whether ratings are preserved in the curation result. The
        dataset builder is passed to the pipeline so cleanup events are
        recorded.
        """

        from datacreek.pipelines import (
            run_generation_pipeline,
            run_generation_pipeline_async,
        )

        if redis_client is None:
            redis_client = self.redis_client

        if async_mode:
            result = asyncio.run(
                run_generation_pipeline_async(
                    self.dataset_type,
                    self.graph,
                    dataset_builder=self,
                    config_path=config_path,
                    provider=provider,
                    profile=profile,
                    api_base=api_base,
                    model=model,
                    num_pairs=num_pairs,
                    curation_threshold=threshold,
                    fmt=fmt,
                    overrides=overrides,
                    verbose=verbose,
                    async_mode=True,
                    multi_answer=multi_answer,
                    batch_size=batch_size,
                    inference_batch=inference_batch,
                    start_step=start_step,
                    pipeline_config_path=pipeline_config_path,
                    dedup_similarity=dedup_similarity,
                    keep_ratings=keep_ratings,
                    redis_client=redis_client,
                )
            )
        else:
            result = run_generation_pipeline(
                self.dataset_type,
                self.graph,
                dataset_builder=self,
                config_path=config_path,
                provider=provider,
                profile=profile,
                api_base=api_base,
                model=model,
                num_pairs=num_pairs,
                curation_threshold=threshold,
                fmt=fmt,
                overrides=overrides,
                verbose=verbose,
                async_mode=False,
                multi_answer=multi_answer,
                batch_size=batch_size,
                inference_batch=inference_batch,
                start_step=start_step,
                pipeline_config_path=pipeline_config_path,
                dedup_similarity=dedup_similarity,
                keep_ratings=keep_ratings,
                redis_client=redis_client,
            )

        if self.dataset_type == DatasetType.QA:
            try:
                from datacreek.models.qa import QAPair
                from datacreek.models.results import CurationResult, QAGenerationResult

                if isinstance(result, CurationResult):
                    result.qa_pairs = self.verify_qa_pairs(result.qa_pairs)
                    if result.rated_pairs:
                        result.rated_pairs = self.verify_qa_pairs(result.rated_pairs)
                elif isinstance(result, QAGenerationResult):
                    result.qa_pairs = self.verify_qa_pairs(result.qa_pairs)
                elif isinstance(result, dict) and "qa_pairs" in result:
                    pairs = [
                        QAPair(**p) if isinstance(p, dict) else p
                        for p in result["qa_pairs"]
                    ]
                    verified = self.verify_qa_pairs(pairs)
                    result["qa_pairs"] = [p.to_dict() for p in verified]
            except Exception:
                logger.exception("Failed to verify QA pairs")

        params = {
            "provider": provider,
            "profile": profile,
            "api_base": api_base,
            "model": model,
            "num_pairs": num_pairs,
            "threshold": threshold,
            "fmt": fmt,
            "multi_answer": multi_answer,
            "batch_size": batch_size,
            "inference_batch": inference_batch,
            "start_step": start_step.value if start_step else None,
            "pipeline_config_path": (
                str(pipeline_config_path) if pipeline_config_path else None
            ),
            "dedup_similarity": dedup_similarity,
            "keep_ratings": keep_ratings,
        }

        res_data = (
            asdict(result)
            if is_dataclass(result)
            else result if isinstance(result, dict) else None
        )
        self.versions.append(
            {
                "params": params,
                "time": datetime.now(timezone.utc).isoformat(),
                "result": res_data,
            }
        )
        self.stage = max(self.stage, DatasetStage.GENERATED)
        self._record_event("generate", f"Post-KG pipeline run (v{len(self.versions)})")

        env_limit = os.getenv("DATASET_MAX_VERSIONS")
        if env_limit:
            try:
                limit_val = int(env_limit)
            except Exception:
                limit_val = None
            if limit_val and limit_val > 0:
                try:
                    self.prune_versions(limit_val)
                except Exception:
                    logger.exception("Failed to prune versions for %s", self.name)

        return result

    @persist_after
    def mark_exported(self) -> None:  # pragma: no cover
        """Update the dataset stage and history after export."""

        self.stage = max(self.stage, DatasetStage.EXPORTED)
        self._record_event("export_dataset", "Dataset exported")

    def export_prompts(  # pragma: no cover
        self,
        *,
        auto_fractal: bool = True,
        radii: Iterable[int] = (1, 2, 3),
        max_levels: int = 5,
        mdl_radii: Iterable[int] | None = None,
        ib_beta: float = 1.0,
        encrypt_key: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Return prompt records with fractal and perception metadata.

        When ``encrypt_key`` is provided, author and organization fields are
        encrypted using a simple XOR cipher so that personally identifiable
        information is not exposed in plaintext.

        Each entry includes the MD5 digest of the topological signature
        obtained via :meth:`KnowledgeGraph.topological_signature_hash` so that
        exported prompts can be traced back to a specific graph state.

        Parameters
        ----------
        auto_fractal:
            When ``True`` and chunks lack a ``fractal_level`` attribute the
            method annotates them using :meth:`annotate_mdl_levels` before
            export. This ensures every prompt carries at least a coarse fractal
            position.
        radii:
            Candidate box radii used if ``auto_fractal`` triggers an annotation.
        max_levels:
            Maximum depth for the MDL-guided hierarchy when annotating.
        """

        if auto_fractal and not any(
            "fractal_level" in n[1] for n in self.graph.graph.nodes(data=True)
        ):
            # annotate nodes in-place so the metadata is available for export
            self.annotate_mdl_levels(radii, max_levels=max_levels)

        mdl_radii = tuple(mdl_radii or radii)
        _, counts = self.graph.box_counting_dimension(mdl_radii)
        from ..analysis.fractal import mdl_value

        mdl_gain = mdl_value(counts)

        from ..utils.gitinfo import get_commit_hash

        commit = get_commit_hash()

        signature = self.graph.topological_signature(max_dim=1)
        # call the wrapper so the event is recorded
        sig_hash = self.topological_signature_hash(max_dim=1)
        data: List[Dict[str, Any]] = []
        for node, attrs in self.graph.graph.nodes(data=True):
            if attrs.get("type") != "chunk":
                continue
            prompt_text = attrs.get("text", "")
            docs = [
                u
                for u, v, d in self.graph.graph.in_edges(node, data=True)
                if d.get("relation") == "has_chunk"
            ]
            doc_attrs = self.graph.graph.nodes[docs[0]] if docs else {}
            record = {
                "prompt": prompt_text,
                "fractal_level": attrs.get("fractal_level"),
                "perception_id": attrs.get("perception_id"),
                "topo_signature": signature,
                "signature_hash": sig_hash,
                "prompt_hash": hashlib.md5(
                    prompt_text.encode(), usedforsecurity=False
                ).hexdigest(),  # nosec B324
                "git_commit": commit,
                "mdl_gain": mdl_gain,
                "ib_beta": ib_beta,
                "tag": "inferred",
                "author": doc_attrs.get("author"),
                "organization": doc_attrs.get("organization"),
            }
            if encrypt_key:
                from ..utils import encrypt_pii_fields

                encrypt_pii_fields(record, encrypt_key, ("author", "organization"))
            data.append(record)
        self._record_event(
            "export_prompts",
            "Prompt data exported",
            commit=commit,
            mdl_gain=mdl_gain,
            ib_beta=ib_beta,
            encrypted=bool(encrypt_key),
        )
        try:
            push_metrics({"prompts_exported": float(len(data))})
        except Exception:
            pass
        return data

    @persist_after
    def delete_version(self, index: int) -> None:  # pragma: no cover
        """Remove a stored generation version."""

        if index < 1 or index > len(self.versions):
            raise IndexError("version index out of range")
        removed = self.versions.pop(index - 1)
        self._record_event(
            "delete_version",
            f"Removed version {index}",
            params={"version": removed.get("time")},
        )

    @persist_after
    def restore_version(self, index: int) -> None:  # pragma: no cover
        """Restore ``index`` as the latest generation result."""

        if index < 1 or index > len(self.versions):
            raise IndexError("version index out of range")
        restored = deepcopy(self.versions[index - 1])
        self.versions.append(restored)
        self.stage = max(self.stage, DatasetStage.GENERATED)
        self._record_event(
            "restore_version",
            f"Restored version {index}",
            version=restored.get("time"),
        )

    @persist_after
    def prune_versions(self, limit: int | None = None) -> int:  # pragma: no cover
        """Remove oldest versions beyond ``limit``.

        The default ``limit`` is read from the ``DATASET_MAX_VERSIONS``
        environment variable if not provided. Returns the number of removed
        versions.
        """

        if limit is None:
            env = os.getenv("DATASET_MAX_VERSIONS")
            try:
                limit = int(env) if env else None
            except Exception:
                limit = None
        if not limit or limit < 1:
            return 0
        removed = 0
        while len(self.versions) > limit:
            self.versions.pop(0)
            removed += 1
        if removed:
            self._record_event(
                "prune_versions",
                f"Pruned to {limit} versions",
                removed=removed,
            )
        return removed

    # ------------------------------------------------------------------
    # Neo4j helpers
    # ------------------------------------------------------------------

    @persist_after
    def save_neo4j(self, driver: Driver | None = None) -> None:  # pragma: no cover
        """Persist the knowledge graph to Neo4j."""

        driver = driver or self.neo4j_driver
        if not driver:
            raise ValueError("Neo4j driver required")
        self.graph.to_neo4j(driver, dataset=self.name)
        self._record_event("save_neo4j", "Graph saved to Neo4j")

    @persist_after
    def load_neo4j(self, driver: Driver | None = None) -> None:  # pragma: no cover
        """Load the knowledge graph from Neo4j."""

        driver = driver or self.neo4j_driver
        if not driver:
            raise ValueError("Neo4j driver required")
        self.graph = self.graph.__class__.from_neo4j(driver, dataset=self.name)
        self._record_event("load_neo4j", "Graph loaded from Neo4j")

    # ------------------------------------------------------------------
    # RedisGraph helpers
    # ------------------------------------------------------------------

    @persist_after
    def save_redis_graph(
        self, graph: RGGraph | None = None
    ) -> None:  # pragma: no cover
        """Persist the knowledge graph to RedisGraph."""

        graph = graph or get_redis_graph(self.name)
        if graph is None:
            raise ValueError("RedisGraph not configured")
        try:
            graph.query("MATCH (n {dataset:$ds}) DETACH DELETE n", {"ds": self.name})
        except Exception:
            pass
        for node_id, attrs in self.graph.graph.nodes(data=True):
            label = attrs.get("type", "Node")
            props = {k: v for k, v in attrs.items() if k != "type"}
            props_str = ", ".join(f"{k}: ${k}_{node_id}" for k in props)
            params = {f"{k}_{node_id}": v for k, v in props.items()}
            params.update({"id": node_id, "ds": self.name})
            q = f"CREATE (:{label} {{id:$id, dataset:$ds{', ' + props_str if props_str else ''}}})"
            graph.query(q, params)
        for src, dst, attrs in self.graph.graph.edges(data=True):
            relation = attrs.get("relation", "REL")
            props = {k: v for k, v in attrs.items() if k != "relation"}
            props_str = ", ".join(f"{k}: ${k}_{src}_{dst}" for k in props)
            params = {f"{k}_{src}_{dst}": v for k, v in props.items()}
            params.update({"src": src, "dst": dst, "ds": self.name})
            q = (
                f"MATCH (a {{id:$src, dataset:$ds}}), (b {{id:$dst, dataset:$ds}}) "
                f"CREATE (a)-[:{relation} {{dataset:$ds{', ' + props_str if props_str else ''}}}]->(b)"
            )
            graph.query(q, params)
        self._record_event("save_redis_graph", "Graph saved to RedisGraph")

    @persist_after
    def load_redis_graph(
        self, graph: RGGraph | None = None
    ) -> None:  # pragma: no cover
        """Load the knowledge graph from RedisGraph."""

        graph = graph or get_redis_graph(self.name)
        if graph is None:
            raise ValueError("RedisGraph not configured")
        self.graph.graph.clear()
        res_nodes = graph.query(
            "MATCH (n {dataset:$ds}) RETURN n.id, labels(n)[0], properties(n)",
            {"ds": self.name},
        )
        for nid, label, props in res_nodes.result_set:
            if isinstance(props, dict):
                props.pop("dataset", None)
            self.graph.graph.add_node(nid, type=label, **props)
        res_edges = graph.query(
            "MATCH (a {dataset:$ds})-[r]->(b {dataset:$ds}) RETURN a.id, b.id, type(r), properties(r)",
            {"ds": self.name},
        )
        for src, dst, rel, props in res_edges.result_set:
            if isinstance(props, dict):
                props.pop("dataset", None)
            self.graph.graph.add_edge(src, dst, relation=rel, **props)
        self._record_event("load_redis_graph", "Graph loaded from RedisGraph")

    def delete_redis_graph(
        self, graph: RGGraph | None = None
    ) -> None:  # pragma: no cover
        """Remove the dataset's graph from RedisGraph if configured."""

        graph = graph or get_redis_graph(self.name)
        if graph is None:
            return
        try:
            graph.query("MATCH (n {dataset:$ds}) DETACH DELETE n", {"ds": self.name})
        except Exception:
            logger.exception("Failed to delete graph %s from RedisGraph", self.name)

    def run_orchestrator(  # pragma: no cover
        self,
        llm_call: Callable[[str], str] | None,
        paths: Iterable[str],
        *,
        ingest_options: "IngestOptions" | None = None,
    ) -> str:
        """Run the end-to-end generation pipeline on ``paths``.

        The routine sequentially invokes the ingestion, quality, fractal,
        embedding, topological perception, information, generation,
        compression and export layers. After each stage
        :meth:`_enforce_policy` refreshes the hierarchy and ensures
        invariants remain within bounds. The final Alpaca formatted dataset is returned. This helper is meant for
        ``llm_call`` defaults to :attr:`llm_service` when ``None``. The helper is
        meant for quick prototyping and testing of the full workflow.
        """

        if llm_call is None:
            if self.llm_service is None:
                raise ValueError("llm_call required when no llm_service configured")
            llm_call = self.llm_service

        # start background monitoring while the pipeline runs
        self.start_policy_monitor_thread([1])

        # Ingest documents and build the initial graph
        docs = self.run_ingestion_layer(paths, options=ingest_options)
        # Basic cleanup and fractal hierarchy
        self.run_quality_layer(use_neo4j=False)
        self.run_fractal_layer([1], max_levels=1)
        self._enforce_policy([1])
        # Compute light-weight embeddings
        self.compute_graph_embeddings(dimensions=4, walk_length=4, num_walks=5, seed=0)

        self.run_hypergraph_layer(embed_dim=8, k=5, threshold=0.8)

        self._enforce_policy([1])

        target = self.graph.graph.to_undirected()
        try:
            from ..analysis import fractal as _fr

            if _fr.gd is not None:
                self.run_topological_perception_layer(target, [1], loops=1, epsilon=0.0)
            else:  # pragma: no cover - optional dependency missing
                self._record_event(
                    "topological_perception_layer",
                    "Skipped - gudhi unavailable",
                )
        except Exception:  # pragma: no cover - unexpected failure
            self._record_event(
                "topological_perception_layer",
                "Failed - exception raised",
            )

        self._enforce_policy([1])

        labels = {n: i % 2 for i, n in enumerate(self.graph.graph.nodes)}
        self.run_information_layer(labels, [], beta=1.0)

        self._enforce_policy([1])

        # Generate a single prompt per document
        query = next(iter(docs)) if docs else ""
        self.run_generation_layer(llm_call, query=query, k=1)

        self.run_compression_layer()
        out = self.run_export_layer("alpaca")
        self.mark_exported()

        # stop the background monitor
        self.stop_policy_monitor_thread()

        self._record_event(
            "orchestrator",
            "Full pipeline executed",
            docs=len(docs),
        )
        return out

    async def run_orchestrator_async(
        self,
        llm_call: Callable[[str], Awaitable[str]] | None,
        paths: Iterable[str],
        *,
        ingest_options: "IngestOptions" | None = None,
    ) -> str:
        """Asynchronous variant of :meth:`run_orchestrator`."""

        if llm_call is None:
            if self.llm_service is None:
                raise ValueError("llm_call required when no llm_service configured")

            async def default_call(prompt: str) -> str:
                return (await self.llm_service.acomplete([prompt]))[0]

            llm_call = default_call

        self.start_policy_monitor_thread([1])

        docs = self.run_ingestion_layer(paths, options=ingest_options)
        self.run_quality_layer(use_neo4j=False)
        self.run_fractal_layer([1], max_levels=1)
        self._enforce_policy([1])
        self.compute_graph_embeddings(dimensions=4, walk_length=4, num_walks=5, seed=0)
        self.run_hypergraph_layer(embed_dim=8, k=5, threshold=0.8)
        self._enforce_policy([1])

        target = self.graph.graph.to_undirected()
        try:
            from ..analysis import fractal as _fr

            if _fr.gd is not None:
                self.run_topological_perception_layer(target, [1], loops=1, epsilon=0.0)
            else:  # pragma: no cover - optional dependency missing
                self._record_event(
                    "topological_perception_layer",
                    "Skipped - gudhi unavailable",
                )
        except Exception:  # pragma: no cover - unexpected failure
            self._record_event(
                "topological_perception_layer",
                "Failed - exception raised",
            )

        self._enforce_policy([1])

        labels = {n: i % 2 for i, n in enumerate(self.graph.graph.nodes)}
        self.run_information_layer(labels, [], beta=1.0)

        self._enforce_policy([1])

        query = next(iter(docs)) if docs else ""
        await self.run_generation_layer_async(llm_call, query=query, k=1)

        self.run_compression_layer()
        out = self.run_export_layer("alpaca")
        self.mark_exported()

        self.stop_policy_monitor_thread()

        self._record_event(
            "orchestrator_async",
            "Full async pipeline executed",
            docs=len(docs),
        )
        return out
