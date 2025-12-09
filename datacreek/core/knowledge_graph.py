# pydocstyle: ignore=D100,D101,D102,D103,D107,D401
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # optional dependency for graph operations
    import networkx as nx
except Exception:  # pragma: no cover - minimal stub when networkx missing
    import types

    class _GraphStub:
        def __init__(self, *a, **k) -> None:
            pass

    nx = types.SimpleNamespace(DiGraph=_GraphStub, Graph=_GraphStub)  # type: ignore
try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency missing
    np = None  # type: ignore
try:
    import requests
except Exception:  # pragma: no cover - optional dependency missing
    requests = None  # type: ignore
try:
    from dateutil import parser
except Exception:  # pragma: no cover - optional dependency missing
    parser = None  # type: ignore

try:  # optional dependency
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except Exception:  # pragma: no cover - fallback when watchdog missing
    FileSystemEventHandler = object  # type: ignore[misc]

    class _DummyObserver:  # pragma: no cover - lightweight stub
        def schedule(self, *a, **k):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def join(self, timeout=None):
            pass

    Observer = _DummyObserver  # type: ignore[assignment]

try:
    from ..analysis.index import ann_latency
except Exception:  # pragma: no cover - optional metric
    ann_latency = None

# Base directory for cached artifacts
CACHE_ROOT = os.environ.get("DATACREEK_CACHE", "./cache")

from ..utils.config import load_config

try:
    from neo4j import Driver, GraphDatabase
except Exception:  # pragma: no cover - optional dependency for tests
    Driver = object  # type: ignore
    GraphDatabase = None  # type: ignore

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - optional dependency for tests
    KMeans = None
    cosine_similarity = None
    PCA = None

from ..utils.retrieval import EmbeddingIndex

# ---------------------------------------------------------------------------
# Cleanup config hot-reload support
# ---------------------------------------------------------------------------
_cleanup_cfg: Dict[str, float | int] = {}
_cleanup_lock = threading.Lock()
_observer: Observer | None = None
_cfg_path: Path | None = None


def _load_cleanup() -> None:  # pragma: no cover - heavy
    """Load cleanup parameters from YAML configuration."""
    cfg = load_config()
    cleanup = cfg.get("cleanup", {})
    data = {
        "tau": int(cleanup.get("tau", 5)),
        "sigma": float(cleanup.get("sigma", 0.95)),
        "k_min": int(cleanup.get("k_min", 5)),
        "lp_sigma": float(cleanup.get("lp_sigma", 0.3)),
        "lp_topk": int(cleanup.get("lp_topk", 50)),
        "hub_deg": int(cleanup.get("hub_deg", 1000)),
    }
    with _cleanup_lock:
        _cleanup_cfg.update(data)


def get_cleanup_cfg() -> Dict[str, float | int]:  # pragma: no cover - heavy
    """Return the currently loaded cleanup configuration."""
    with _cleanup_lock:
        return dict(_cleanup_cfg)


@dataclass
class CleanupConfig:
    """Current cleanup thresholds used across the application."""

    tau: int = 5
    sigma: float = 0.95
    k_min: int = 5
    lp_sigma: float = 0.3
    lp_topk: int = 50
    hub_deg: int = 1000


def apply_cleanup_config() -> None:  # pragma: no cover - heavy
    """Update :class:`CleanupConfig` with freshly loaded values."""

    vals = get_cleanup_cfg()
    CleanupConfig.tau = int(vals.get("tau", CleanupConfig.tau))
    CleanupConfig.sigma = float(vals.get("sigma", CleanupConfig.sigma))
    CleanupConfig.k_min = int(vals.get("k_min", CleanupConfig.k_min))
    CleanupConfig.lp_sigma = float(vals.get("lp_sigma", CleanupConfig.lp_sigma))
    CleanupConfig.lp_topk = int(vals.get("lp_topk", CleanupConfig.lp_topk))
    CleanupConfig.hub_deg = int(vals.get("hub_deg", CleanupConfig.hub_deg))
    logging.getLogger(__name__).info(
        "[CFG-HOT] cleanup thresholds updated at %s",
        datetime.now(timezone.utc).isoformat(),
    )


class ConfigReloader(FileSystemEventHandler):
    """Watch a YAML file and reload cleanup thresholds on changes."""

    def __init__(self, cfg_path: Path) -> None:  # pragma: no cover - heavy
        self.cfg_path = Path(cfg_path).resolve()

    def on_modified(
        self, event
    ) -> None:  # pragma: no cover - heavy type: ignore[override]
        if Path(event.src_path).resolve() == self.cfg_path:
            try:
                _load_cleanup()
                apply_cleanup_config()
            except Exception:
                logging.getLogger(__name__).exception("cleanup reload failed")


def verify_thresholds() -> None:  # pragma: no cover - heavy
    """Assert live cleanup thresholds match the YAML configuration.

    This loads the cleanup section from the current YAML config file and
    compares each value against :class:`CleanupConfig`. A mismatch indicates
    the hot-reload watcher failed to update live parameters.
    """

    cfg = load_config()
    expected = cfg.get("cleanup", {})
    for key in ["tau", "sigma", "k_min", "lp_sigma", "lp_topk", "hub_deg"]:
        yaml_val = expected.get(key)
        if yaml_val is None:
            continue
        curr_val = getattr(CleanupConfig, key)
        if curr_val != yaml_val:
            raise RuntimeError(f"CFG-HOT mismatch: {key}\u2260{yaml_val}")


def start_cleanup_watcher(  # pragma: no cover - heavy
    cfg_path: str | os.PathLike | None = None, interval: float = 300.0
) -> None:  # pragma: no cover - heavy
    """Start filesystem watcher reloading cleanup thresholds.

    Parameters
    ----------
    cfg_path:
        Optional path to the YAML configuration. When ``None`` the value of
        ``DATACREEK_CONFIG`` is used and defaults to ``configs/default.yaml``.
    interval:
        Polling interval for the :class:`watchdog.observers.Observer`.
    """

    global _observer, _cfg_path
    if _observer is not None:
        return

    env_path = os.environ.get("DATACREEK_CONFIG", "configs/default.yaml")
    cfg_file = Path(cfg_path or env_path)
    _cfg_path = cfg_file.resolve()
    _load_cleanup()
    apply_cleanup_config()
    handler = ConfigReloader(_cfg_path)
    observer = Observer()
    observer.schedule(handler, str(_cfg_path.parent), recursive=False)
    observer.daemon = True
    observer.start()
    logging.getLogger(__name__).info("CFG-HOT watcher started")
    _observer = observer


def stop_cleanup_watcher() -> None:  # pragma: no cover - heavy
    """Stop the cleanup configuration watcher."""

    global _observer
    if _observer is not None:
        _observer.stop()
        _observer.join(timeout=0.5)
        _observer = None


@dataclass
class KnowledgeGraph:
    """Simple wrapper storing documents and chunks with source info."""

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    use_hnsw: bool = False
    index: EmbeddingIndex = field(init=False)
    faiss_index: object | None = field(init=False, default=None, repr=False)
    faiss_ids: list[str] | None = field(init=False, default=None, repr=False)
    faiss_index_type: str | None = field(init=False, default=None, repr=False)
    faiss_node_attr: str | None = field(init=False, default=None, repr=False)
    _mapper_cache: Dict[int, tuple[nx.Graph, list[set[str]]]] = field(
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

    def add_document(  # pragma: no cover - heavy
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
        """Insert a document node in the graph."""
        if self.graph.has_node(doc_id):
            raise ValueError(f"Document already exists: {doc_id}")
        self.graph.add_node(
            doc_id,
            type="document",
            source=source,
            author=author,
            organization=organization,
            checksum=checksum,
            uid=uid,
        )
        if text:
            self.graph.nodes[doc_id]["text"] = text
            self.index.add(doc_id, text)

    def add_entity(
        self, entity_id: str, text: str, source: str | None = None
    ) -> None:  # pragma: no cover - heavy
        """Insert an entity node."""

        if self.graph.has_node(entity_id):
            raise ValueError(f"Entity already exists: {entity_id}")

        try:
            from ..utils.text import detect_language

            lang = detect_language(text)
        except Exception:  # pragma: no cover - optional dependency
            lang = "und"

        self.graph.add_node(
            entity_id, type="entity", text=text, source=source, lang=lang
        )
        if text:
            self.index.add(entity_id, text)

    def add_fact(  # pragma: no cover - heavy
        self,
        subject: str,
        predicate: str,
        obj: str,
        fact_id: Optional[str] = None,
        *,
        source: Optional[str] = None,
    ) -> str:
        """Insert a standalone fact and link corresponding entities."""

        fact_id = fact_id or f"fact_{len(self.graph.nodes)}"
        if self.graph.has_node(fact_id):
            raise ValueError(f"Fact already exists: {fact_id}")

        # ensure entity nodes exist
        if not self.graph.has_node(subject):
            self.add_entity(subject, subject, source)
        if not self.graph.has_node(obj):
            self.add_entity(obj, obj, source)

        self.graph.add_node(
            fact_id,
            type="fact",
            subject=subject,
            predicate=predicate,
            object=obj,
            source=source,
        )
        self.graph.add_edge(fact_id, subject, relation="subject", provenance=source)
        self.graph.add_edge(fact_id, obj, relation="object", provenance=source)
        self.graph.add_edge(subject, obj, relation=predicate, provenance=source)
        self.index.add(fact_id, f"{subject} {predicate} {obj}")
        return fact_id

    def link_entity(  # pragma: no cover - heavy
        self,
        node_id: str,
        entity_id: str,
        relation: str = "mentions",
        *,
        provenance: str | None = None,
    ) -> None:
        """Create a relation between ``node_id`` and ``entity_id``.

        Parameters
        ----------
        node_id: str
            ID of the source node (chunk or document).
        entity_id: str
            ID of the entity node.
        relation: str, optional
            Label for the relation. Defaults to ``"mentions"``.
        provenance: str, optional
            Provenance for the relation. If omitted, ``node_id``'s ``source`` is
            used when available.
        """

        if not self.graph.has_node(node_id) or not self.graph.has_node(entity_id):
            raise ValueError("Unknown node")
        if provenance is None:
            provenance = self.graph.nodes[node_id].get("source")
        self.graph.add_edge(
            node_id, entity_id, relation=relation, provenance=provenance
        )

    def link_transcript(  # pragma: no cover - heavy
        self,
        chunk_id: str,
        audio_id: str,
        *,
        provenance: str | None = None,
    ) -> None:
        """Connect a chunk to its audio with a ``transcript_of`` relation."""

        if not self.graph.has_node(chunk_id) or not self.graph.has_node(audio_id):
            raise ValueError("Unknown node")
        if provenance is None:
            provenance = self.graph.nodes[chunk_id].get("source")
        self.graph.add_edge(
            chunk_id, audio_id, relation="transcript_of", provenance=provenance
        )

    def add_section(  # pragma: no cover - heavy
        self,
        doc_id: str,
        section_id: str,
        title: str | None = None,
        source: Optional[str] = None,
        *,
        page: int | None = None,
    ) -> None:
        """Insert a section node and attach it to ``doc_id``."""

        if source is None:
            source = self.graph.nodes[doc_id].get("source")
        if self.graph.has_node(section_id):
            raise ValueError(f"Section already exists: {section_id}")

        existing = self.get_sections_for_document(doc_id)
        sequence = len(existing)

        self.graph.add_node(
            section_id,
            type="section",
            title=title,
            source=source,
            page=page,
        )
        if title:
            self.index.add(section_id, title)
        self.graph.add_edge(
            doc_id,
            section_id,
            relation="has_section",
            sequence=sequence,
            provenance=source,
        )

        if existing:
            prev = existing[-1]
            self.graph.add_edge(prev, section_id, relation="next_section")

    def add_chunk(  # pragma: no cover - heavy
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
        """Insert a chunk node linked to ``doc_id``."""
        if source is None:
            source = self.graph.nodes[doc_id].get("source")
        if self.graph.has_node(chunk_id):
            raise ValueError(f"Chunk already exists: {chunk_id}")

        # Determine the sequence index within the document and optional section
        doc_chunks = self.get_chunks_for_document(doc_id)
        doc_sequence = len(doc_chunks)
        if section_id and self.graph.has_node(section_id):
            existing_chunks = self.get_chunks_for_section(section_id)
            section_sequence = len(existing_chunks)
        else:
            existing_chunks = doc_chunks
            section_sequence = None

        # Add the chunk node and relation to the document
        if page is None:
            page = 1

        self.graph.add_node(
            chunk_id,
            type="chunk",
            text=text,
            source=source,
            page=page,
            overlap=chunk_overlap,
        )
        if emotion:
            self.graph.nodes[chunk_id]["emotion"] = emotion
        if modality:
            self.graph.nodes[chunk_id]["modality"] = modality
        if entities:
            self.graph.nodes[chunk_id]["entities"] = entities
        self.graph.add_edge(
            doc_id,
            chunk_id,
            relation="has_chunk",
            sequence=doc_sequence,
            provenance=source,
        )
        if section_id and self.graph.has_node(section_id):
            self.graph.add_edge(
                section_id,
                chunk_id,
                relation="under_section",
                sequence=section_sequence,
                provenance=source,
            )
            if self.graph.nodes[section_id].get("page") is None and page is not None:
                self.graph.nodes[section_id]["page"] = page

        # Connect to the previous chunk to keep the original order
        if doc_chunks:
            prev_chunk = doc_chunks[-1]
            self.graph.add_edge(prev_chunk, chunk_id, relation="next_chunk")

        self.index.add(chunk_id, text)

    def add_image(  # pragma: no cover - heavy
        self,
        doc_id: str,
        image_id: str,
        path: str,
        source: Optional[str] = None,
        *,
        page: int | None = None,
        alt_text: str | None = None,
    ) -> None:
        """Insert an image node linked to ``doc_id``."""

        if source is None:
            source = self.graph.nodes[doc_id].get("source")
        if self.graph.has_node(image_id):
            raise ValueError(f"Image already exists: {image_id}")

        doc_images = self.get_images_for_document(doc_id)
        sequence = len(doc_images)
        if page is None:
            page = 1

        self.graph.add_node(
            image_id,
            type="image",
            path=path,
            source=source,
            page=page,
        )
        if alt_text:
            self.graph.nodes[image_id]["alt_text"] = alt_text
            caption_id = f"{image_id}_caption"
            if not self.graph.has_node(caption_id):
                self.graph.add_node(
                    caption_id,
                    type="caption",
                    text=alt_text,
                    source=source,
                    page=page,
                )
                captions = self.get_captions_for_document(doc_id)
                self.graph.add_edge(
                    doc_id,
                    caption_id,
                    relation="has_caption",
                    sequence=len(captions),
                    provenance=source,
                )
            self.graph.add_edge(
                caption_id,
                image_id,
                relation="caption_of",
                provenance=source,
            )
        self.graph.add_edge(
            doc_id,
            image_id,
            relation="has_image",
            sequence=sequence,
            provenance=source,
        )

    def add_audio(  # pragma: no cover - heavy
        self,
        doc_id: str,
        audio_id: str,
        path: str,
        source: str | None = None,
        *,
        page: int | None = None,
        lang: str | None = None,
    ) -> None:
        """Insert an audio node linked to ``doc_id``."""

        if source is None:
            source = self.graph.nodes[doc_id].get("source")
        if self.graph.has_node(audio_id):
            raise ValueError(f"Audio already exists: {audio_id}")

        audios = self.get_audios_for_document(doc_id)
        sequence = len(audios)
        if page is None:
            page = 1

        self.graph.add_node(
            audio_id,
            type="audio",
            path=path,
            source=source,
            page=page,
            **({"lang": lang} if lang else {}),
        )
        self.graph.add_edge(
            doc_id,
            audio_id,
            relation="has_audio",
            sequence=sequence,
            provenance=source,
        )

    # ------------------------------------------------------------------
    # Atom/molecule hierarchy
    # ------------------------------------------------------------------

    def add_atom(  # pragma: no cover - heavy
        self,
        doc_id: str,
        atom_id: str,
        text: str,
        element_type: str,
        source: str | None = None,
        *,
        page: int | None = None,
        lang: str | None = None,
        timestamp: datetime | None = None,
        emotion: str | None = None,
        modality: str | None = None,
        entities: list[str] | None = None,
    ) -> None:
        """Insert an atom node linked to ``doc_id``."""

        if source is None:
            source = self.graph.nodes[doc_id].get("source")
        if self.graph.has_node(atom_id):
            raise ValueError(f"Atom already exists: {atom_id}")

        atoms = self.get_atoms_for_document(doc_id)
        sequence = len(atoms)
        if page is None:
            page = 1

        if lang is None:
            try:
                from ..utils.text import detect_language

                lang = detect_language(text)
            except Exception:  # pragma: no cover - optional dependency
                lang = "und"

        self.graph.add_node(
            atom_id,
            type="atom",
            text=text,
            element_type=element_type,
            source=source,
            page=page,
            lang=lang,
            timestamp=(timestamp or datetime.now(timezone.utc)).isoformat(),
        )
        if emotion:
            self.graph.nodes[atom_id]["emotion"] = emotion
        if modality:
            self.graph.nodes[atom_id]["modality"] = modality
        if entities:
            self.graph.nodes[atom_id]["entities"] = entities
        self.graph.add_edge(
            doc_id,
            atom_id,
            relation="has_atom",
            sequence=sequence,
            provenance=source,
        )
        if atoms:
            prev = atoms[-1]
            self.graph.add_edge(prev, atom_id, relation="next_atom")

    def add_molecule(  # pragma: no cover - heavy
        self,
        doc_id: str,
        molecule_id: str,
        atom_ids: Iterable[str],
        source: str | None = None,
    ) -> None:
        """Insert a molecule node composed of ``atom_ids``."""

        if source is None:
            source = self.graph.nodes[doc_id].get("source")
        if self.graph.has_node(molecule_id):
            raise ValueError(f"Molecule already exists: {molecule_id}")

        molecules = self.get_molecules_for_document(doc_id)
        sequence = len(molecules)

        self.graph.add_node(molecule_id, type="molecule", source=source)
        self.graph.add_edge(
            doc_id,
            molecule_id,
            relation="has_molecule",
            sequence=sequence,
            provenance=source,
        )
        if molecules:
            prev = molecules[-1]
            self.graph.add_edge(prev, molecule_id, relation="next_molecule")
        for idx, aid in enumerate(atom_ids):
            self.graph.add_edge(
                molecule_id,
                aid,
                relation="inside",
                sequence=idx,
                provenance=source,
            )

    def add_hyperedge(  # pragma: no cover - heavy
        self,
        edge_id: str,
        node_ids: Iterable[str],
        *,
        relation: str = "member",
        source: str | None = None,
    ) -> None:
        """Insert a hyperedge connecting ``node_ids``."""

        if self.graph.has_node(edge_id):
            raise ValueError(f"Hyperedge already exists: {edge_id}")

        self.graph.add_node(edge_id, type="hyperedge", source=source)

        for idx, nid in enumerate(node_ids):
            if not self.graph.has_node(nid):
                raise ValueError(f"Unknown node: {nid}")
            self.graph.add_edge(
                edge_id,
                nid,
                relation=relation,
                sequence=idx,
                provenance=source,
            )

    def add_simplex(  # pragma: no cover - heavy
        self,
        simplex_id: str,
        node_ids: Iterable[str],
        *,
        source: str | None = None,
    ) -> None:
        """Insert a simplex node linked to ``node_ids``.

        Parameters
        ----------
        simplex_id:
            Identifier for the simplex node.
        node_ids:
            Vertices forming this simplex.
        source:
            Optional provenance information.
        """

        if self.graph.has_node(simplex_id):
            raise ValueError(f"Simplex already exists: {simplex_id}")

        dim = len(list(node_ids)) - 1
        self.graph.add_node(simplex_id, type="simplex", dimension=dim, source=source)

        for idx, nid in enumerate(node_ids):
            if not self.graph.has_node(nid):
                raise ValueError(f"Unknown node: {nid}")
            self.graph.add_edge(
                simplex_id,
                nid,
                relation="face",
                sequence=idx,
                provenance=source,
            )

    def _renumber_chunks(self, doc_id: str) -> None:  # pragma: no cover - heavy
        """Update sequence numbers and next_chunk links for ``doc_id``."""
        chunks = self.get_chunks_for_document(doc_id)
        # Update sequence numbers
        for i, cid in enumerate(chunks):
            if (doc_id, cid) in self.graph.edges:
                self.graph.edges[doc_id, cid]["sequence"] = i

        # Remove existing next_chunk edges for this document
        for cid in chunks:
            for succ in list(self.graph.successors(cid)):
                if self.graph.edges[cid, succ].get("relation") == "next_chunk":
                    self.graph.remove_edge(cid, succ)

        # Recreate next_chunk edges
        for a, b in zip(chunks, chunks[1:]):
            self.graph.add_edge(a, b, relation="next_chunk")

    def remove_chunk(self, chunk_id: str) -> None:  # pragma: no cover - heavy
        """Delete ``chunk_id`` from the graph and index."""
        if not self.graph.has_node(chunk_id):
            return
        preds = [
            p
            for p in self.graph.predecessors(chunk_id)
            if self.graph.edges[p, chunk_id].get("relation") == "has_chunk"
        ]
        doc_id = preds[0] if preds else None
        self.graph.remove_node(chunk_id)
        self.index.remove(chunk_id)
        if doc_id:
            self._renumber_chunks(doc_id)

    def remove_document(self, doc_id: str) -> None:  # pragma: no cover - heavy
        """Remove a document and all its chunks."""
        if not self.graph.has_node(doc_id):
            return
        chunks = self.get_chunks_for_document(doc_id)
        for cid in chunks:
            self.index.remove(cid)
        self.graph.remove_nodes_from(chunks)
        if self.graph.has_node(doc_id):
            self.graph.remove_node(doc_id)
        if chunks:
            self.index.build()

    def cascade_delete_document(
        self, doc_id: str, driver: object | None = None
    ) -> None:  # pragma: no cover - heavy
        """Remove ``doc_id`` from the graph, Neo4j and FAISS with tombstone."""

        vector = None
        if self.graph.has_node(doc_id):
            vector = self.graph.nodes[doc_id].get(self.faiss_node_attr)
        self.remove_document(doc_id)
        if driver is not None:
            try:
                with driver.session() as session:
                    session.run(
                        "MATCH (d:Doc {uid:$uid}) DETACH DELETE d",
                        uid=doc_id,
                    )
            except Exception:
                pass

        if (
            vector is not None
            and self.faiss_index is not None
            and self.faiss_ids is not None
        ):
            try:
                import numpy as np

                idx = self.faiss_ids.index(doc_id)
                self.faiss_index.remove_ids(np.asarray([idx], dtype=np.int64))
                self.faiss_ids.pop(idx)
                self.faiss_index.add(np.asarray([vector], dtype=np.float32))
                self.faiss_ids.append(f"doc:deleted:{doc_id}")
            except Exception:
                # index type may not support removal; invalidate instead
                self.faiss_index = None
                self.faiss_ids = None

    def search(
        self, query: str, node_type: str = "chunk"
    ) -> list[str]:  # pragma: no cover - heavy
        """Return node IDs of the given type matching the query.

        For chunks we search in the ``text`` attribute while documents are
        matched against their id or ``source``.
        """

        query_lower = query.lower()
        results: list[str] = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") != node_type:
                continue
            if node_type == "document":
                if (
                    query_lower in node.lower()
                    or query_lower in str(data.get("source", "")).lower()
                ):
                    results.append(node)
            elif node_type == "fact":
                if (
                    query_lower in str(data.get("subject", "")).lower()
                    or query_lower in str(data.get("predicate", "")).lower()
                    or query_lower in str(data.get("object", "")).lower()
                ):
                    results.append(node)
            elif node_type == "section":
                if (
                    query_lower in node.lower()
                    or query_lower in str(data.get("title", "")).lower()
                ):
                    results.append(node)
            else:
                if query_lower in str(data.get("text", "")).lower():
                    results.append(node)
        return results

    def search_chunks(self, query: str) -> list[str]:  # pragma: no cover - heavy
        """Return chunk IDs containing the query string."""

        return self.search(query, node_type="chunk")

    def search_documents(self, query: str) -> list[str]:  # pragma: no cover - heavy
        """Return document IDs whose id or source matches the query."""

        return self.search(query, node_type="document")

    def search_embeddings(  # pragma: no cover - heavy
        self,
        query: str,
        k: int = 3,
        fetch_neighbors: bool = True,
        *,
        node_type: str = "chunk",
    ) -> list[str]:
        """Return node IDs of ``node_type`` relevant to ``query`` using embeddings."""

        # retrieve a larger candidate set and then filter by node type
        indices = self.index.search(query, max(k * 4, k))
        ids: List[str] = []
        for idx in indices:
            nid = self.index.get_id(idx)
            if self.graph.nodes[nid].get("type") != node_type:
                continue
            ids.append(nid)
            if fetch_neighbors and node_type == "chunk":
                # add previous and next chunks from same document if available
                preds = list(self.graph.predecessors(nid))
                if preds:
                    doc = preds[0]
                    doc_chunks = self.get_chunks_for_document(doc)
                    pos = doc_chunks.index(nid)
                    if pos > 0:
                        ids.append(doc_chunks[pos - 1])
                    if pos < len(doc_chunks) - 1:
                        ids.append(doc_chunks[pos + 1])
            if len(ids) >= k:
                break

        # remove duplicates while preserving order
        seen = set()
        result = []
        for n in ids:
            if n not in seen:
                seen.add(n)
                result.append(n)
        return result

    def search_hybrid(  # pragma: no cover - heavy
        self, query: str, k: int = 5, *, node_type: str = "chunk"
    ) -> list[str]:
        """Return node IDs by combining lexical and embedding search.

        Results from plain text search are returned first followed by
        semantic matches from the embedding index. Duplicate IDs are
        removed while preserving order. Only the top ``k`` items are
        returned. The ``node_type`` parameter restricts results to nodes
        of the given type.
        """

        lexical_matches = self.search(query, node_type=node_type)
        embedding_ids = self.search_embeddings(
            query, k=k, fetch_neighbors=False, node_type=node_type
        )

        seen = set()
        results: List[str] = []
        for cid in lexical_matches + embedding_ids:
            if cid not in seen:
                seen.add(cid)
                results.append(cid)
            if len(results) >= k:
                break
        return results

    def hybrid_score(  # pragma: no cover - heavy
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
        r"""Return the multi-view similarity between ``src`` and ``tgt``.

        The score mixes cosine similarity in Node2Vec space, Poincar\xe9
        distance and GraphWave similarity. It is computed as

        .. math::

            S = \gamma \cos(\text{n2v}) + \eta (1 - d_{\mathbb{B}}) +
            (1-\gamma-\eta) (1 - \cos(\text{gw}))

        where :math:`d_{\mathbb{B}}` is the Poincar\xe9 distance. ``gamma`` and
        ``eta`` control the influence of the Node2Vec and Poincar\xe9 terms.

        Parameters
        ----------
        src, tgt:
            Node identifiers whose embeddings will be compared.
        n2v_attr, gw_attr, hyper_attr:
            Node properties storing the Node2Vec, GraphWave and Poincar\xe9
            embeddings.
        gamma, eta:
            Weights of the Node2Vec and hyperbolic terms in the score.
        """

        from ..analysis import hybrid_score as _hs

        a = self.graph.nodes[src].get(n2v_attr)
        b = self.graph.nodes[tgt].get(n2v_attr)
        gw_a = self.graph.nodes[src].get(gw_attr)
        gw_b = self.graph.nodes[tgt].get(gw_attr)
        hyp_a = self.graph.nodes[src].get(hyper_attr)
        hyp_b = self.graph.nodes[tgt].get(hyper_attr)
        if (
            a is None
            or b is None
            or gw_a is None
            or gw_b is None
            or hyp_a is None
            or hyp_b is None
        ):
            return 0.0
        return _hs(a, b, gw_a, gw_b, hyp_a, hyp_b, gamma=gamma, eta=eta)

    def similar_by_hybrid(  # pragma: no cover - heavy
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
    ) -> List[tuple[str, float]]:
        """Return nodes ranked by :meth:`hybrid_score` with ``node_id``."""

        scores: List[tuple[str, float]] = []
        for n, data in self.graph.nodes(data=True):
            if n == node_id or data.get("type") != node_type:
                continue
            s = self.hybrid_score(
                node_id,
                n,
                n2v_attr=n2v_attr,
                gw_attr=gw_attr,
                hyper_attr=hyper_attr,
                gamma=gamma,
                eta=eta,
            )
            scores.append((n, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def ann_hybrid_search(  # pragma: no cover - heavy
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
        r"""Return top ``k`` nodes by the multi-view similarity.

        A FAISS index built on ``n2v_attr`` retrieves ``ann_k`` candidates.
        Each candidate is then scored via

        .. math::

            S = \gamma \cos(\text{n2v}) + \eta (1 - d_{\mathbb{B}}) +
            (1-\gamma-\eta)(1 - \cos(\text{gw}))

        where the query vectors (``q_n2v``, ``q_gw``, ``q_hyp``) are compared to
        the stored embeddings. ``gamma`` and ``eta`` control the contribution of
        the Euclidean and hyperbolic terms.
        """

        if self.faiss_index is None:
            raise RuntimeError("FAISS index not built")

        candidates = self.search_faiss(list(q_n2v), k=ann_k)
        results: List[Tuple[str, float]] = []
        for cid in candidates:
            data = self.graph.nodes[cid]
            if data.get("type") != node_type:
                continue
            vec_n2v = data.get(n2v_attr)
            vec_gw = data.get(gw_attr)
            vec_hyp = data.get(hyper_attr)
            if vec_n2v is None or vec_gw is None or vec_hyp is None:
                continue
            from ..analysis import hybrid_score as _hs

            s = _hs(q_n2v, vec_n2v, q_gw, vec_gw, q_hyp, vec_hyp, gamma=gamma, eta=eta)
            results.append((cid, s))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def cypher_ann_query(  # pragma: no cover - heavy
        self,
        driver: Driver,
        query: str,
        cypher: str,
        *,
        k: int = 5,
        node_type: str = "chunk",
    ) -> List[Dict[str, Any]]:
        """Return Cypher query results seeded by ANN search.

        Parameters
        ----------
        driver:
            Neo4j driver used to execute the query.
        query:
            Text input searched via the FAISS index.
        cypher:
            Cypher statement expecting an ``ids`` parameter.
        k:
            Number of ANN candidates fetched from ``query``.
        node_type:
            Restrict matches to nodes with this ``type``.

        The query text ``query`` is first used to retrieve up to ``k`` node
        identifiers via :meth:`search_embeddings`. Those identifiers are passed
        as ``ids`` to the provided Cypher statement, allowing complex graph
        queries on a narrow candidate set.
        """

        ids = self.search_embeddings(
            query, k=k, fetch_neighbors=False, node_type=node_type
        )

        if not ids:
            return []

        with driver.session() as session:
            records = session.run(cypher, ids=ids)
        return [dict(r) for r in records]

    def recall_at_k(  # pragma: no cover - heavy
        self,
        queries: Sequence[str],
        ground_truth: Dict[str, Sequence[str]],
        *,
        k: int = 10,
        gamma: float = 0.5,
        eta: float = 0.25,
    ) -> float:
        """Return mean recall@k using hybrid similarity search.

        Parameters
        ----------
        queries:
            List of node identifiers acting as queries.
        ground_truth:
            Mapping from query to relevant node ids.
        k:
            Rank cutoff.
        gamma, eta:
            Weights for Node2Vec and hyperbolic terms.
        """

        from ..analysis.autotune import recall_at_k as _recall

        score = _recall(
            self.graph,
            queries,
            ground_truth,
            k=k,
            gamma=gamma,
            eta=eta,
        )
        if k == 10:
            self.graph.graph["recall10"] = score
        return score

    def search_with_links(  # pragma: no cover - heavy
        self,
        query: str,
        k: int = 5,
        hops: int = 1,
        *,
        fractal_level: int | None = None,
    ) -> list[str]:
        """Return chunk IDs related to a query and expand via graph links.

        Parameters
        ----------
        query:
            Text to search for.
        k:
            Number of initial matches to retrieve via :meth:`search_hybrid`.
        hops:
            How many hops to traverse through ``next_chunk`` or ``similar_to``
            relations starting from the initial results.
        """

        seeds = self.search_hybrid(query, k)
        if fractal_level is not None:
            seeds = [
                s
                for s in seeds
                if self.graph.nodes[s].get("fractal_level", 0) <= fractal_level
            ]
        seen = set(seeds)
        results = list(seeds)
        queue = [(cid, 0) for cid in seeds]

        while queue:
            node, depth = queue.pop(0)
            if depth >= hops:
                continue
            # traverse both successors and predecessors so that similar chunks
            # are discovered regardless of edge direction
            for neighbor in list(self.graph.successors(node)) + list(
                self.graph.predecessors(node)
            ):
                rel = self.graph.edges.get((node, neighbor)) or self.graph.edges.get(
                    (neighbor, node)
                )
                if not rel:
                    continue
                if rel.get("relation") not in {"next_chunk", "similar_to"}:
                    continue
                if neighbor in seen:
                    continue
                if (
                    fractal_level is not None
                    and self.graph.nodes[neighbor].get("fractal_level", 0)
                    > fractal_level
                ):
                    continue
                seen.add(neighbor)
                results.append(neighbor)
                queue.append((neighbor, depth + 1))

        return results

    def search_with_links_data(  # pragma: no cover - heavy
        self,
        query: str,
        k: int = 5,
        hops: int = 1,
        *,
        fractal_level: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Return enriched search results expanding through graph links.

        Each item contains the chunk ``id``, its ``text``, owning ``document``
        and ``source`` plus the hop ``depth`` and the ``path`` from the initial
        result used to reach it.
        """

        seeds = self.search_hybrid(query, k)
        if fractal_level is not None:
            seeds = [
                s
                for s in seeds
                if self.graph.nodes[s].get("fractal_level", 0) <= fractal_level
            ]
        seen = set(seeds)
        queue: List[tuple[str, int, List[str]]] = [(cid, 0, [cid]) for cid in seeds]
        results: List[tuple[str, int, List[str]]] = queue.copy()

        while queue:
            node, depth, path = queue.pop(0)
            if depth >= hops:
                continue
            for nb in list(self.graph.successors(node)) + list(
                self.graph.predecessors(node)
            ):
                rel = self.graph.edges.get((node, nb)) or self.graph.edges.get(
                    (nb, node)
                )
                if not rel or rel.get("relation") not in {"next_chunk", "similar_to"}:
                    continue
                if nb in seen:
                    continue
                if (
                    fractal_level is not None
                    and self.graph.nodes[nb].get("fractal_level", 0) > fractal_level
                ):
                    continue
                seen.add(nb)
                new_path = path + [nb]
                results.append((nb, depth + 1, new_path))
                queue.append((nb, depth + 1, new_path))

        out: List[Dict[str, Any]] = []
        for cid, depth, path in results:
            node = self.graph.nodes[cid]
            doc_id = None
            for pred in self.graph.predecessors(cid):
                if self.graph.edges[pred, cid].get("relation") == "has_chunk":
                    doc_id = pred
                    break
            out.append(
                {
                    "id": cid,
                    "text": node.get("text"),
                    "document": doc_id,
                    "source": node.get("source"),
                    "depth": depth,
                    "path": path,
                }
            )
        return out

    def chunks_by_emotion(self, emotion: str) -> list[str]:  # pragma: no cover - heavy
        """Return chunk IDs tagged with ``emotion``."""

        return [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("type") == "chunk" and d.get("emotion") == emotion
        ]

    def chunks_by_modality(
        self, modality: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return chunk IDs tagged with ``modality``."""

        return [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("type") == "chunk" and d.get("modality") == modality
        ]

    def explain_node(  # pragma: no cover - heavy
        self, node_id: str, hops: int = 3, *, node_attr: str = "embedding"
    ) -> Dict[str, Any]:
        """Return a k-hop subgraph around ``node_id`` with attention heatmap.

        Parameters
        ----------
        node_id:
            Identifier of the node to explain.
        hops:
            Number of hops to traverse when collecting neighbors.
        node_attr:
            Node attribute storing embeddings used for attention scores.

        Returns
        -------
        dict
            Dictionary containing ``nodes`` and ``edges`` lists along with
            an ``attention`` mapping ``"u->v"`` to a score.
        """

        if not self.graph.has_node(node_id):
            raise ValueError(f"Unknown node: {node_id}")

        import numpy as np

        nodes = {node_id}
        frontier = {node_id}
        for _ in range(hops):
            nb: set[str] = set()
            for n in frontier:
                nb.update(self.graph.successors(n))
                nb.update(self.graph.predecessors(n))
            nb.difference_update(nodes)
            nodes.update(nb)
            frontier = nb

        sub = self.graph.subgraph(nodes)

        index: Dict[str, int] = {}
        features: list[np.ndarray] = []
        for n in sub.nodes:
            emb = self.graph.nodes[n].get(node_attr)
            if emb is not None:
                index[n] = len(features)
                features.append(np.asarray(emb, dtype=float))

        edges: list[tuple[str, str]] = []
        pairs: list[list[int]] = []
        for u, v in sub.edges():
            if u in index and v in index:
                edges.append((u, v))
                pairs.append([index[u], index[v]])

        attention: Dict[str, float] = {}
        if pairs and features:
            from ..analysis.hypergraph import hyperedge_attention_scores as _att

            scores = _att(pairs, np.stack(features))
            for (u, v), s in zip(edges, scores):
                attention[f"{u}->{v}"] = float(s)

        return {
            "nodes": list(sub.nodes),
            "edges": list(sub.edges),
            "attention": attention,
        }

    def link_similar_chunks(self, k: int = 3) -> None:  # pragma: no cover - heavy
        """Add ``similar_to`` edges between semantically close chunks."""

        neighbors = self.index.nearest_neighbors(k, return_distances=True)
        for src, nb_list in neighbors.items():
            if self.graph.nodes[src].get("type") != "chunk":
                continue
            for tgt, score in nb_list:
                if src == tgt or self.graph.nodes[tgt].get("type") != "chunk":
                    continue
                if (
                    self.graph.has_edge(src, tgt)
                    and self.graph.edges[src, tgt].get("relation") == "similar_to"
                ):
                    continue
                self.graph.add_edge(src, tgt, relation="similar_to", similarity=score)

    def link_similar_sections(self, k: int = 3) -> None:  # pragma: no cover - heavy
        """Add ``similar_to`` edges between section titles."""

        neighbors = self.index.nearest_neighbors(k, return_distances=True)
        for src, nb_list in neighbors.items():
            if self.graph.nodes[src].get("type") != "section":
                continue
            for tgt, score in nb_list:
                if src == tgt or self.graph.nodes[tgt].get("type") != "section":
                    continue
                if (
                    self.graph.has_edge(src, tgt)
                    and self.graph.edges[src, tgt].get("relation") == "similar_to"
                ):
                    continue
                self.graph.add_edge(src, tgt, relation="similar_to", similarity=score)

    def link_similar_documents(self, k: int = 3) -> None:  # pragma: no cover - heavy
        """Add ``similar_to`` edges between document texts."""

        neighbors = self.index.nearest_neighbors(k, return_distances=True)
        for src, nb_list in neighbors.items():
            if self.graph.nodes[src].get("type") != "document":
                continue
            for tgt, score in nb_list:
                if src == tgt or self.graph.nodes[tgt].get("type") != "document":
                    continue
                if (
                    self.graph.has_edge(src, tgt)
                    and self.graph.edges[src, tgt].get("relation") == "similar_to"
                ):
                    continue
                self.graph.add_edge(src, tgt, relation="similar_to", similarity=score)

    def deduplicate_chunks(
        self, similarity: float = 1.0
    ) -> int:  # pragma: no cover - heavy
        """Remove duplicate chunk nodes based on text similarity."""

        from difflib import SequenceMatcher

        seen: Dict[str, str] = {}
        removed = 0
        for node, data in list(self.graph.nodes(data=True)):
            if data.get("type") != "chunk":
                continue
            text = data.get("text", "")
            norm = re.sub(r"\s+", " ", text.strip().lower())
            found = False
            for stext, sid in seen.items():
                if similarity >= 1.0:
                    if norm == stext:
                        self.remove_chunk(node)
                        removed += 1
                        found = True
                        break
                else:
                    if SequenceMatcher(None, norm, stext).ratio() >= similarity:
                        self.remove_chunk(node)
                        removed += 1
                        found = True
                        break
            if not found:
                seen[norm] = node
        if removed:
            self.index.build()
        return removed

    def prune_sources(self, sources: list[str]) -> int:  # pragma: no cover - heavy
        """Remove nodes and edges originating from the given sources.

        Parameters
        ----------
        sources:
            List of source identifiers to remove from the graph. Nodes with a
            ``source`` attribute matching any of these values are deleted along
            with their associated edges. Edges whose ``provenance`` attribute
            matches are removed as well.

        Returns
        -------
        int
            Number of nodes removed from the graph.
        """

        removed = 0

        for node, data in list(self.graph.nodes(data=True)):
            if data.get("source") in sources:
                self.graph.remove_node(node)
                self.index.remove(node)
                removed += 1

        for u, v, edata in list(self.graph.edges(data=True)):
            if edata.get("provenance") in sources:
                if self.graph.has_edge(u, v):
                    self.graph.remove_edge(u, v)

        if removed:
            self.index.build()

        return removed

    def mark_conflicting_facts(self) -> int:  # pragma: no cover - heavy
        """Flag edges that express conflicting facts.

        A conflict occurs when the same subject and predicate are linked to
        multiple distinct objects. All edges in such groups receive a
        ``conflict`` attribute set to ``True``.

        Returns
        -------
        int
            Number of edges marked as conflicting.
        """

        groups: Dict[tuple[str, str], set[str]] = {}
        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation")
            if not rel or rel in {
                "has_chunk",
                "next_chunk",
                "subject",
                "object",
                "has_fact",
                "mentions",
            }:
                continue
            groups.setdefault((u, rel), set()).add(v)

        marked = 0
        for (subj, rel), objs in groups.items():
            if len(objs) <= 1:
                continue
            for obj in objs:
                edge = self.graph.edges[subj, obj]
                if not edge.get("conflict"):
                    edge["conflict"] = True
                    marked += 1
        return marked

    def clean_chunk_texts(self) -> int:  # pragma: no cover - heavy
        """Normalize text of chunk nodes by stripping HTML and collapsing whitespace.

        Returns
        -------
        int
            Number of chunks modified.
        """

        from bs4 import BeautifulSoup

        changed = 0
        for cid, data in list(self.graph.nodes(data=True)):
            if data.get("type") != "chunk":
                continue
            text = data.get("text", "")
            cleaned = BeautifulSoup(text, "html.parser").get_text(" ")
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            cleaned = "".join(ch for ch in cleaned if ch.isprintable())
            if cleaned != text:
                data["text"] = cleaned
                self.index.remove(cid)
                self.index.add(cid, cleaned)
                changed += 1
        if changed:
            self.index.build()
        return changed

    def normalize_date_fields(self) -> int:  # pragma: no cover - heavy
        """Normalize date-like attributes on nodes to ISO format.

        Fields with names containing ``date`` or ``time`` are parsed using
        :func:`dateutil.parser.parse` and converted to ``YYYY-MM-DD`` strings.

        Returns
        -------
        int
            Number of fields updated.
        """

        changed = 0
        for _, data in list(self.graph.nodes(data=True)):
            for key, value in list(data.items()):
                if not isinstance(value, str):
                    continue
                if "date" not in key and "time" not in key:
                    continue
                try:
                    dt = parser.parse(value)
                except Exception:  # pragma: no cover - parse failure
                    continue
                iso = dt.date().isoformat()
                if iso != value:
                    data[key] = iso
                    changed += 1
        return changed

    def validate_coherence(self) -> int:  # pragma: no cover - heavy
        """Flag logically inconsistent edges such as impossible birth dates.

        The current implementation checks ``parent_of`` relations. If both the
        parent node and child node have a ``birth_date`` attribute and the
        parent's date is later than the child's, the edge receives an
        ``inconsistent`` attribute.

        Returns
        -------
        int
            Number of edges marked as inconsistent.
        """

        marked = 0
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") != "parent_of":
                continue
            p_date = self.graph.nodes[u].get("birth_date")
            c_date = self.graph.nodes[v].get("birth_date")
            if not p_date or not c_date:
                continue
            try:
                p_dt = parser.parse(p_date)
                c_dt = parser.parse(c_date)
            except Exception:  # pragma: no cover - parse failure
                continue
            if p_dt > c_dt and not data.get("inconsistent"):
                data["inconsistent"] = True
                marked += 1
        return marked

    def link_chunks_by_entity(self) -> int:  # pragma: no cover - heavy
        """Connect chunks mentioning the same entity with ``co_mentions`` edges.

        Returns
        -------
        int
            Number of edges added.
        """

        ent_to_chunks: Dict[str, list[str]] = {}
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") != "mentions":
                continue
            if self.graph.nodes[u].get("type") != "chunk":
                continue
            ent_to_chunks.setdefault(v, []).append(u)

        added = 0
        for chunks in ent_to_chunks.values():
            for i, c1 in enumerate(chunks):
                for c2 in chunks[i + 1 :]:
                    if (
                        self.graph.has_edge(c1, c2)
                        and self.graph.edges[c1, c2].get("relation") == "co_mentions"
                    ):
                        continue
                    self.graph.add_edge(c1, c2, relation="co_mentions")
                    added += 1
        return added

    def link_documents_by_entity(self) -> int:  # pragma: no cover - heavy
        """Connect documents that mention the same entity with ``co_mentions`` edges.

        Returns
        -------
        int
            Number of edges added.
        """

        ent_to_docs: Dict[str, list[str]] = {}
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") != "mentions":
                continue
            src_type = self.graph.nodes[u].get("type")
            doc_id: str | None = None
            if src_type == "chunk":
                doc_id = self.get_document_for_chunk(u)
            elif src_type == "document":
                doc_id = u
            if not doc_id:
                continue
            ent_to_docs.setdefault(v, []).append(doc_id)

        added = 0
        for docs in ent_to_docs.values():
            uniq_docs = list(dict.fromkeys(docs))
            for i, d1 in enumerate(uniq_docs):
                for d2 in uniq_docs[i + 1 :]:
                    if (
                        self.graph.has_edge(d1, d2)
                        and self.graph.edges[d1, d2].get("relation") == "co_mentions"
                    ):
                        continue
                    self.graph.add_edge(d1, d2, relation="co_mentions")
                    added += 1
        return added

    def link_sections_by_entity(self) -> int:  # pragma: no cover - heavy
        """Connect sections that mention the same entity with ``co_mentions`` edges.

        Returns
        -------
        int
            Number of edges added.
        """

        ent_to_sections: Dict[str, list[str]] = {}
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") != "mentions":
                continue
            src_type = self.graph.nodes[u].get("type")
            sec_id: str | None = None
            if src_type == "chunk":
                sec_id = self.get_section_for_chunk(u)
            elif src_type == "section":
                sec_id = u
            if not sec_id:
                continue
            ent_to_sections.setdefault(v, []).append(sec_id)

        added = 0
        for secs in ent_to_sections.values():
            uniq_secs = list(dict.fromkeys(secs))
            for i, s1 in enumerate(uniq_secs):
                for s2 in uniq_secs[i + 1 :]:
                    if (
                        self.graph.has_edge(s1, s2)
                        and self.graph.edges[s1, s2].get("relation") == "co_mentions"
                    ):
                        continue
                    self.graph.add_edge(s1, s2, relation="co_mentions")
                    added += 1
        return added

    def link_authors_organizations(self) -> int:  # pragma: no cover - heavy
        """Connect author nodes to their organizations using ``affiliated_with``.

        Documents may store ``author`` and ``organization`` attributes. This
        helper ensures corresponding entity nodes exist and links them.

        Returns
        -------
        int
            Number of edges created.
        """

        added = 0
        for doc_id, data in list(self.graph.nodes(data=True)):
            if data.get("type") != "document":
                continue
            author = data.get("author")
            org = data.get("organization")
            if not author or not org:
                continue
            if not self.graph.has_node(author):
                self.add_entity(author, author, data.get("source"))
            if not self.graph.has_node(org):
                self.add_entity(org, org, data.get("source"))
            if not self.graph.has_edge(author, org):
                self.graph.add_edge(author, org, relation="affiliated_with")
                added += 1
        return added

    def _merge_entity_nodes(
        self, target: str, source: str
    ) -> None:  # pragma: no cover - heavy
        """Merge ``source`` entity into ``target``."""

        for pred, _, edata in list(self.graph.in_edges(source, data=True)):
            self.graph.add_edge(pred, target, **edata)
        for _, succ, edata in list(self.graph.out_edges(source, data=True)):
            self.graph.add_edge(target, succ, **edata)
        if "text" not in self.graph.nodes[target]:
            self.graph.nodes[target]["text"] = self.graph.nodes[source].get("text")
        self.graph.remove_node(source)

    def resolve_entities(  # pragma: no cover - heavy
        self,
        threshold: float = 0.8,
        aliases: dict[str, list[str]] | None = None,
    ) -> int:
        """Merge entity nodes that refer to the same concept.

        Parameters
        ----------
        threshold:
            Minimum cosine similarity for merging using text embeddings.
        aliases:
            Optional mapping of canonical labels to lists of aliases. Nodes whose
            ``text`` matches an alias will be merged into the node matching the
            canonical label before similarity-based merging occurs.
        """

        entities = [
            n for n, d in self.graph.nodes(data=True) if d.get("type") == "entity"
        ]
        texts = [self.graph.nodes[e].get("text", "") for e in entities]
        cfg = load_config()
        min_conf = float(cfg.get("language", {}).get("min_confidence", 0.7))
        langs: List[str] = []
        probs: List[float] = []
        for eid, txt in zip(entities, texts):
            lang = self.graph.nodes[eid].get("lang")
            try:
                from ..utils.text import detect_language

                detected, prob = detect_language(txt, return_prob=True)
            except Exception:  # pragma: no cover - optional
                detected, prob = "und", 0.0
            if not lang:
                lang = detected
                self.graph.nodes[eid]["lang"] = lang
            langs.append(lang)
            probs.append(float(prob))
        if len(entities) < 2:
            return 0

        merged = 0

        used_alias_indices: set[int] = set()
        if aliases:

            def _norm(t: str) -> str:
                return re.sub(r"\W+", "", t).lower()

            label_to_id = {
                _norm(self.graph.nodes[e].get("text", "")): e for e in entities
            }
            for canon, alist in aliases.items():
                target = label_to_id.get(_norm(canon))
                if not target:
                    continue
                for alias in alist:
                    source = label_to_id.get(_norm(alias))
                    if source and source != target:
                        self._merge_entity_nodes(target, source)
                        merged += 1
                        idx = entities.index(source)
                        texts[idx] = canon
                        used_alias_indices.add(idx)
                        label_to_id[_norm(canon)] = target
                        label_to_id.pop(_norm(alias), None)

        embeddings = self.index.transform(texts)
        used: set[int] = set()
        for i, eid1 in enumerate(entities):
            if i in used or i in used_alias_indices:
                continue
            for j in range(i + 1, len(entities)):
                if j in used or j in used_alias_indices:
                    continue
                sim = float(
                    cosine_similarity(
                        embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1)
                    )[0, 0]
                )
                t1 = re.sub(r"\W+", "", texts[i]).lower()
                t2 = re.sub(r"\W+", "", texts[j]).lower()
                allowed = (
                    langs[i] == langs[j]
                    and probs[i] >= min_conf
                    and probs[j] >= min_conf
                )
                if allowed and (sim >= threshold or t1 == t2 or t1 in t2 or t2 in t1):
                    self._merge_entity_nodes(eid1, entities[j])
                    used.add(j)
                    merged += 1
                elif langs[i] != langs[j]:
                    try:
                        from ..analysis.monitoring import lang_mismatch_total as _ctr

                        if _ctr is not None:
                            _ctr.inc()
                    except Exception:
                        pass
        if merged:
            self.index = EmbeddingIndex(use_hnsw=self.use_hnsw)
            for n, d in self.graph.nodes(data=True):
                if d.get("type") in {"chunk", "entity"} and "text" in d:
                    self.index.add(n, d["text"])
            self.index.build()
        return merged

    def extract_entities(
        self, model: str | None = "en_core_web_sm"
    ) -> None:  # pragma: no cover - heavy
        """Run named entity recognition on all chunks and link entities."""

        from datacreek.utils.entity_extraction import extract_entities

        for cid, data in list(self.graph.nodes(data=True)):
            if data.get("type") != "chunk":
                continue
            ents = extract_entities(data.get("text", ""), model=model)
            for ent in ents:
                if not self.graph.has_node(ent):
                    self.add_entity(ent, ent, data.get("source"))
                if not self.graph.has_edge(cid, ent):
                    self.link_entity(cid, ent, provenance=data.get("source"))

    def enrich_entity_wikidata(
        self, entity_id: str
    ) -> None:  # pragma: no cover - heavy
        """Fetch description from Wikidata and store it on the entity node."""

        node = self.graph.nodes.get(entity_id)
        if not node or node.get("type") != "entity":
            raise ValueError("Unknown entity")
        label = node.get("text")
        if not label:
            return
        search = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": label,
                "format": "json",
                "language": "en",
                "limit": 1,
            },
            timeout=10,
        )
        data = search.json()
        if not data.get("search"):
            return
        item = data["search"][0]
        node["wikidata_id"] = item.get("id")
        if desc := item.get("description"):
            node["description"] = desc

    def enrich_entity_dbpedia(self, entity_id: str) -> None:  # pragma: no cover - heavy
        """Fetch description from DBpedia and store it on the entity node."""

        node = self.graph.nodes.get(entity_id)
        if not node or node.get("type") != "entity":
            raise ValueError("Unknown entity")
        label = node.get("text")
        if not label:
            return
        search = requests.get(
            "https://lookup.dbpedia.org/api/search",
            params={"query": label, "maxResults": 1},
            headers={"Accept": "application/json"},
            timeout=10,
        )
        data = search.json()
        results = data.get("docs") or data.get("results")
        if not results:
            return
        item = results[0]
        uri = item.get("uri") or item.get("id")
        if uri:
            node["dbpedia_uri"] = uri
        desc = item.get("description") or item.get("overview")
        if desc:
            node["description_dbpedia"] = desc

    def predict_links(  # pragma: no cover - heavy
        self, threshold: float = 0.8, *, use_graph_embeddings: bool = False
    ) -> None:
        """Create ``related_to`` edges between similar entities.

        Parameters
        ----------
        threshold:
            Minimum cosine similarity for creating a link.
        use_graph_embeddings:
            If ``True`` use Node2Vec embeddings stored on the nodes instead of
            text embeddings for similarity computation. If embeddings are not
            present they will be generated automatically.
        """

        entities = [
            n for n, d in self.graph.nodes(data=True) if d.get("type") == "entity"
        ]
        if len(entities) < 2:
            return

        if use_graph_embeddings:
            if not all("embedding" in self.graph.nodes[e] for e in entities):
                self.compute_node2vec_embeddings()
            emb_matrix = np.array([self.graph.nodes[e]["embedding"] for e in entities])
        else:
            texts = [self.graph.nodes[e].get("text", "") for e in entities]
            emb_matrix = self.index.transform(texts)

        for i, eid1 in enumerate(entities):
            for j in range(i + 1, len(entities)):
                eid2 = entities[j]
                sim = float(
                    cosine_similarity(
                        emb_matrix[i].reshape(1, -1), emb_matrix[j].reshape(1, -1)
                    )[0, 0]
                )
                if sim >= threshold and not self.graph.has_edge(eid1, eid2):
                    self.graph.add_edge(
                        eid1, eid2, relation="related_to", similarity=sim
                    )

    # ------------------------------------------------------------------
    # Advanced operations
    # ------------------------------------------------------------------

    def consolidate_schema(self) -> None:  # pragma: no cover - heavy
        """Normalize node types and relation labels to lowercase."""

        for n, data in list(self.graph.nodes(data=True)):
            if "type" in data:
                data["type"] = str(data["type"]).lower()
        for u, v, data in list(self.graph.edges(data=True)):
            if "relation" in data:
                data["relation"] = str(data["relation"]).lower()

    def _node_embedding(
        self, node: str
    ) -> Optional[np.ndarray]:  # pragma: no cover - heavy
        """Return or compute the embedding vector for ``node``."""

        data = self.graph.nodes[node]
        if "embedding" in data:
            emb = np.array(data["embedding"], dtype=float)
            return emb
        text = data.get("text")
        if not text:
            return None
        vec = self.index.embed(text)
        if vec.size == 0:
            return None
        data["embedding"] = vec.tolist()
        return vec

    def update_embeddings(
        self, node_type: str = "chunk"
    ) -> None:  # pragma: no cover - heavy
        """Materialize embeddings for nodes of ``node_type``."""

        for n, d in self.graph.nodes(data=True):
            if d.get("type") != node_type:
                continue
            text = d.get("text")
            if not text:
                continue
            vec = self.index.embed(text)
            if vec.size:
                self.graph.nodes[n]["embedding"] = vec.tolist()

    def cluster_chunks(self, n_clusters: int = 3) -> None:  # pragma: no cover - heavy
        """Cluster chunk nodes and attach ``community`` nodes."""

        chunks = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "chunk"]
        embeddings = [self._node_embedding(n) for n in chunks]
        embeddings = [e for e in embeddings if e is not None]
        if not embeddings:
            return
        X = np.vstack(embeddings)
        n_clusters = min(n_clusters, len(X))
        km = KMeans(n_clusters=n_clusters, n_init=10)
        labels = km.fit_predict(X)
        for cid in {f"community_{i}" for i in labels}:
            if not self.graph.has_node(cid):
                self.graph.add_node(cid, type="community")
        for node, label in zip(chunks, labels):
            cid = f"community_{label}"
            self.graph.add_edge(node, cid, relation="in_community")

    def cluster_entities(self, n_clusters: int = 3) -> None:  # pragma: no cover - heavy
        """Cluster entity nodes into groups using embeddings."""

        entities = [
            n for n, d in self.graph.nodes(data=True) if d.get("type") == "entity"
        ]
        embeddings = [self._node_embedding(n) for n in entities]
        embeddings = [e for e in embeddings if e is not None]
        if not embeddings:
            return
        X = np.vstack(embeddings)
        n_clusters = min(n_clusters, len(X))
        km = KMeans(n_clusters=n_clusters, n_init=10)
        labels = km.fit_predict(X)
        for gid in {f"entity_group_{i}" for i in labels}:
            if not self.graph.has_node(gid):
                self.graph.add_node(gid, type="entity_group")
        for node, label in zip(entities, labels):
            gid = f"entity_group_{label}"
            self.graph.add_edge(node, gid, relation="in_group")

    def summarize_communities(self) -> None:  # pragma: no cover - heavy
        """Create a simple summary text for each community node."""

        for c in [
            n for n, d in self.graph.nodes(data=True) if d.get("type") == "community"
        ]:
            members = [
                u
                for u, v in self.graph.in_edges(c)
                if self.graph.edges[u, c].get("relation") == "in_community"
            ]
            texts = [self.graph.nodes[m].get("text", "") for m in members]
            joined = " ".join(texts)
            words = joined.split()
            summary = " ".join(words[:20])
            self.graph.nodes[c]["summary"] = summary

    def summarize_entity_groups(self) -> None:  # pragma: no cover - heavy
        """Assign a naive summary to each entity group."""

        for g in [
            n for n, d in self.graph.nodes(data=True) if d.get("type") == "entity_group"
        ]:
            members = [
                u
                for u, v in self.graph.in_edges(g)
                if self.graph.edges[u, g].get("relation") == "in_group"
            ]
            texts = [self.graph.nodes[m].get("text", "") for m in members]
            joined = " ".join(texts)
            words = joined.split()
            self.graph.nodes[g]["summary"] = " ".join(words[:20])

    def score_trust(self) -> None:  # pragma: no cover - heavy
        """Assign a naive trust score based on source frequency."""

        src_counts: Dict[str, int] = {}
        for n, d in self.graph.nodes(data=True):
            src = d.get("source")
            if src:
                src_counts[src] = src_counts.get(src, 0) + 1
        for u, v, d in self.graph.edges(data=True):
            src = d.get("provenance")
            if src:
                src_counts[src] = src_counts.get(src, 0) + 1
        for n, d in self.graph.nodes(data=True):
            src = d.get("source")
            if not src:
                continue
            count = src_counts.get(src, 1)
            d["trust"] = min(1.0, count / 3)
        for u, v, d in self.graph.edges(data=True):
            src = d.get("provenance")
            if not src:
                continue
            count = src_counts.get(src, 1)
            d["trust"] = min(1.0, count / 3)

    def compute_centrality(  # pragma: no cover - heavy
        self, node_type: str = "entity", metric: str = "degree"
    ) -> None:
        """Compute centrality scores for nodes of ``node_type``."""

        if metric == "degree":
            values = nx.degree_centrality(self.graph)
        elif metric == "betweenness":
            values = nx.betweenness_centrality(self.graph)
        else:
            raise ValueError("Unknown metric")

        for node, score in values.items():
            if self.graph.nodes[node].get("type") == node_type:
                self.graph.nodes[node]["centrality"] = float(score)

    def compute_node2vec_embeddings(  # pragma: no cover - heavy
        self,
        dimensions: int = 64,
        walk_length: int = 10,
        num_walks: int = 50,
        workers: int = 1,
        seed: int = 0,
        *,
        p: float = 1.0,
        q: float = 1.0,
    ) -> None:
        """Compute Node2Vec embeddings for all nodes and store them on the nodes."""

        try:
            from node2vec import Node2Vec
        except Exception as e:  # pragma: no cover - dependency missing
            raise RuntimeError("node2vec library is required") from e

        n2v = Node2Vec(
            self.graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            seed=seed,
            p=p,
            q=q,
        )
        model = n2v.fit()
        for node in self.graph.nodes:
            vec = model.wv[str(node)]
            self.graph.nodes[node]["embedding"] = vec.tolist()

    def compute_node2vec_gds(  # pragma: no cover - heavy
        self,
        driver: Driver,
        *,
        dimensions: int = 128,
        walk_length: int = 40,
        walks_per_node: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        dataset: str | None = None,
        write_property: str = "embedding",
    ) -> None:
        """Compute Node2Vec embeddings using Neo4j GDS.

        The graph is first persisted to Neo4j under ``dataset``. The GDS
        procedure ``gds.beta.node2vec.write`` materializes the embeddings on the
        nodes. They are then loaded back into the local NetworkX graph and the
        temporary projection is removed.
        """

        if GraphDatabase is None:
            raise RuntimeError("neo4j library is required")

        ds = dataset or "kg_n2v_temp"
        self.to_neo4j(driver, dataset=ds, clear=True)

        node_query = "MATCH (n {dataset:$ds}) RETURN id(n) AS id"
        rel_query = "MATCH (n {dataset:$ds})-[r]->(m {dataset:$ds}) RETURN id(n) AS source, id(m) AS target"

        with driver.session() as session:
            session.run("CALL gds.graph.drop('kg_n2v', false)")
            session.run(
                "CALL gds.graph.project.cypher('kg_n2v', $nq, $rq)",
                nq=node_query,
                rq=rel_query,
                ds=ds,
            )
            session.run(
                "CALL gds.beta.node2vec.write('kg_n2v', $config)",
                config={
                    "embeddingDimension": dimensions,
                    "walkLength": walk_length,
                    "walksPerNode": walks_per_node,
                    "p": p,
                    "q": q,
                    "writeProperty": write_property,
                },
            )
            res = session.run(
                "MATCH (n {dataset:$ds}) RETURN n.id AS id, n[$prop] AS emb",
                ds=ds,
                prop=write_property,
            )
            for rec in res:
                self.graph.nodes[rec["id"]][write_property] = list(rec["emb"])
            session.run("CALL gds.graph.drop('kg_n2v')")
            session.run("MATCH (n {dataset:$ds}) DETACH DELETE n", ds=ds)

    def compute_graphwave_embeddings(  # pragma: no cover - heavy
        self,
        scales: Iterable[float],
        num_points: int = 10,
        *,
        chebyshev_order: int | None = None,
        gpu: bool = False,
    ) -> None:
        """Compute GraphWave embeddings for all nodes.

        Parameters
        ----------
        scales:
            Diffusion scales used for the wavelets.
        num_points:
            Number of sample points for the characteristic function.
        """

        if gpu:
            from ..analysis.graphwave_cuda import graphwave_embedding_gpu as _gwg

            emb = _gwg(
                self.graph.to_undirected(),
                scales,
                num_points=num_points,
                order=chebyshev_order or 7,
            )
        else:
            if chebyshev_order is None:
                from ..analysis.fractal import graphwave_embedding as _gw

                emb = _gw(self.graph.to_undirected(), scales, num_points)
            else:
                from ..analysis.fractal import graphwave_embedding_chebyshev as _gwc

                emb = _gwc(
                    self.graph.to_undirected(),
                    scales,
                    num_points=num_points,
                    order=chebyshev_order,
                )
        for node, vec in emb.items():
            self.graph.nodes[node]["graphwave_embedding"] = vec.tolist()

    def graphwave_entropy(self) -> float:  # pragma: no cover - heavy
        """Return differential entropy of stored GraphWave embeddings."""

        feats = {
            n: data["graphwave_embedding"]
            for n, data in self.graph.nodes(data=True)
            if "graphwave_embedding" in data
        }
        if not feats:
            return 0.0
        from ..analysis.fractal import graphwave_entropy as _ge

        H = _ge(feats)
        self.graph.graph["gw_entropy"] = H
        logger = logging.getLogger(__name__)
        logger.debug("gw_entropy=%.4f", H)
        return H

    def build_faiss_index(  # pragma: no cover - heavy
        self, node_attr: str = "embedding", *, method: str = "flat"
    ) -> None:
        """Build a FAISS index from node embeddings.

        Parameters
        ----------
        node_attr:
            Node attribute containing the embedding vectors.
        """

        try:
            import faiss
            import numpy as np
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("faiss is required") from exc

        vectors = []
        ids = []
        for n, data in self.graph.nodes(data=True):
            if node_attr in data:
                vectors.append(np.asarray(data[node_attr], dtype=np.float32))
                ids.append(n)
        if not vectors:
            raise ValueError("no embeddings found")

        xb = np.vstack(vectors)
        faiss.normalize_L2(xb)
        d = xb.shape[1]
        if method == "hnsw":
            index = faiss.IndexHNSWFlat(d, 32)
            index.hnsw.efSearch = 200
            index.add(xb)
        elif method == "faiss_gpu_ivfpq":
            quantizer = faiss.IndexFlatIP(d)
            cpu_index = faiss.IndexIVFPQ(quantizer, d, 4096, 16, 8)
            cpu_index.train(xb)
            cpu_index.add(xb)
            cpu_index.nprobe = 32
            try:
                if faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                else:
                    index = cpu_index
            except Exception:
                index = cpu_index
        else:
            index = faiss.IndexFlatIP(d)
            index.add(xb)

        self.faiss_index = index
        self.faiss_ids = ids
        self.faiss_index_type = method
        self.faiss_node_attr = node_attr
        try:
            from ..analysis.monitoring import ann_backend

            if ann_backend is not None:
                if method == "faiss_gpu_ivfpq":
                    ann_backend.set(3)
                elif method == "hnsw":
                    ann_backend.set(2)
                else:
                    ann_backend.set(1)
        except Exception:
            pass

    def search_faiss(  # pragma: no cover - heavy
        self,
        vector: Iterable[float],
        k: int = 5,
        *,
        adaptive: bool = False,
        latency_threshold: float = 0.1,
    ) -> list[str]:
        """Return ``k`` nearest nodes using the FAISS index.

        If ``adaptive`` is ``True`` and search latency exceeds
        ``latency_threshold`` seconds with a flat index, the index is
        rebuilt using HNSW for faster queries.
        """

        if self.faiss_index is None or self.faiss_ids is None:
            raise RuntimeError("index not built")

        import faiss
        import numpy as np

        xq = np.asarray([vector], dtype=np.float32)
        faiss.normalize_L2(xq)
        start = time.monotonic()
        ctx = ann_latency.time() if ann_latency is not None else nullcontext()
        with ctx:
            _, idx = self.faiss_index.search(xq, k)
        latency = time.monotonic() - start

        if (
            adaptive
            and self.faiss_index_type == "flat"
            and latency > latency_threshold
            and self.faiss_node_attr is not None
        ):
            self.build_faiss_index(self.faiss_node_attr, method="hnsw")
            return self.search_faiss(
                vector,
                k,
                adaptive=False,
                latency_threshold=latency_threshold,
            )

        return [self.faiss_ids[i] for i in idx[0]]

    def ensure_graphwave_entropy(  # pragma: no cover - heavy
        self,
        threshold: float,
        *,
        scales: Iterable[float] = (0.5, 1.0),
        num_points: int = 10,
    ) -> float:
        """Ensure GraphWave entropy above ``threshold``.

        If the current entropy is below ``threshold``, embeddings are recomputed
        with slight random noise added to ``scales``.
        """

        H = self.graphwave_entropy()
        if H < threshold:
            noisy = [float(s) + np.random.uniform(-0.1, 0.1) for s in scales]
            self.compute_graphwave_embeddings(noisy, num_points)
            H = self.graphwave_entropy()
        self.graph.graph["gw_entropy"] = H
        return H

    def embedding_entropy(
        self, node_attr: str = "embedding"
    ) -> float:  # pragma: no cover - heavy
        """Return differential entropy of vectors stored under ``node_attr``."""

        feats = {
            n: data[node_attr]
            for n, data in self.graph.nodes(data=True)
            if node_attr in data
        }
        if not feats:
            return 0.0
        from ..analysis.fractal import embedding_entropy as _ee

        return _ee(feats)

    def compute_poincare_embeddings(  # pragma: no cover - heavy
        self,
        dim: int = 2,
        negative: int = 5,
        epochs: int = 50,
        learning_rate: float = 0.1,
        burn_in: int = 10,
    ) -> None:
        """Compute hyperbolic Poincaré embeddings for nodes.

        Parameters
        ----------
        dim:
            Dimension of the embedding space.
        negative:
            Number of negative samples.
        epochs:
            Number of training epochs.
        learning_rate:
            Learning rate for optimization.
        burn_in:
            Burn-in epochs before negative sampling.
        """

        from ..analysis import fp8_quantize
        from ..analysis.fractal import poincare_embedding

        emb = poincare_embedding(
            self.graph,
            dim=dim,
            negative=negative,
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
        )
        for node, vec in emb.items():
            q, scale = fp8_quantize(vec)
            self.graph.nodes[node]["poincare_fp8"] = q.tolist()
            self.graph.nodes[node]["poincare_scale"] = scale
            self.graph.nodes[node]["poincare_embedding"] = vec.tolist()

    def compute_hyperbolic_hypergraph_embeddings(  # pragma: no cover - heavy
        self,
        dim: int = 2,
        negative: int = 5,
        epochs: int = 50,
        learning_rate: float = 0.1,
        burn_in: int = 10,
    ) -> Dict[object, list[float]]:
        """Compute hyperbolic embeddings treating hyperedges explicitly.

        Hyperedges are represented as nodes with ``type`` ``"hyperedge"``. The
        returned embeddings are stored under ``"hyperbolic_embedding"`` for all
        non-hyperedge nodes and also returned as a dictionary.
        """

        import networkx as nx

        from ..analysis.fractal import poincare_embedding

        g = nx.Graph()
        g.add_nodes_from(self.graph.nodes())
        for u, v, data in self.graph.edges(data=True):
            if (
                self.graph.nodes[u].get("type") == "hyperedge"
                or self.graph.nodes[v].get("type") == "hyperedge"
            ):
                g.add_edge(u, v)

        emb = poincare_embedding(
            g,
            dim=dim,
            negative=negative,
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
        )
        results: Dict[object, list[float]] = {}
        for node, vec in emb.items():
            if self.graph.nodes[node].get("type") != "hyperedge":
                self.graph.nodes[node]["hyperbolic_embedding"] = vec.tolist()
                results[node] = vec.tolist()
        return results

    def compute_learned_hyperbolic_embeddings(  # pragma: no cover - heavy
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
        """Learn hyperbolic embeddings with fractal regularization."""

        import numpy as np

        nodes = {
            n: np.asarray(data[feature_attr], dtype=float)
            for n, data in self.graph.nodes(data=True)
            if feature_attr in data
        }
        if not nodes:
            return {}
        from ..analysis import latent_box_dimension as _lbd
        from ..analysis import learn_hyperbolic_projection as _lhp

        if target_dim is None:
            target_dim, _ = _lbd(nodes, radii=[0.5, 1.0])

        emb = _lhp(
            nodes,
            self.graph.subgraph(nodes.keys()),
            dim=dim,
            lr=lr,
            epochs=epochs,
            geo_weight=geo_weight,
            frac_weight=frac_weight,
            frac_target=target_dim,
        )
        for node, vec in emb.items():
            self.graph.nodes[node][write_property] = vec.astype(float).tolist()
        return {n: v.astype(float).tolist() for n, v in emb.items()}

    def compute_hyper_sagnn_embeddings(  # pragma: no cover - heavy
        self,
        *,
        node_attr: str = "embedding",
        edge_attr: str = "hyper_sagnn_embedding",
        embed_dim: int | None = None,
        seed: int | None = None,
    ) -> Dict[str, list[float]]:
        """Compute Hyper-SAGNN-like embeddings for hyperedges.

        Hyperedges are nodes with ``type`` ``"hyperedge"`` connected to their
        members. Node features are read from ``node_attr`` and combined using a
        lightweight self-attention mechanism similar to Hyper-SAGNN.
        """

        import numpy as np

        index: Dict[str, int] = {}
        features: List[np.ndarray] = []
        for node, data in self.graph.nodes(data=True):
            if node_attr in data:
                index[node] = len(features)
                features.append(np.asarray(data[node_attr], dtype=float))

        if not features:
            # gracefully handle empty graphs without embeddings
            return {}

        hyper_list: List[tuple[str, List[int]]] = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "hyperedge":
                members = [index[v] for _, v in self.graph.edges(node) if v in index]
                if members:
                    hyper_list.append((node, members))

        if not hyper_list:
            return {}

        from ..analysis.hypergraph import hyper_sagnn_embeddings as _hs

        embeddings = _hs(
            [m for _, m in hyper_list],
            np.stack(features),
            embed_dim=embed_dim,
            seed=seed,
        )

        result: Dict[str, list[float]] = {}
        for (node, _), vec in zip(hyper_list, embeddings):
            self.graph.nodes[node][edge_attr] = vec.astype(float).tolist()
            result[node] = vec.astype(float).tolist()

        return result

    def hyper_adamic_adar_scores(
        self,
    ) -> Dict[tuple[str, str], float]:  # pragma: no cover - heavy
        """Return Hyper-Adamic–Adar scores for node pairs.

        Hyperedges are nodes of type ``"hyperedge"`` connected to their members.
        The score between two nodes is the sum of ``1 / log(|H|)`` over
        hyperedges ``H`` that contain both nodes.
        """

        hyperedges: List[List[str]] = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "hyperedge":
                members = [v for _, v in self.graph.edges(node)]
                if len(members) > 1:
                    hyperedges.append(members)

        from ..analysis.hypergraph import hyper_adamic_adar_scores as _haa

        scores = _haa(hyperedges)
        return {(str(u), str(v)): float(s) for (u, v), s in scores.items()}

    def edge_attention_scores(  # pragma: no cover - heavy
        self,
        *,
        node_attr: str = "embedding",
        seed: int | None = None,
    ) -> Dict[str, float]:
        """Return attention-based importance scores for hyperedges.

        The score of a hyperedge is the average absolute attention weight
        produced by a lightweight self-attention layer over its member nodes.
        """

        import numpy as np

        hyper_nodes: list[str] = []
        hyperedges: list[list[str]] = []
        features = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "hyperedge":
                members = [v for _, v in self.graph.edges(node)]
                if not members:
                    continue
                hyper_nodes.append(node)
                hyperedges.append(members)
                features.append(
                    np.vstack([self.graph.nodes[m][node_attr] for m in members])
                )

        if not hyperedges:
            return {}

        from ..analysis.hypergraph import hyperedge_attention_scores as _att

        node_features = np.concatenate(features, axis=0)
        sizes = [len(m) for m in hyperedges]
        offsets = np.cumsum([0] + sizes[:-1])
        edge_arrays = [list(range(o, o + s)) for o, s in zip(offsets, sizes)]

        scores = _att(edge_arrays, node_features, seed=seed)

        result: Dict[str, float] = {}
        for node, score in zip(hyper_nodes, scores):
            result[str(node)] = float(score)
        return result

    def compute_hyper_sagnn_head_drop_embeddings(  # pragma: no cover - heavy
        self,
        *,
        node_attr: str = "embedding",
        edge_attr: str = "hyper_sagnn_hd_embedding",
        num_heads: int = 4,
        threshold: float = 0.1,
        seed: int | None = None,
    ) -> Dict[str, list[float]]:
        """Compute Hyper-SAGNN embeddings with HEAD-Drop pruning."""

        import numpy as np

        index: Dict[str, int] = {}
        features: List[np.ndarray] = []
        for node, data in self.graph.nodes(data=True):
            if node_attr in data:
                index[node] = len(features)
                features.append(np.asarray(data[node_attr], dtype=float))

        if not features:
            return {}

        hyper_list: List[tuple[str, List[int]]] = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "hyperedge":
                members = [index[v] for _, v in self.graph.edges(node) if v in index]
                if members:
                    hyper_list.append((node, members))

        if not hyper_list:
            return {}

        from ..analysis.hypergraph import hyper_sagnn_head_drop_embeddings as _hd

        embeddings = _hd(
            [m for _, m in hyper_list],
            np.stack(features),
            num_heads=num_heads,
            threshold=threshold,
            seed=seed,
        )

        result: Dict[str, list[float]] = {}
        for (node, _), vec in zip(hyper_list, embeddings):
            self.graph.nodes[node][edge_attr] = vec.astype(float).tolist()
            result[node] = vec.astype(float).tolist()

        return result

    def compute_hgcn_sagnn_embeddings(  # pragma: no cover - heavy
        self,
        *,
        node_attr: str = "embedding",
        edge_attr: str = "hgcn_sagnn_embedding",
        K: int = 2,
        embed_dim: int | None = None,
        seed: int | None = None,
    ) -> Dict[str, list[float]]:
        """Compute Hyper-SAGNN embeddings using HGCN spectral features.

        Node features are first filtered through a hypergraph Laplacian via a
        :math:`K`-order Chebyshev convolution, capturing global structure. The
        filtered features are then combined with standard Hyper-SAGNN embeddings
        computed from the original features. The final embedding is the
        concatenation of local and spectral components.
        """

        import numpy as np

        index: Dict[str, int] = {}
        features: list[np.ndarray] = []
        for node, data in self.graph.nodes(data=True):
            if node_attr in data:
                index[node] = len(features)
                features.append(np.asarray(data[node_attr], dtype=float))

        if not features:
            return {}

        hyper_list: list[tuple[str, list[int]]] = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "hyperedge":
                members = [index[v] for _, v in self.graph.edges(node) if v in index]
                if members:
                    hyper_list.append((node, members))

        if not hyper_list:
            return {}

        num_nodes = len(features)
        num_edges = len(hyper_list)
        B = np.zeros((num_nodes, num_edges), dtype=float)
        for j, (_, members) in enumerate(hyper_list):
            B[members, j] = 1.0

        from ..analysis.hypergraph import hyper_sagnn_embeddings as _hs
        from ..analysis.hypergraph_conv import chebyshev_conv as _cc
        from ..analysis.hypergraph_conv import hypergraph_laplacian as _hl

        Delta = _hl(B)
        feats = np.stack(features)
        spec_feats = _cc(feats, Delta, K=K)

        local_emb = _hs(
            [m for _, m in hyper_list], feats, embed_dim=embed_dim, seed=seed
        )
        global_emb = _hs(
            [m for _, m in hyper_list], spec_feats, embed_dim=embed_dim, seed=seed
        )
        embeddings = np.concatenate([local_emb, global_emb], axis=1)

        result: Dict[str, list[float]] = {}
        for (node, _), vec in zip(hyper_list, embeddings):
            self.graph.nodes[node][edge_attr] = vec.astype(float).tolist()
            result[node] = vec.astype(float).tolist()

        return result

    def compute_graphsage_embeddings(  # pragma: no cover - heavy
        self,
        *,
        dimensions: int = 64,
        num_layers: int = 2,
    ) -> None:
        """Compute GraphSAGE-style embeddings for all nodes.

        This implementation performs mean aggregation over node neighborhoods
        using existing Node2Vec embeddings as initial features. The aggregated
        representations are reduced back to ``dimensions`` using PCA.

        Parameters
        ----------
        dimensions:
            Dimensionality of the final embeddings. Node2Vec embeddings are
            generated with the same size if missing.
        num_layers:
            Number of neighborhood aggregation rounds.
        """

        if not all(
            "embedding" in self.graph.nodes[n]
            and len(self.graph.nodes[n]["embedding"]) == dimensions
            for n in self.graph.nodes
        ):
            self.compute_node2vec_embeddings(
                dimensions=dimensions,
                walk_length=10,
                num_walks=50,
                workers=1,
                seed=0,
            )

        feats = {
            n: np.asarray(self.graph.nodes[n]["embedding"], dtype=float)
            for n in self.graph.nodes
        }

        for _ in range(max(1, num_layers)):
            new_feats: Dict[str, np.ndarray] = {}
            for node in self.graph.nodes:
                neigh_vecs = [
                    feats[n] for n in self.graph.neighbors(node) if n in feats
                ]
                if neigh_vecs:
                    neigh_mean = np.mean(neigh_vecs, axis=0)
                else:
                    neigh_mean = np.zeros_like(feats[node])
                new_feats[node] = np.concatenate([feats[node], neigh_mean])
            feats = new_feats

        matrix = np.stack([feats[n] for n in self.graph.nodes])
        n_components = min(dimensions, matrix.shape[0], matrix.shape[1])
        pca = PCA(n_components=n_components, random_state=0)
        reduced = pca.fit_transform(matrix)
        if n_components < dimensions:
            pad_width = dimensions - n_components
            reduced = np.pad(reduced, ((0, 0), (0, pad_width)))
        for node, vec in zip(self.graph.nodes, reduced):
            self.graph.nodes[node]["graphsage_embedding"] = vec.astype(float).tolist()

    def compute_transe_embeddings(  # pragma: no cover - heavy
        self,
        *,
        dimensions: int = 64,
    ) -> None:
        """Compute simple TransE-style relation embeddings.

        This method assumes node embeddings exist under the ``"embedding"``
        attribute. If missing, they are generated with
        :meth:`compute_node2vec_embeddings` using ``dimensions``.

        Parameters
        ----------
        dimensions:
            Dimensionality of the node embeddings. Relation vectors share the
            same size.
        """

        if not all(
            "embedding" in self.graph.nodes[n]
            and len(self.graph.nodes[n]["embedding"]) == dimensions
            for n in self.graph.nodes
        ):
            self.compute_node2vec_embeddings(
                dimensions=dimensions,
                walk_length=10,
                num_walks=50,
                workers=1,
                seed=0,
            )

        rel_vectors: Dict[str, List[np.ndarray]] = {}
        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation")
            if rel is None:
                continue
            head = np.asarray(self.graph.nodes[u]["embedding"], dtype=float)
            tail = np.asarray(self.graph.nodes[v]["embedding"], dtype=float)
            rel_vectors.setdefault(rel, []).append(tail - head)

        rel_means = {
            rel: np.mean(vectors, axis=0) if vectors else np.zeros(dimensions)
            for rel, vectors in rel_vectors.items()
        }

        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation")
            if rel and rel in rel_means:
                data["transe_embedding"] = rel_means[rel].astype(float).tolist()

    def compute_distmult_embeddings(  # pragma: no cover - heavy
        self,
        *,
        dimensions: int = 64,
    ) -> None:
        """Compute DistMult-style relation embeddings.

        Relation vectors are derived from the elementwise products of head
        and tail node embeddings. Node embeddings are generated with
        :meth:`compute_node2vec_embeddings` if missing.
        """

        if not all(
            "embedding" in self.graph.nodes[n]
            and len(self.graph.nodes[n]["embedding"]) == dimensions
            for n in self.graph.nodes
        ):
            self.compute_node2vec_embeddings(
                dimensions=dimensions,
                walk_length=10,
                num_walks=50,
                workers=1,
                seed=0,
            )

        rel_vectors: Dict[str, List[np.ndarray]] = {}
        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation")
            if rel is None:
                continue
            head = np.asarray(self.graph.nodes[u]["embedding"], dtype=float)
            tail = np.asarray(self.graph.nodes[v]["embedding"], dtype=float)
            rel_vectors.setdefault(rel, []).append(head * tail)

        rel_means = {
            rel: np.mean(vectors, axis=0) if vectors else np.zeros(dimensions)
            for rel, vectors in rel_vectors.items()
        }

        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation")
            if rel and rel in rel_means:
                data["distmult_embedding"] = rel_means[rel].astype(float).tolist()

    def compute_multigeometric_embeddings(  # pragma: no cover - heavy
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
    ) -> None:
        """Compute Node2Vec, GraphWave and Poincar\u00e9 embeddings."""

        if graphwave_scales is None:
            graphwave_scales = [0.5, 1.0]

        self.compute_node2vec_embeddings(
            dimensions=node2vec_dim,
            walk_length=10,
            num_walks=50,
            workers=1,
            seed=0,
        )
        self.compute_graphwave_embeddings(
            scales=graphwave_scales,
            num_points=graphwave_points,
        )
        max_neg = max(1, len(self.graph.nodes) - 2)
        self.compute_poincare_embeddings(
            dim=poincare_dim,
            negative=min(negative, max_neg),
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
        )

    def compute_product_manifold_embeddings(  # pragma: no cover - heavy
        self,
        *,
        hyperbolic_attr: str = "poincare_embedding",
        euclidean_attr: str = "embedding",
        write_property: str = "product_embedding",
    ) -> None:
        """Combine hyperbolic and Euclidean embeddings."""

        from ..analysis import product_embedding as _pe

        hyper = {
            n: data[hyperbolic_attr]
            for n, data in self.graph.nodes(data=True)
            if hyperbolic_attr in data and euclidean_attr in data
        }
        eucl = {
            n: data[euclidean_attr]
            for n, data in self.graph.nodes(data=True)
            if hyperbolic_attr in data and euclidean_attr in data
        }
        emb = _pe(hyper, eucl)
        for node, vec in emb.items():
            self.graph.nodes[node][write_property] = vec.tolist()

    def train_product_manifold_embeddings(  # pragma: no cover - heavy
        self,
        contexts: Iterable[tuple[str, str]],
        *,
        hyperbolic_attr: str = "poincare_embedding",
        euclidean_attr: str = "embedding",
        alpha: float = 0.5,
        lr: float = 0.01,
        epochs: int = 1,
    ) -> None:
        """Optimize embeddings using a simple product-manifold loss."""

        from ..analysis import train_product_manifold as _tpm

        hyper = {
            n: data[hyperbolic_attr]
            for n, data in self.graph.nodes(data=True)
            if hyperbolic_attr in data and euclidean_attr in data
        }
        eucl = {n: self.graph.nodes[n][euclidean_attr] for n in hyper}

        h_new, e_new = _tpm(
            hyper,
            eucl,
            contexts,
            alpha=alpha,
            lr=lr,
            epochs=epochs,
        )

        for node, vec in h_new.items():
            self.graph.nodes[node][hyperbolic_attr] = vec.tolist()
        for node, vec in e_new.items():
            self.graph.nodes[node][euclidean_attr] = vec.tolist()

    def compute_aligned_cca_embeddings(  # pragma: no cover - heavy
        self,
        *,
        n_components: int = 32,
        n2v_attr: str = "embedding",
        gw_attr: str = "graphwave_embedding",
        write_property: str = "acca_embedding",
        path: str = os.path.join(CACHE_ROOT, "cca.pkl"),
    ) -> None:
        """Compute A-CCA latent vectors between Node2Vec and GraphWave."""

        import os

        import numpy as np

        from ..analysis import aligned_cca as _acca
        from ..analysis import cca_align, load_cca

        n2v = {
            n: data[n2v_attr]
            for n, data in self.graph.nodes(data=True)
            if n2v_attr in data and gw_attr in data
        }
        gw = {
            n: data[gw_attr]
            for n, data in self.graph.nodes(data=True)
            if n2v_attr in data and gw_attr in data
        }
        if os.path.exists(path):
            W1, _ = load_cca(path)
            nodes = list(n2v)
            X = np.vstack([np.asarray(n2v[n], dtype=float) for n in nodes])
            X_c = X @ W1
            latent = {nodes[i]: X_c[i] for i in range(len(nodes))}
        else:
            latent = cca_align(n2v, gw, n_components=n_components, path=path)
        for node, vec in latent.items():
            self.graph.nodes[node][write_property] = vec.tolist()

    def multiview_contrastive_loss(  # pragma: no cover - heavy
        self,
        *,
        n2v_attr: str = "embedding",
        gw_attr: str = "graphwave_embedding",
        hyper_attr: str = "poincare_embedding",
        tau: float = 0.1,
    ) -> float:
        """Compute InfoNCE loss across Node2Vec, GraphWave and Poincar\xe9 embeddings."""

        from ..analysis import multiview_contrastive_loss as _mcl

        n2v = {
            n: data[n2v_attr]
            for n, data in self.graph.nodes(data=True)
            if n2v_attr in data and gw_attr in data and hyper_attr in data
        }
        gw = {n: data[gw_attr] for n, data in self.graph.nodes(data=True) if n in n2v}
        hyp = {n: data[hyper_attr] for n in n2v}
        return _mcl(n2v, gw, hyp, tau=tau)

    def compute_meta_embeddings(  # pragma: no cover - heavy
        self,
        *,
        n2v_attr: str = "embedding",
        gw_attr: str = "graphwave_embedding",
        hyper_attr: str = "poincare_embedding",
        bottleneck: int = 64,
        write_property: str = "meta_embedding",
    ) -> None:
        """Compute meta-embeddings with a simple autoencoder."""

        from ..analysis import meta_autoencoder as _ma

        n2v = {
            n: data[n2v_attr]
            for n, data in self.graph.nodes(data=True)
            if n2v_attr in data and gw_attr in data and hyper_attr in data
        }
        gw = {n: data[gw_attr] for n in n2v}
        hyp = {n: data[hyper_attr] for n in n2v}
        latent, _ = _ma(n2v, gw, hyp, bottleneck=bottleneck)
        for node, vec in latent.items():
            self.graph.nodes[node][write_property] = vec.tolist()

    def prune_embeddings(
        self, *, tol: float = 1e-3
    ) -> Dict[str, int]:  # pragma: no cover - heavy
        """Cluster embeddings using :func:`fractal_net_prune`."""

        from ..analysis.fractal import fractal_net_prune as _fp

        feats = {
            n: np.asarray(data.get("embedding"), dtype=float)
            for n, data in self.graph.nodes(data=True)
            if "embedding" in data
        }
        if not feats:
            return {}
        _, mapping = _fp(feats, tol=tol)
        for node, idx in mapping.items():
            self.graph.nodes[node]["pruned_class"] = int(idx)
        return {str(n): int(c) for n, c in mapping.items()}

    def fractalnet_compress(  # pragma: no cover - heavy
        self, node_attr: str = "embedding"
    ) -> Dict[int, np.ndarray]:
        """Return level-wise averaged embeddings using :func:`fractalnet_compress`."""

        from ..analysis.fractal import fractalnet_compress as _fc

        embeddings = {
            n: np.asarray(data.get(node_attr), dtype=float)
            for n, data in self.graph.nodes(data=True)
            if node_attr in data
        }
        levels = {
            n: int(data["fractal_level"])
            for n, data in self.graph.nodes(data=True)
            if "fractal_level" in data
        }
        if not embeddings or not levels:
            return {}
        return _fc(embeddings, levels)

    def prune_fractalnet_weights(  # pragma: no cover - heavy
        self, weights: "np.ndarray | list[float]", *, ratio: float = 0.5
    ):
        """Prune model weights using :func:`prune_fractalnet`."""

        from ..analysis.compression import prune_fractalnet as _pf

        return _pf(weights, ratio=ratio)

    # ------------------------------------------------------------------
    # Fractal and topological metrics
    # ------------------------------------------------------------------

    def box_counting_dimension(  # pragma: no cover - heavy
        self, radii: Iterable[int]
    ) -> tuple[float, list[tuple[int, int]]]:
        """Estimate fractal dimension via box covering.

        Parameters
        ----------
        radii:
            Iterable of box radii used for the covering.

        Returns
        -------
        tuple[float, list[tuple[int, int]]]
            Estimated dimension and the ``(radius, count)`` pairs.
        """

        from ..analysis.fractal import box_counting_dimension as _bcd

        return _bcd(self.graph.to_undirected(), radii)

    def colour_box_dimension(  # pragma: no cover - heavy
        self, radii: Iterable[int]
    ) -> tuple[float, list[tuple[int, int]]]:
        """Estimate fractal dimension with the COLOUR-box method."""

        from ..analysis.fractal import colour_box_dimension as _cbd

        return _cbd(self.graph.to_undirected(), radii)

    def spectral_dimension(  # pragma: no cover - heavy
        self, times: Iterable[float]
    ) -> tuple[float, list[tuple[float, float]]]:
        """Estimate spectral dimension from heat trace scaling."""

        from ..analysis.fractal import spectral_dimension as _sd

        return _sd(self.graph.to_undirected(), times)

    def spectral_entropy(
        self, normed: bool = True
    ) -> float:  # pragma: no cover - heavy
        """Return the Shannon entropy of the Laplacian spectrum."""

        from ..analysis.fractal import spectral_entropy as _se

        return _se(self.graph.to_undirected(), normed=normed)

    def spectral_gap(self, normed: bool = True) -> float:  # pragma: no cover - heavy
        """Return the spectral gap of the graph."""

        from ..analysis.fractal import spectral_gap as _sg

        return _sg(self.graph.to_undirected(), normed=normed)

    def laplacian_energy(
        self, normed: bool = True
    ) -> float:  # pragma: no cover - heavy
        """Return the Laplacian energy of the graph."""

        from ..analysis.fractal import laplacian_energy as _le

        return _le(self.graph.to_undirected(), normed=normed)

    def lacunarity(self, radius: int = 1) -> float:  # pragma: no cover - heavy
        """Return lacunarity of the graph for ``radius``."""

        from ..analysis.fractal import graph_lacunarity as _gl

        return _gl(self.graph.to_undirected(), radius=radius)

    def sheaf_laplacian(
        self, edge_attr: str = "sheaf_sign"
    ) -> np.ndarray:  # pragma: no cover - heavy
        """Return the sheaf Laplacian matrix using ``edge_attr`` for signs."""

        from ..analysis.sheaf import sheaf_laplacian as _sl

        return _sl(self.graph, edge_attr=edge_attr)

    def sheaf_convolution(  # pragma: no cover - heavy
        self,
        features: Dict[str, Iterable[float]],
        *,
        edge_attr: str = "sheaf_sign",
        alpha: float = 0.1,
    ) -> Dict[str, list[float]]:
        """Return one sheaf convolution step on ``features``."""

        from ..analysis.sheaf import sheaf_convolution as _sc

        arr_feats = {n: np.asarray(f, dtype=float) for n, f in features.items()}
        result = _sc(self.graph, arr_feats, edge_attr=edge_attr, alpha=alpha)
        return {str(n): vec.tolist() for n, vec in result.items()}

    def sheaf_neural_network(  # pragma: no cover - heavy
        self,
        features: Dict[str, Iterable[float]],
        *,
        layers: int = 2,
        alpha: float = 0.1,
        edge_attr: str = "sheaf_sign",
    ) -> Dict[str, list[float]]:
        """Return features after a simple sheaf neural network."""

        from ..analysis.sheaf import sheaf_neural_network as _snn

        arr_feats = {n: np.asarray(f, dtype=float) for n, f in features.items()}
        result = _snn(
            self.graph,
            arr_feats,
            layers=layers,
            alpha=alpha,
            edge_attr=edge_attr,
        )
        return {str(n): vec.tolist() for n, vec in result.items()}

    def sheaf_cohomology(
        self, edge_attr: str = "sheaf_sign", tol: float = 1e-5
    ) -> int:  # pragma: no cover - heavy
        """Return dimension of :math:`H^1` for the sheaf defined by ``edge_attr``."""

        from ..analysis.sheaf import sheaf_first_cohomology as _sfc

        return _sfc(self.graph, edge_attr=edge_attr, tol=tol)

    def sheaf_cohomology_blocksmith(  # pragma: no cover - heavy
        self,
        *,
        edge_attr: str = "sheaf_sign",
        block_size: int = 40000,
        tol: float = 1e-5,
    ) -> int:
        """Approximate :math:`H^1` using a block-Smith reduction."""

        from ..analysis.sheaf import sheaf_first_cohomology_blocksmith as _sfcbs

        return _sfcbs(self.graph, edge_attr=edge_attr, block_size=block_size, tol=tol)

    def resolve_sheaf_obstruction(  # pragma: no cover - heavy
        self, *, edge_attr: str = "sheaf_sign", max_iter: int = 10
    ) -> int:
        """Resolve cohomological obstructions by flipping edge signs."""

        from ..analysis.sheaf import resolve_sheaf_obstruction as _rso

        return _rso(self.graph, edge_attr=edge_attr, max_iter=max_iter)

    def sheaf_consistency_score(
        self, edge_attr: str = "sheaf_sign"
    ) -> float:  # pragma: no cover - heavy
        """Return a score in [0, 1] measuring sheaf consistency."""

        from ..analysis.sheaf import sheaf_consistency_score as _scs

        return _scs(self.graph, edge_attr=edge_attr)

    def sheaf_consistency_score_batched(  # pragma: no cover - heavy
        self,
        batches: Iterable[Iterable[str]],
        *,
        edge_attr: str = "sheaf_sign",
    ) -> list[float]:
        """Return sheaf consistency scores for ``batches`` of nodes."""

        from ..analysis.sheaf import sheaf_consistency_score_batched as _scsb

        return _scsb(self.graph, batches, edge_attr=edge_attr)

    def spectral_bound_exceeded(  # pragma: no cover - heavy
        self, k: int, tau: float, *, edge_attr: str = "sheaf_sign"
    ) -> bool:
        r"""Return ``True`` if :math:`\lambda_k^\mathcal{F} > \tau`.

        Parameters
        ----------
        k:
            Index of the eigenvalue (1-indexed).
        tau:
            Threshold used for early stopping.
        edge_attr:
            Edge attribute storing sheaf restrictions.
        """

        from ..analysis.sheaf import spectral_bound_exceeded as _sbe

        return _sbe(self.graph, k, tau, edge_attr=edge_attr)

    def rollback_gremlin_diff(
        self, output: str = "rollback.diff"
    ) -> str:  # pragma: no cover - heavy
        """Write diff of the last commit and return its path."""

        from ..analysis.rollback import rollback_gremlin_diff as _rgd

        return _rgd(Path.cwd(), output)

    def sheaf_checker_sla(
        self, failures: Iterable[float]
    ) -> float:  # pragma: no cover - heavy
        """Return MTTR in hours for sheaf checker failures."""

        from ..analysis.rollback import SheafSLA as _sla

        sla = _sla()
        for ts in failures:
            sla.record_failure(ts)
        return sla.mttr_hours()

    def path_to_text(self, path: Iterable) -> str:  # pragma: no cover - heavy
        """Return a textual description for ``path`` using node attributes."""

        from ..utils.graph_text import neighborhood_to_sentence

        return neighborhood_to_sentence(self.graph, path)

    def neighborhood_to_sentence(
        self, path: Iterable
    ) -> str:  # pragma: no cover - heavy
        """Alias of :meth:`path_to_text` for backward compatibility."""

        return self.path_to_text(path)

    def subgraph_to_text(self, nodes: Iterable) -> str:  # pragma: no cover - heavy
        """Return a textual summary for a subgraph made of ``nodes``."""

        from ..utils.graph_text import subgraph_to_text

        return subgraph_to_text(self.graph, nodes)

    def graph_to_text(self) -> str:  # pragma: no cover - heavy
        """Return a textual summary of the entire graph."""

        from ..utils.graph_text import graph_to_text

        return graph_to_text(self.graph)

    def auto_tool_calls(  # pragma: no cover - heavy
        self, node_id: str, tools: Iterable[tuple[str, str]]
    ) -> str:
        """Insert tool call placeholders into a node's text."""

        if not self.graph.has_node(node_id):
            raise ValueError(f"Unknown node: {node_id}")

        from ..utils.toolformer import insert_tool_calls

        text = str(self.graph.nodes[node_id].get("text", ""))
        updated = insert_tool_calls(text, tools)
        self.graph.nodes[node_id]["text"] = updated
        return updated

    def auto_tool_calls_all(  # pragma: no cover - heavy
        self, tools: Iterable[tuple[str, str]]
    ) -> Dict[object, str]:
        """Insert tool call placeholders on every node with text."""

        from ..utils.toolformer import insert_tool_calls

        results: Dict[object, str] = {}
        for n, data in self.graph.nodes(data=True):
            text = data.get("text")
            if not text:
                continue
            updated = insert_tool_calls(str(text), tools)
            self.graph.nodes[n]["text"] = updated
            results[n] = updated
        return results

    def graph_information_bottleneck(  # pragma: no cover - heavy
        self,
        labels: Dict[str, int],
        *,
        beta: float = 1.0,
    ) -> float:
        """Return the Graph Information Bottleneck loss for node embeddings."""

        from ..analysis.information import graph_information_bottleneck as _gib

        feats = {
            n: np.asarray(self.graph.nodes[n]["embedding"], dtype=float)
            for n in self.graph.nodes
            if "embedding" in self.graph.nodes[n]
        }
        return _gib(feats, labels, beta=beta)

    def graph_entropy(self, *, base: float = 2.0) -> float:  # pragma: no cover - heavy
        """Return Shannon entropy of the degree distribution."""

        from ..analysis.information import graph_entropy as _ge

        return _ge(self.graph.to_undirected(), base=base)

    def subgraph_entropy(
        self, nodes: Iterable, *, base: float = 2.0
    ) -> float:  # pragma: no cover - heavy
        """Return entropy of node degrees restricted to ``nodes``."""

        from ..analysis.information import subgraph_entropy as _se

        return _se(self.graph.to_undirected(), nodes, base=base)

    def subgraph_fractal_dimension(
        self, nodes: Iterable, radii: Iterable[int]
    ) -> float:
        """Compute box-counting fractal dimension for ``nodes``.

        The resulting dimension is stored in each node's ``fractal_dim``
        attribute so that downstream analyses can query for complex regions
        of the graph.
        """

        from ..analysis.fractal import box_counting_dimension as _bcd

        sub = self.graph.subgraph(nodes)
        dim, _ = _bcd(sub.to_undirected(), radii)
        for n in sub.nodes:
            self.graph.nodes[n]["fractal_dim"] = dim
        return dim

    def structural_entropy(
        self, tau: int, *, base: float = 2.0
    ) -> float:  # pragma: no cover - heavy
        """Return structural entropy filtered by triangle threshold ``tau``."""

        from ..analysis.information import structural_entropy as _se

        return _se(self.graph.to_undirected(), tau, base=base)

    def adaptive_triangle_threshold(  # pragma: no cover - heavy
        self,
        *,
        weight: str = "weight",
        base: float = 2.0,
        scale: float = 10.0,
    ) -> int:
        """Return entropy-based triangle threshold.

        Parameters
        ----------
        weight:
            Edge attribute storing weights.
        base:
            Logarithm base used for entropy.
        scale:
            Scaling factor to convert entropy into an integer threshold.
        """

        from ..analysis.filtering import entropy_triangle_threshold as _ett

        return _ett(self.graph.to_undirected(), weight=weight, base=base, scale=scale)

    def governance_metrics(  # pragma: no cover - heavy
        self,
        *,
        n2v_attr: str = "embedding",
        gw_attr: str = "graphwave_embedding",
        hyp_attr: str = "poincare_embedding",
    ) -> Dict[str, float]:
        """Return alignment and bias metrics for stored embeddings."""

        from ..analysis.governance import governance_metrics as _gm

        n2v = {
            n: np.asarray(data[n2v_attr], dtype=float)
            for n, data in self.graph.nodes(data=True)
            if n2v_attr in data
        }
        gw = {
            n: np.asarray(data[gw_attr], dtype=float)
            for n, data in self.graph.nodes(data=True)
            if gw_attr in data
        }
        hyp = {
            n: np.asarray(data[hyp_attr], dtype=float)
            for n, data in self.graph.nodes(data=True)
            if hyp_attr in data
        }
        return _gm(n2v, gw, hyp)

    def mitigate_bias_wasserstein(  # pragma: no cover - heavy
        self,
        groups: Dict[str, str],
        *,
        attr: str = "embedding",
    ) -> Dict[str, np.ndarray]:
        """Return reweighted embeddings using Wasserstein bias mitigation."""

        from ..analysis.governance import mitigate_bias_wasserstein as _mbw

        emb = {
            n: np.asarray(data[attr], dtype=float)
            for n, data in self.graph.nodes(data=True)
            if attr in data and n in groups
        }
        return _mbw(emb, groups)

    def average_hyperbolic_radius(  # pragma: no cover - heavy
        self, *, attr: str = "poincare_embedding"
    ) -> float:
        """Return mean Poincaré radius of stored embeddings.

        Parameters
        ----------
        attr:
            Node attribute containing Poincaré vectors.
        """

        from ..analysis.governance import average_hyperbolic_radius as _ahr

        emb = {
            n: np.asarray(data[attr], dtype=float)
            for n, data in self.graph.nodes(data=True)
            if attr in data
        }
        if not emb:
            return 0.0
        return _ahr(emb)

    def autotune_step(  # pragma: no cover - heavy
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
        """Run one autotuning iteration on the current graph."""

        from ..analysis.autotune import autotune_step as _at

        feats = {
            n: np.asarray(data[node_attr], dtype=float)
            for n, data in self.graph.nodes(data=True)
            if node_attr in data
        }
        return _at(
            self.graph.to_undirected(),
            feats,
            labels,
            motifs,
            state,
            weights=weights,
            recall_data=None,
            penalty_cfg=penalty_cfg,
            lr=lr,
            latency=latency,
        )

    def svgp_ei_propose(  # pragma: no cover - heavy
        self,
        history: Sequence[tuple[Sequence[float], float]],
        bounds: Sequence[tuple[float, float]],
        *,
        m: int = 100,
        n_samples: int = 256,
    ) -> list[float]:
        """Propose new parameters using SVGP Expected Improvement."""

        from ..analysis.autotune import svgp_ei_propose as _sv

        params = [h[0] for h in history]
        scores = [h[1] for h in history]
        x_next = _sv(params, scores, bounds, m=m, n_samples=n_samples)
        return x_next.tolist()

    def kw_gradient(  # pragma: no cover - heavy
        self, f: Callable[[float], float], x: float, *, h: float = 1.0, n: int = 4
    ) -> float:
        """Wrapper for :func:`analysis.kw_gradient`."""

        from ..analysis.autotune import kw_gradient as _kw

        return float(_kw(f, x, h=h, n=n))

    def prototype_subgraph(  # pragma: no cover - heavy
        self,
        labels: Dict[str, int],
        class_id: int,
        *,
        radius: int = 1,
    ) -> nx.Graph:
        """Return a prototype subgraph for ``class_id`` using embeddings."""

        from ..analysis.information import prototype_subgraph as _ps

        feats = {
            n: np.asarray(self.graph.nodes[n]["embedding"], dtype=float)
            for n in self.graph.nodes
            if "embedding" in self.graph.nodes[n]
        }
        sub = _ps(self.graph.to_undirected(), feats, labels, class_id, radius=radius)
        return sub

    def select_mdl_motifs(
        self, motifs: Iterable[nx.Graph]
    ) -> List[nx.Graph]:  # pragma: no cover - heavy
        """Return motifs that reduce description length."""

        from ..analysis.information import select_mdl_motifs as _sel

        return _sel(self.graph.to_undirected(), motifs)

    def laplacian_spectrum(
        self, normed: bool = True
    ) -> np.ndarray:  # pragma: no cover - heavy
        """Return the Laplacian eigenvalues of the graph."""

        from ..analysis.fractal import laplacian_spectrum as _ls

        return _ls(self.graph.to_undirected(), normed=normed)

    def spectral_density(  # pragma: no cover - heavy
        self, bins: int = 50, *, normed: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return histogram of Laplacian eigenvalues."""

        from ..analysis.fractal import spectral_density as _sd

        return _sd(self.graph.to_undirected(), bins=bins, normed=normed)

    def graph_fourier_transform(  # pragma: no cover - heavy
        self, signal: Dict[str, float] | np.ndarray, *, normed: bool = True
    ) -> np.ndarray:
        """Return the graph Fourier transform of ``signal``."""

        from ..analysis.fractal import graph_fourier_transform as _gft

        return _gft(self.graph.to_undirected(), signal, normed=normed)

    def inverse_graph_fourier_transform(  # pragma: no cover - heavy
        self, coeffs: np.ndarray, *, normed: bool = True
    ) -> np.ndarray:
        """Return the inverse graph Fourier transform of ``coeffs``."""

        from ..analysis.fractal import inverse_graph_fourier_transform as _igft

        return _igft(self.graph.to_undirected(), coeffs, normed=normed)

    def persistence_entropy(
        self, dimension: int = 0
    ) -> float:  # pragma: no cover - heavy
        """Return persistence entropy of the graph."""

        from ..analysis.fractal import persistence_entropy as _pe

        g = nx.convert_node_labels_to_integers(self.graph.to_undirected())
        return _pe(g, dimension)

    def persistence_diagrams(
        self, max_dim: int = 2
    ) -> Dict[int, np.ndarray]:  # pragma: no cover - heavy
        """Return persistence diagrams up to ``max_dim``."""

        from ..analysis.fractal import persistence_diagrams as _pd

        g = nx.convert_node_labels_to_integers(self.graph.to_undirected())
        return _pd(g, max_dim)

    def persistence_wasserstein_distance(  # pragma: no cover - heavy
        self, other: nx.Graph, *, dimension: int = 0, order: int = 1
    ) -> float:
        """Return Wasserstein distance to ``other`` graph."""

        from ..analysis.fractal import persistence_wasserstein_distance as _pwd

        g1 = nx.convert_node_labels_to_integers(self.graph.to_undirected())
        g2 = nx.convert_node_labels_to_integers(other)
        return _pwd(g1, g2, dimension=dimension, order=order)

    def topological_signature(
        self, max_dim: int = 1
    ) -> Dict[str, Any]:  # pragma: no cover - heavy
        """Return persistence diagrams and entropies for ``max_dim``."""
        try:
            diags = self.persistence_diagrams(max_dim=max_dim)
        except (RuntimeError, AttributeError):
            diags = {}

        entropies: Dict[int, float] = {}
        for dim in range(max_dim + 1):
            try:
                ent = self.persistence_entropy(dimension=dim)
            except RuntimeError:
                ent = 0.0
            entropies[dim] = ent

        return {
            "diagrams": {d: diag.tolist() for d, diag in diags.items()},
            "entropy": entropies,
        }

    def topological_signature_hash(
        self, max_dim: int = 1
    ) -> str:  # pragma: no cover - heavy
        """Return an MD5 hash of the topological signature."""

        import hashlib
        import json

        signature = self.topological_signature(max_dim=max_dim)
        blob = json.dumps(signature, sort_keys=True).encode()
        return hashlib.md5(blob, usedforsecurity=False).hexdigest()  # nosec B324

    def betti_number(self, dimension: int = 1) -> int:  # pragma: no cover - heavy
        """Return Betti number of ``dimension`` for the graph."""

        if dimension == 0:
            return nx.number_connected_components(self.graph)
        if dimension == 1:
            return (
                self.graph.number_of_edges()
                - self.graph.number_of_nodes()
                + nx.number_connected_components(self.graph)
            )
        # fallback to persistence diagrams for higher dimensions
        try:
            diags = self.persistence_diagrams(max_dim=dimension)
            diag = diags.get(dimension)
        except Exception:
            diag = None
        if diag is None:
            return 0
        return int(sum(np.isinf(diag[:, 1]) == False))

    def compute_fractal_features(  # pragma: no cover - heavy
        self, radii: Iterable[int], *, max_dim: int = 1
    ) -> Dict[str, Any]:
        """Return fractal dimension, optimal radius and topological signature."""

        dim, counts = self.box_counting_dimension(radii)
        from ..analysis.fractal import mdl_optimal_radius

        idx = mdl_optimal_radius(counts)
        radius = counts[idx][0] if counts else 1
        signature = self.topological_signature(max_dim=max_dim)
        return {
            "dimension": dim,
            "radius": radius,
            "counts": counts,
            "signature": signature,
        }

    def fractal_information_metrics(  # pragma: no cover - heavy
        self, radii: Iterable[int], *, max_dim: int = 1
    ) -> Dict[str, Any]:
        """Return fractal dimension and persistence entropies."""

        from ..analysis.fractal import fractal_information_metrics as _fim

        return _fim(self.graph.to_undirected(), radii, max_dim=max_dim)

    def fractal_information_density(  # pragma: no cover - heavy
        self, radii: Iterable[int], *, max_dim: int = 1
    ) -> float:
        """Return fractal information density for ``radii``."""

        from ..analysis.fractal import fractal_information_density as _fid

        return _fid(self.graph.to_undirected(), radii, max_dim=max_dim)

    def fractal_coverage(self) -> float:  # pragma: no cover - heavy
        """Return fraction of nodes with a ``fractal_level`` attribute."""

        from ..analysis.fractal import fractal_level_coverage as _fc

        return _fc(self.graph)

    def ensure_fractal_coverage(  # pragma: no cover - heavy
        self,
        threshold: float,
        radii: Iterable[int],
        *,
        max_levels: int = 5,
    ) -> float:
        """Ensure that at least ``threshold`` nodes are annotated with a fractal level."""

        cov = self.fractal_coverage()
        if cov < threshold:
            self.annotate_fractal_levels(radii, max_levels=max_levels)
            cov = self.fractal_coverage()
        return cov

    def diversification_score(  # pragma: no cover - heavy
        self,
        nodes: Iterable,
        radii: Iterable[int],
        *,
        max_dim: int = 1,
        dimension: int = 0,
    ) -> float:
        """Return diversification score for ``nodes`` against the whole graph."""

        from ..analysis.fractal import diversification_score as _ds

        sub = self.graph.subgraph(nodes)
        return _ds(
            self.graph.to_undirected(),
            sub.to_undirected(),
            radii,
            max_dim=max_dim,
            dimension=dimension,
        )

    def select_diverse_nodes(  # pragma: no cover - heavy
        self, candidates: Iterable[str], count: int, radii: Iterable[int]
    ) -> list[str]:
        """Return ``count`` nodes maximizing the diversification score."""

        chosen: list[str] = []
        for cand in candidates:
            if len(chosen) >= count:
                break
            new_set = chosen + [cand]
            score = self.diversification_score(new_set, radii)
            if not chosen or score > self.diversification_score(chosen, radii):
                chosen.append(cand)
        return chosen

    def hyperbolic_neighbors(
        self, node_id: str, k: int = 5
    ) -> List[tuple[str, float]]:  # pragma: no cover - heavy
        r"""Return ``k`` nearest neighbors using the Poincar\xe9 distance.

        The neighbors are ranked by :math:`d_{\mathbb{B}}(u,v)` between the
        query embedding ``node_id`` and every other node with a
        ``hyperbolic_embedding`` attribute.

        Parameters
        ----------
        node_id:
            Identifier of the query node.
        k:
            Number of closest nodes to return.
        """

        if not self.graph.has_node(node_id):
            raise ValueError(f"Unknown node: {node_id}")
        vec = self.graph.nodes[node_id].get("hyperbolic_embedding")
        if vec is None:
            raise ValueError("node has no hyperbolic embedding")

        from ..analysis.fractal import hyperbolic_nearest_neighbors

        emb = {
            n: self.graph.nodes[n]["hyperbolic_embedding"]
            for n in self.graph.nodes
            if "hyperbolic_embedding" in self.graph.nodes[n]
        }
        neighs = hyperbolic_nearest_neighbors(emb, k=k).get(node_id, [])
        return [(str(n), float(d)) for n, d in neighs]

    def hyperbolic_reasoning(  # pragma: no cover - heavy
        self, start: str, goal: str, *, max_steps: int = 5
    ) -> List[str]:
        r"""Return a greedy path from ``start`` to ``goal`` using hyperbolic distance.

        At each step the neighbor minimizing :math:`d_{\mathbb{B}}` to the goal
        is selected until the path length reaches ``max_steps`` or the goal is
        found.
        """

        emb = {
            n: self.graph.nodes[n]["hyperbolic_embedding"]
            for n in self.graph.nodes
            if "hyperbolic_embedding" in self.graph.nodes[n]
        }
        from ..analysis.fractal import hyperbolic_reasoning as _hr

        path = _hr(emb, start, goal, max_steps=max_steps)
        return [str(n) for n in path]

    def hyperbolic_hypergraph_reasoning(  # pragma: no cover - heavy
        self,
        start: str,
        goal: str,
        *,
        penalty: float = 1.0,
        max_steps: int = 5,
    ) -> List[str]:
        """Return a greedy path using hyperedge embeddings.

        The search expands through hyperedges weighted by ``penalty`` to
        discourage long hops. Distances are measured in the Poincar\xe9 ball and
        the path stops when ``goal`` is reached or ``max_steps`` is exceeded.
        """

        emb = {
            n: self.graph.nodes[n]["hyperbolic_embedding"]
            for n in self.graph.nodes
            if "hyperbolic_embedding" in self.graph.nodes[n]
        }
        hyperedges = [
            n
            for n in self.graph.nodes
            if self.graph.nodes[n].get("type") == "hyperedge"
        ]
        from ..analysis.fractal import hyperbolic_hypergraph_reasoning as _hhr

        path = _hhr(
            emb,
            hyperedges,
            start,
            goal,
            penalty=penalty,
            max_steps=max_steps,
        )
        return [str(n) for n in path]

    def predict_hyperedges(  # pragma: no cover - heavy
        self, *, k: int = 5, threshold: float = 0.8
    ) -> list[tuple[str, list[str]]]:
        """Suggest new hyperedges based on embedding similarity.

        Parameters
        ----------
        k:
            Maximum number of hyperedges to propose.
        threshold:
            Minimum cosine similarity between Hyper-SAGNN embeddings to
            trigger a suggestion.
        """

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        emb = self.compute_hyper_sagnn_embeddings()
        ids = list(emb)
        if not ids:
            return []

        vecs = np.stack([emb[i] for i in ids])
        sim = cosine_similarity(vecs)
        suggestions: list[tuple[str, list[str]]] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if sim[i, j] >= threshold:
                    nodes_i = [v for _, v in self.graph.edges(ids[i])]
                    nodes_j = [v for _, v in self.graph.edges(ids[j])]
                    new_nodes = sorted(set(nodes_i + nodes_j))
                    suggestions.append((f"{ids[i]}_{ids[j]}", new_nodes))
                    if len(suggestions) >= k:
                        return suggestions
        return suggestions

    def hyperbolic_multi_curvature_reasoning(  # pragma: no cover - heavy
        self,
        start: str,
        goal: str,
        *,
        curvatures: Iterable[float],
        weights: Optional[Dict[float, float]] = None,
        max_steps: int = 5,
    ) -> List[str]:
        """Return a greedy path mixing several curvature embeddings.

        Embeddings for each curvature in ``curvatures`` are loaded from node
        attributes named ``hyperbolic_embedding_{c}``. Optionally ``weights``
        can adjust the influence of each curvature when ranking neighbors.
        The path search stops when ``goal`` is reached or ``max_steps`` is hit.
        """

        embeddings: Dict[float, Dict[str, Iterable[float]]] = {}
        for c in curvatures:
            embs: Dict[str, Iterable[float]] = {}
            key = f"hyperbolic_embedding_{c}"
            for n in self.graph.nodes:
                vec = self.graph.nodes[n].get(key)
                if vec is not None:
                    embs[n] = vec
            if embs:
                embeddings[c] = embs

        from ..analysis.fractal import hyperbolic_multi_curvature_reasoning as _hmcr

        path = _hmcr(
            embeddings,
            start,
            goal,
            weights=weights,
            max_steps=max_steps,
        )
        return [str(n) for n in path]

    def dimension_distortion(
        self, radii: Iterable[int]
    ) -> float:  # pragma: no cover - heavy
        """Return difference between graph and embedding fractal dimensions.

        The fractal dimension of the graph is estimated with
        :func:`box_counting_dimension`. The dimension of the hyperbolic
        embeddings stored under ``poincare_embedding`` is computed with
        :func:`embedding_box_counting_dimension`. The distortion is the absolute
        difference

        .. math::

            |D_{\text{graph}} - D_{\text{emb}}|

        where ``D_{\text{graph}}`` is the dimension of the original graph and
        ``D_{\text{emb}}`` the dimension of the embedding space.

        Parameters
        ----------
        radii:
            Iterable of ball radii used for the box counting.

        Returns
        -------
        float
            Absolute difference between the two dimensions.
        """

        graph_dim, _ = self.box_counting_dimension(radii)
        coords = {
            n: self.graph.nodes[n].get("poincare_embedding")
            for n in self.graph.nodes
            if self.graph.nodes[n].get("poincare_embedding") is not None
        }
        if not coords:
            return float("nan")
        from ..analysis.fractal import embedding_box_counting_dimension

        emb_dim, _ = embedding_box_counting_dimension(coords, radii)
        return abs(graph_dim - emb_dim)

    def embedding_box_counting_dimension(  # pragma: no cover - heavy
        self, node_attr: str, radii: Iterable[float]
    ) -> tuple[float, list[tuple[float, int]]]:
        """Return fractal dimension of stored embeddings.

        Parameters
        ----------
        node_attr:
            Node attribute containing embedding vectors.
        radii:
            Iterable of ball radii used for the box counting.

        Returns
        -------
        tuple
            ``(dimension, counts)`` where ``counts`` records the number of
            covering balls for each radius.
        """

        coords = {
            n: self.graph.nodes[n].get(node_attr)
            for n in self.graph.nodes
            if self.graph.nodes[n].get(node_attr) is not None
        }
        if not coords:
            return float("nan"), []
        from ..analysis.fractal import embedding_box_counting_dimension as _ebcd

        return _ebcd(coords, radii)

    def detect_automorphisms(
        self, max_count: int = 10
    ) -> List[Dict[str, str]]:  # pragma: no cover - heavy
        """Return up to ``max_count`` automorphisms as node mappings."""

        from ..analysis.symmetry import automorphisms as _auto

        autos = _auto(self.graph.to_undirected(), max_count=max_count)
        return [{str(k): str(v) for k, v in a.items()} for a in autos]

    def automorphism_group_order(
        self, max_count: int = 100
    ) -> int:  # pragma: no cover - heavy
        """Return the number of automorphisms explored."""

        from ..analysis.symmetry import automorphism_group_order as _ago

        return _ago(self.graph.to_undirected(), max_count=max_count)

    def quotient_by_symmetry(  # pragma: no cover - heavy
        self, *, max_count: int = 10
    ) -> tuple[nx.Graph, Dict[str, int]]:
        """Return quotient graph collapsing automorphism orbits."""

        from ..analysis.symmetry import automorphism_orbits, quotient_graph

        orbits = automorphism_orbits(self.graph.to_undirected(), max_count=max_count)
        q, mapping = quotient_graph(self.graph.to_undirected(), orbits)
        str_map = {str(n): int(m) for n, m in mapping.items()}
        return q, str_map

    def mapper_nerve(
        self, radius: int
    ) -> tuple[nx.Graph, list[set[str]]]:  # pragma: no cover - heavy
        """Return the Mapper nerve of the graph with balls of ``radius``.

        Results are cached per ``radius`` so repeated calls avoid recomputation.
        The cache is reset when :meth:`clear_mapper_cache` is invoked.
        """

        if radius in self._mapper_cache:
            nerve, cover = self._mapper_cache[radius]
            # return copies to prevent accidental mutation
            return nerve.copy(), [set(c) for c in cover]

        from ..analysis.mapper import mapper_nerve as _mn

        nerve, cover = _mn(self.graph.to_undirected(), radius)
        str_cover = [{str(n) for n in c} for c in cover]
        self._mapper_cache[radius] = (nerve.copy(), str_cover)
        return nerve, str_cover

    def clear_mapper_cache(self) -> None:  # pragma: no cover - heavy
        """Clear any cached Mapper nerve computations."""

        self._mapper_cache.clear()

    def inverse_mapper(  # pragma: no cover - heavy
        self, nerve: nx.Graph, cover: Iterable[Iterable[str]]
    ) -> nx.Graph:
        """Reconstruct a graph from ``nerve`` and ``cover`` sets."""

        from ..analysis.mapper import inverse_mapper as _im

        sets = [{n for n in c} for c in cover]
        return _im(nerve, sets)

    def fractalize_level(
        self, radius: int
    ) -> tuple[nx.Graph, Dict[str, int]]:  # pragma: no cover - heavy
        """Return a coarse-grained graph via box covering."""

        from ..analysis.fractal import fractalize_graph as _fg

        coarse, mapping = _fg(self.graph.to_undirected(), radius)
        str_mapping = {str(node): idx for node, idx in mapping.items()}
        return coarse, str_mapping

    def fractalize_optimal(  # pragma: no cover - heavy
        self, radii: Iterable[int]
    ) -> tuple[nx.Graph, Dict[str, int], int]:
        """Coarse-grain the graph using the MDL-optimal radius.

        Parameters
        ----------
        radii:
            Candidate radii for box covering.

        Returns
        -------
        tuple[nx.Graph, Dict[str, int], int]
            The coarse-grained graph, node-to-box mapping, and the radius
            chosen via :func:`mdl_optimal_radius`.
        """

        from ..analysis.fractal import fractalize_optimal as _fo

        coarse, mapping, radius = _fo(self.graph.to_undirected(), radii)
        str_mapping = {str(node): box for node, box in mapping.items()}
        return coarse, str_mapping, radius

    def build_fractal_hierarchy(  # pragma: no cover - heavy
        self, radii: Iterable[int], *, max_levels: int = 5
    ) -> list[tuple[nx.Graph, Dict[str, int], int]]:
        """Return a hierarchy of coarse graphs using MDL-optimal radii."""

        from ..analysis.fractal import build_fractal_hierarchy as _bfh

        hierarchy = _bfh(self.graph.to_undirected(), radii, max_levels=max_levels)
        converted: list[tuple[nx.Graph, Dict[str, int], int]] = []
        for g, mapping, r in hierarchy:
            str_mapping = {str(node): box for node, box in mapping.items()}
            converted.append((g, str_mapping, r))
        return converted

    def build_mdl_hierarchy(  # pragma: no cover - heavy
        self, radii: Iterable[int], *, max_levels: int = 5
    ) -> list[tuple[nx.Graph, Dict[str, int], int]]:
        """Return a hierarchy stopping when MDL increases."""

        from ..analysis.fractal import build_mdl_hierarchy as _bmh

        hierarchy = _bmh(self.graph.to_undirected(), radii, max_levels=max_levels)
        converted: list[tuple[nx.Graph, Dict[str, int], int]] = []
        for g, mapping, r in hierarchy:
            str_mapping = {str(node): box for node, box in mapping.items()}
            converted.append((g, str_mapping, r))
        return converted

    def annotate_fractal_levels(  # pragma: no cover - heavy
        self, radii: Iterable[int], *, max_levels: int = 5
    ) -> None:
        """Annotate nodes with their fractal level using box covering."""
        from ..analysis.fractal import build_fractal_hierarchy as _bfh

        hierarchy = _bfh(self.graph.to_undirected(), radii, max_levels=max_levels)
        current_map = {n: n for n in self.graph.nodes()}
        for level, (_, mapping, _radius) in enumerate(hierarchy, start=1):
            for node, box in list(current_map.items()):
                if box in mapping:
                    current_map[node] = mapping[box]
                    self.graph.nodes[node]["fractal_level"] = level

    def annotate_mdl_levels(
        self, radii: Iterable[int], *, max_levels: int = 5
    ) -> None:  # pragma: no cover - heavy
        """Annotate nodes with levels from an MDL-guided hierarchy.

        The hierarchy is built using :func:`build_mdl_hierarchy` which stops
        coarse-graining once the description length starts increasing. Each node
        receives a ``fractal_level`` attribute corresponding to its depth in the
        resulting hierarchy.
        """

        from ..analysis.fractal import build_mdl_hierarchy as _bmh

        hierarchy = _bmh(self.graph.to_undirected(), radii, max_levels=max_levels)
        current_map = {n: n for n in self.graph.nodes()}
        for level, (_, mapping, _radius) in enumerate(hierarchy, start=1):
            for node, box in list(current_map.items()):
                if box in mapping:
                    current_map[node] = mapping[box]
                    self.graph.nodes[node]["fractal_level"] = level

    # ------------------------------------------------------------------
    # Graph generation utilities
    # ------------------------------------------------------------------

    def generate_graph_rnn_like(
        self, num_nodes: int, num_edges: int
    ) -> nx.Graph:  # pragma: no cover - heavy
        """Return a random graph mimicking GraphRNN output."""

        from ..analysis.generation import generate_graph_rnn_like as _gg

        return _gg(num_nodes, num_edges)

    def generate_graph_rnn(  # pragma: no cover - heavy
        self, num_nodes: int, num_edges: int, *, p: float = 0.5, directed: bool = False
    ) -> nx.Graph:
        """Return a simple sequential GraphRNN-style graph."""

        from ..analysis.generation import generate_graph_rnn as _gr

        return _gr(num_nodes, num_edges, p=p, directed=directed)

    def generate_graph_rnn_stateful(  # pragma: no cover - heavy
        self,
        num_nodes: int,
        num_edges: int,
        *,
        hidden_dim: int = 8,
        seed: int | None = None,
    ) -> nx.DiGraph:
        """Return a directed graph from a tiny stateful RNN generator."""

        from ..analysis.generation import generate_graph_rnn_stateful as _grs

        return _grs(num_nodes, num_edges, hidden_dim=hidden_dim, seed=seed)

    def generate_graph_rnn_sequential(  # pragma: no cover - heavy
        self,
        num_nodes: int,
        num_edges: int,
        *,
        hidden_dim: int = 8,
        seed: int | None = None,
        directed: bool = True,
    ) -> nx.Graph:
        """Return a graph using a sequential RNN-style generator."""

        from ..analysis.generation import generate_graph_rnn_sequential as _grsq

        return _grsq(
            num_nodes,
            num_edges,
            hidden_dim=hidden_dim,
            seed=seed,
            directed=directed,
        )

    def filter_semantic_cycles(  # pragma: no cover - heavy
        self,
        *,
        attr: str = "text",
        stopwords: Iterable[str] | None = None,
        max_len: int = 4,
    ) -> None:
        """Remove trivial GraphRNN cycles based on text labels."""

        from ..analysis.filtering import filter_semantic_cycles as _fsc

        self.graph = _fsc(self.graph, attr=attr, stopwords=stopwords, max_len=max_len)

    # ------------------------------------------------------------------
    # Quality checks inspired by Neo4j GDS
    # ------------------------------------------------------------------

    def quality_check(  # pragma: no cover - heavy
        self,
        *,
        min_component_size: int = 2,
        triangle_threshold: int = 1,
        similarity: float = 0.95,
        link_threshold: float = 0.0,
    ) -> dict[str, int]:
        """Clean the graph using lightweight GDS‑like heuristics."""

        import networkx as nx

        removed_nodes = 0
        removed_edges = 0
        merged_nodes = 0
        added_edges = 0

        # Remove small components
        ug = self.graph.to_undirected()
        for comp in list(nx.connected_components(ug)):
            if len(comp) < min_component_size:
                self.graph.remove_nodes_from(comp)
                removed_nodes += len(comp)

        # Triangle-based edge pruning
        tri = nx.triangles(self.graph.to_undirected())
        for u, v in list(self.graph.edges()):
            if tri.get(u, 0) < triangle_threshold or tri.get(v, 0) < triangle_threshold:
                self.graph.remove_edge(u, v)
                removed_edges += 1

        # Deduplicate similar nodes via Jaccard
        nodes = list(self.graph.nodes())
        for i, u in enumerate(nodes):
            if not self.graph.has_node(u):
                continue
            nu = set(self.graph.neighbors(u))
            for v in nodes[i + 1 :]:
                if not self.graph.has_node(v):
                    continue
                nv = set(self.graph.neighbors(v))
                union = nu | nv
                if not union:
                    continue
                jac = len(nu & nv) / len(union)
                if jac > similarity:
                    for w in nv:
                        if w != u:
                            self.graph.add_edge(u, w)
                    self.graph.remove_node(v)
                    merged_nodes += 1

        # Simple link prediction via Adamic‑Adar
        preds = nx.adamic_adar_index(self.graph.to_undirected())
        for u, v, score in preds:
            if score > link_threshold and not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v, relation="suggested")
                added_edges += 1

        return {
            "removed_nodes": removed_nodes,
            "removed_edges": removed_edges,
            "merged_nodes": merged_nodes,
            "added_edges": added_edges,
        }

    def optimize_topology(  # pragma: no cover - heavy
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
        """Edit ``perception_link`` edges to approach ``target`` topology."""

        from ..analysis.fractal import minimize_bottleneck_distance

        skeleton = nx.Graph()
        skeleton.add_nodes_from(self.graph.nodes())
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == "perception_link":
                skeleton.add_edge(u, v)

        optimized, dist = minimize_bottleneck_distance(
            skeleton,
            target,
            dimension=dimension,
            epsilon=epsilon,
            max_iter=max_iter,
            seed=seed,
        )

        if use_generator and dist > epsilon:
            if use_netgan:
                from ..analysis.generation import generate_netgan_like
            else:
                from ..analysis.generation import generate_graph_rnn_like

            extra = (
                generate_netgan_like(skeleton)
                if use_netgan
                else generate_graph_rnn_like(
                    skeleton.number_of_nodes(), skeleton.number_of_edges()
                )
            )
            node_map = {i: n for i, n in enumerate(skeleton.nodes())}
            for u, v in extra.edges():
                a, b = node_map.get(u), node_map.get(v)
                if a is None or b is None:
                    continue
                if not optimized.has_edge(a, b):
                    optimized.add_edge(a, b)
            dist = minimize_bottleneck_distance(
                optimized,
                target,
                dimension=dimension,
                epsilon=epsilon,
                max_iter=max_iter,
                seed=seed,
            )[1]

        self.graph.remove_edges_from(
            [
                (u, v)
                for u, v, d in self.graph.edges(data=True)
                if d.get("relation") == "perception_link"
            ]
        )
        for u, v in optimized.edges():
            self.graph.add_edge(u, v, relation="perception_link")

        return dist

    def optimize_topology_iterative(  # pragma: no cover - heavy
        self,
        target: nx.Graph,
        *,
        loops: int = 8,
        dimension: int = 1,
        epsilon: float = 0.0,
        max_iter: int = 100,
        seed: int | None = None,
    ) -> float:
        """Repeatedly call :meth:`optimize_topology` until convergence."""

        dist = float("inf")
        for _ in range(max(1, loops)):
            dist = self.optimize_topology(
                target,
                dimension=dimension,
                epsilon=epsilon,
                max_iter=max_iter,
                seed=seed,
                use_generator=True,
            )
            if dist <= epsilon:
                break
        return dist

    def optimize_topology_constrained(  # pragma: no cover - heavy
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
        """Edit edges while preserving fractal dimension within ``delta``."""

        from ..analysis.fractal import box_counting_dimension

        dim_before, _ = box_counting_dimension(self.graph.to_undirected(), radii)
        before_edges = [
            (u, v, d.copy())
            for u, v, d in self.graph.edges(data=True)
            if d.get("relation") == "perception_link"
        ]

        dist = self.optimize_topology(
            target,
            dimension=dimension,
            epsilon=epsilon,
            max_iter=max_iter,
            seed=seed,
            use_generator=use_generator,
            use_netgan=use_netgan,
        )

        dim_after, _ = box_counting_dimension(self.graph.to_undirected(), radii)
        diff = abs(dim_after - dim_before)
        if diff > delta:
            # revert to previous perception edges
            self.graph.remove_edges_from(
                [
                    (u, v)
                    for u, v, d in self.graph.edges(data=True)
                    if d.get("relation") == "perception_link"
                ]
            )
            for u, v, d in before_edges:
                self.graph.add_edge(u, v, **d)
            return float("inf"), diff

        return dist, diff

    def validate_topology(  # pragma: no cover - heavy
        self,
        target: nx.Graph,
        radii: Iterable[int],
        *,
        dimension: int = 1,
    ) -> Tuple[float, float]:
        """Return bottleneck and fractal dimension differences to ``target``."""

        from ..analysis.fractal import bottleneck_distance, box_counting_dimension

        skeleton = nx.Graph()
        skeleton.add_nodes_from(self.graph.nodes())
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation") == "perception_link":
                skeleton.add_edge(u, v)

        dist = bottleneck_distance(skeleton, target, dimension=dimension)
        dim_self, _ = box_counting_dimension(skeleton, radii)
        dim_target, _ = box_counting_dimension(target, radii)
        diff = abs(dim_self - dim_target)
        return dist, diff

    def tpl_correct_graph(  # pragma: no cover - heavy
        self,
        target: nx.Graph,
        *,
        epsilon: float = 0.1,
        dimension: int = 1,
        order: int = 1,
        max_iter: int = 5,
    ) -> Dict[str, float | bool]:
        """Apply Wasserstein-based TPL correction toward ``target``."""

        from ..analysis.tpl import tpl_correct_graph as _tpl

        return _tpl(
            self.graph.to_undirected(),
            target.to_undirected(),
            epsilon=epsilon,
            dimension=dimension,
            order=order,
            max_iter=max_iter,
        )

    def tpl_incremental(  # pragma: no cover - heavy
        self,
        *,
        radius: int = 1,
        dimension: int = 1,
    ) -> Dict[int, np.ndarray]:
        """Compute incremental TPL diagrams for nodes."""

        from ..analysis.tpl_incremental import tpl_incremental as _tpl_inc

        return _tpl_inc(self.graph.to_undirected(), radius=radius, dimension=dimension)

    # ------------------------------------------------------------------
    # Perception helpers
    # ------------------------------------------------------------------

    def apply_perception(  # pragma: no cover - heavy
        self,
        node_id: str,
        new_text: str,
        *,
        perception_id: str | None = None,
        strength: float | None = None,
    ) -> None:
        """Update ``node_id`` text and embedding with perception metadata."""

        if not self.graph.has_node(node_id):
            raise ValueError(f"Unknown node: {node_id}")

        self.graph.nodes[node_id]["text"] = new_text
        if perception_id is not None:
            self.graph.nodes[node_id]["perception_id"] = perception_id
        if strength is not None:
            self.graph.nodes[node_id]["perception_strength"] = float(strength)

        vec = self.index.embed(new_text)
        if vec.size:
            self.graph.nodes[node_id]["embedding"] = vec.tolist()

    def apply_perception_all(  # pragma: no cover - heavy
        self,
        transform: Callable[[str], str],
        *,
        perception_id: str | None = None,
        strength: float | None = None,
    ) -> Dict[object, str]:
        """Apply ``transform`` to every node text and store results."""

        updates: Dict[object, str] = {}
        for n, data in self.graph.nodes(data=True):
            text = data.get("text")
            if text is None:
                continue
            new_text = transform(str(text))
            self.apply_perception(
                n,
                new_text,
                perception_id=perception_id,
                strength=strength,
            )
            updates[n] = new_text
        return updates

    # ------------------------------------------------------------------
    # Structure helpers
    # ------------------------------------------------------------------

    def get_sections_for_document(
        self, doc_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return IDs of sections belonging to ``doc_id`` ordered by sequence."""

        edges = [
            (data.get("sequence", i), tgt)
            for i, (src, tgt, data) in enumerate(self.graph.edges(doc_id, data=True))
            if data.get("relation") == "has_section"
        ]
        edges.sort(key=lambda x: x[0])
        return [t for _, t in edges]

    def get_chunks_for_section(
        self, section_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return chunk IDs under ``section_id`` ordered by sequence."""

        edges = [
            (data.get("sequence", i), tgt)
            for i, (src, tgt, data) in enumerate(
                self.graph.edges(section_id, data=True)
            )
            if data.get("relation") == "under_section"
        ]
        edges.sort(key=lambda x: x[0])
        return [t for _, t in edges]

    def get_section_for_chunk(
        self, chunk_id: str
    ) -> str | None:  # pragma: no cover - heavy
        """Return the section containing ``chunk_id`` if any."""

        for pred in self.graph.predecessors(chunk_id):
            if self.graph.edges[pred, chunk_id].get("relation") == "under_section":
                return pred
        return None

    def get_next_section(
        self, section_id: str
    ) -> str | None:  # pragma: no cover - heavy
        """Return the section that follows ``section_id`` if any."""

        for _, succ, data in self.graph.out_edges(section_id, data=True):
            if data.get("relation") == "next_section":
                return succ
        # fallback using sequence metadata
        for doc_id, _, data in self.graph.in_edges(section_id, data=True):
            if data.get("relation") == "has_section":
                seq = data.get("sequence")
                if seq is None:
                    return None
                sections = self.get_sections_for_document(doc_id)
                if seq + 1 < len(sections):
                    return sections[seq + 1]
        return None

    def get_previous_section(
        self, section_id: str
    ) -> str | None:  # pragma: no cover - heavy
        """Return the section preceding ``section_id`` if any."""

        for pred, _, data in self.graph.in_edges(section_id, data=True):
            if data.get("relation") == "next_section":
                return pred
        # fallback using sequence metadata
        for doc_id, _, data in self.graph.in_edges(section_id, data=True):
            if data.get("relation") == "has_section":
                seq = data.get("sequence")
                if seq is None:
                    return None
                sections = self.get_sections_for_document(doc_id)
                if seq > 0:
                    return sections[seq - 1]
        return None

    def get_next_chunk(self, chunk_id: str) -> str | None:  # pragma: no cover - heavy
        """Return the chunk that follows ``chunk_id`` if any."""

        for _, succ, data in self.graph.out_edges(chunk_id, data=True):
            if data.get("relation") == "next_chunk":
                return succ
        # fallback using sequence metadata
        for doc_id, _, data in self.graph.in_edges(chunk_id, data=True):
            if data.get("relation") == "has_chunk":
                seq = data.get("sequence")
                if seq is None:
                    return None
                chunks = self.get_chunks_for_document(doc_id)
                if seq + 1 < len(chunks):
                    return chunks[seq + 1]
        return None

    def get_previous_chunk(
        self, chunk_id: str
    ) -> str | None:  # pragma: no cover - heavy
        """Return the chunk preceding ``chunk_id`` if any."""

        for pred, _, data in self.graph.in_edges(chunk_id, data=True):
            if data.get("relation") == "next_chunk":
                return pred
        # fallback using sequence metadata
        for doc_id, _, data in self.graph.in_edges(chunk_id, data=True):
            if data.get("relation") == "has_chunk":
                seq = data.get("sequence")
                if seq is None:
                    return None
                chunks = self.get_chunks_for_document(doc_id)
                if seq > 0:
                    return chunks[seq - 1]
        return None

    def get_page_for_chunk(
        self, chunk_id: str
    ) -> int | None:  # pragma: no cover - heavy
        """Return the page number associated with ``chunk_id``."""

        node = self.graph.nodes.get(chunk_id)
        if node and node.get("type") == "chunk":
            return node.get("page")
        return None

    def get_page_for_section(
        self, section_id: str
    ) -> int | None:  # pragma: no cover - heavy
        """Return the starting page recorded for ``section_id``."""

        node = self.graph.nodes.get(section_id)
        if node and node.get("type") == "section":
            return node.get("page")
        return None

    def get_chunks_for_document(
        self, doc_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return all chunk IDs that belong to the given document."""

        edges = [
            (data.get("sequence", i), tgt)
            for i, (src, tgt, data) in enumerate(self.graph.edges(doc_id, data=True))
            if data.get("relation") == "has_chunk"
        ]
        edges.sort(key=lambda x: x[0])
        return [t for _, t in edges]

    def get_images_for_document(
        self, doc_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return image IDs associated with ``doc_id`` ordered by sequence."""

        edges = [
            (data.get("sequence", i), tgt)
            for i, (src, tgt, data) in enumerate(self.graph.edges(doc_id, data=True))
            if data.get("relation") == "has_image"
        ]
        edges.sort(key=lambda x: x[0])
        return [t for _, t in edges]

    def get_captions_for_document(
        self, doc_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return caption IDs linked to ``doc_id`` ordered by sequence."""

        edges = [
            (data.get("sequence", i), tgt)
            for i, (src, tgt, data) in enumerate(self.graph.edges(doc_id, data=True))
            if data.get("relation") == "has_caption"
        ]
        edges.sort(key=lambda x: x[0])
        return [t for _, t in edges]

    def get_caption_for_image(
        self, image_id: str
    ) -> str | None:  # pragma: no cover - heavy
        """Return caption ID describing ``image_id`` if present."""

        for src, tgt, data in self.graph.in_edges(image_id, data=True):
            if data.get("relation") == "caption_of":
                return src
        return None

    def get_audios_for_document(
        self, doc_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return audio IDs associated with ``doc_id`` ordered by sequence."""

        edges = [
            (data.get("sequence", i), tgt)
            for i, (src, tgt, data) in enumerate(self.graph.edges(doc_id, data=True))
            if data.get("relation") == "has_audio"
        ]
        edges.sort(key=lambda x: x[0])
        return [t for _, t in edges]

    def get_atoms_for_document(
        self, doc_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return atom IDs that belong to ``doc_id``."""

        edges = [
            (data.get("sequence", i), tgt)
            for i, (src, tgt, data) in enumerate(self.graph.edges(doc_id, data=True))
            if data.get("relation") == "has_atom"
        ]
        edges.sort(key=lambda x: x[0])
        return [t for _, t in edges]

    def get_molecules_for_document(
        self, doc_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return molecule IDs that belong to ``doc_id``."""

        edges = [
            (data.get("sequence", i), tgt)
            for i, (src, tgt, data) in enumerate(self.graph.edges(doc_id, data=True))
            if data.get("relation") == "has_molecule"
        ]
        edges.sort(key=lambda x: x[0])
        return [t for _, t in edges]

    def get_atoms_for_molecule(
        self, molecule_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return atom IDs contained in ``molecule_id``."""

        edges = [
            (data.get("sequence", i), tgt)
            for i, (src, tgt, data) in enumerate(
                self.graph.edges(molecule_id, data=True)
            )
            if data.get("relation") == "inside"
        ]
        edges.sort(key=lambda x: x[0])
        return [t for _, t in edges]

    def get_facts_for_entity(
        self, entity_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return IDs of facts linked to ``entity_id``."""

        facts = [
            src
            for src, _ in self.graph.in_edges(entity_id)
            if self.graph.edges[src, entity_id].get("relation") in {"subject", "object"}
            and self.graph.nodes[src].get("type") == "fact"
        ]
        return facts

    def get_chunks_for_entity(
        self, entity_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return chunk IDs that mention ``entity_id``."""

        chunks = [
            src
            for src, _ in self.graph.in_edges(entity_id)
            if self.graph.edges[src, entity_id].get("relation") == "mentions"
            and self.graph.nodes[src].get("type") == "chunk"
        ]
        return chunks

    def get_facts_for_chunk(
        self, chunk_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return fact IDs attached to ``chunk_id``."""

        facts = [
            tgt
            for _, tgt, data in self.graph.out_edges(chunk_id, data=True)
            if data.get("relation") == "has_fact"
            and self.graph.nodes[tgt].get("type") == "fact"
        ]
        return facts

    def get_facts_for_document(
        self, doc_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return fact IDs related to any chunk of ``doc_id``."""

        fact_ids: list[str] = []
        for cid in self.get_chunks_for_document(doc_id):
            fact_ids.extend(self.get_facts_for_chunk(cid))
        return fact_ids

    def get_chunks_for_fact(
        self, fact_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return chunk IDs referencing ``fact_id``."""

        chunks = [
            src
            for src, _ in self.graph.in_edges(fact_id)
            if self.graph.edges[src, fact_id].get("relation") == "has_fact"
            and self.graph.nodes[src].get("type") == "chunk"
        ]
        return chunks

    def get_sections_for_fact(
        self, fact_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return section IDs referencing ``fact_id`` via a chunk."""

        sections: list[str] = []
        for cid in self.get_chunks_for_fact(fact_id):
            sec = self.get_section_for_chunk(cid)
            if sec and sec not in sections:
                sections.append(sec)
        return sections

    def get_documents_for_fact(
        self, fact_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return document IDs referencing ``fact_id`` via a chunk."""

        docs: list[str] = []
        for cid in self.get_chunks_for_fact(fact_id):
            doc = self.get_document_for_chunk(cid)
            if doc and doc not in docs:
                docs.append(doc)
        return docs

    def get_pages_for_fact(self, fact_id: str) -> list[int]:  # pragma: no cover - heavy
        """Return page numbers referencing ``fact_id`` via chunks."""

        pages: list[int] = []
        for cid in self.get_chunks_for_fact(fact_id):
            page = self.get_page_for_chunk(cid)
            if page is not None and page not in pages:
                pages.append(page)
        return pages

    def get_entities_for_fact(
        self, fact_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return entity IDs linked as subject or object of ``fact_id``."""

        entities = [
            tgt
            for _, tgt, data in self.graph.out_edges(fact_id, data=True)
            if data.get("relation") in {"subject", "object"}
            and self.graph.nodes[tgt].get("type") == "entity"
        ]
        return entities

    def get_entities_for_chunk(
        self, chunk_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return entity IDs mentioned in ``chunk_id``."""

        entities = [
            tgt
            for _, tgt, data in self.graph.out_edges(chunk_id, data=True)
            if data.get("relation") == "mentions"
            and self.graph.nodes[tgt].get("type") == "entity"
        ]
        return entities

    def get_entities_for_document(
        self, doc_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return entity IDs mentioned anywhere in ``doc_id``."""

        entities: list[str] = [
            tgt
            for _, tgt, data in self.graph.out_edges(doc_id, data=True)
            if data.get("relation") == "mentions"
            and self.graph.nodes[tgt].get("type") == "entity"
        ]
        for cid in self.get_chunks_for_document(doc_id):
            entities.extend(self.get_entities_for_chunk(cid))
        # remove duplicates while preserving order
        seen = set()
        out = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                out.append(e)
        return out

    def get_document_for_section(
        self, section_id: str
    ) -> str | None:  # pragma: no cover - heavy
        """Return the document containing ``section_id`` if any."""

        for pred in self.graph.predecessors(section_id):
            if self.graph.edges[pred, section_id].get("relation") == "has_section":
                return pred
        return None

    def get_document_for_chunk(
        self, chunk_id: str
    ) -> str | None:  # pragma: no cover - heavy
        """Return the document containing ``chunk_id`` if any."""

        for pred in self.graph.predecessors(chunk_id):
            if self.graph.edges[pred, chunk_id].get("relation") == "has_chunk":
                return pred
        section_id = self.get_section_for_chunk(chunk_id)
        if section_id is not None:
            return self.get_document_for_section(section_id)
        return None

    def get_similar_chunks(
        self, chunk_id: str, k: int = 3
    ) -> list[str]:  # pragma: no cover - heavy
        """Return up to ``k`` chunk IDs most similar to ``chunk_id``."""

        if (
            chunk_id not in self.graph.nodes
            or self.graph.nodes[chunk_id].get("type") != "chunk"
        ):
            return []

        text = self.graph.nodes[chunk_id].get("text")
        if not text:
            return []

        indices = self.index.search(text, k=k + 1)
        neighbors: list[str] = []
        for idx in indices:
            nid = self.index.get_id(idx)
            if nid == chunk_id:
                continue
            if self.graph.nodes[nid].get("type") != "chunk":
                continue
            neighbors.append(nid)
            if len(neighbors) >= k:
                break
        return neighbors

    def get_similar_chunks_data(  # pragma: no cover - heavy
        self, chunk_id: str, k: int = 3
    ) -> list[dict[str, Any]]:
        """Return up to ``k`` similar chunk infos for ``chunk_id``."""

        if (
            chunk_id not in self.graph.nodes
            or self.graph.nodes[chunk_id].get("type") != "chunk"
        ):
            return []

        data = []
        neighbors = self.index.nearest_neighbors(k=k, return_distances=True).get(
            chunk_id, []
        )
        for nid, score in neighbors:
            if self.graph.nodes[nid].get("type") != "chunk":
                continue
            data.append(
                {
                    "id": nid,
                    "similarity": score,
                    "text": self.graph.nodes[nid].get("text"),
                    "document": self.get_document_for_chunk(nid),
                }
            )
        return data

    def get_chunk_neighbors(
        self, k: int = 3
    ) -> dict[str, list[str]]:  # pragma: no cover - heavy
        """Return the ``k`` nearest chunk neighbors for each chunk."""

        raw = self.index.nearest_neighbors(k)
        neighbors: dict[str, list[str]] = {}
        for cid, neigh in raw.items():
            if self.graph.nodes.get(cid, {}).get("type") != "chunk":
                continue
            filtered = [
                n for n in neigh if self.graph.nodes.get(n, {}).get("type") == "chunk"
            ]
            neighbors[cid] = filtered
        return neighbors

    def get_chunk_neighbors_data(
        self, k: int = 3
    ) -> dict[str, list[dict[str, Any]]]:  # pragma: no cover - heavy
        """Return neighbor information for each chunk."""

        raw = self.index.nearest_neighbors(k, return_distances=True)
        out: dict[str, list[dict[str, Any]]] = {}
        for cid, neigh in raw.items():
            if self.graph.nodes.get(cid, {}).get("type") != "chunk":
                continue
            data: list[dict[str, Any]] = []
            for nid, score in neigh:
                if self.graph.nodes.get(nid, {}).get("type") != "chunk":
                    continue
                data.append(
                    {
                        "id": nid,
                        "similarity": score,
                        "text": self.graph.nodes[nid].get("text"),
                        "document": self.get_document_for_chunk(nid),
                    }
                )
            out[cid] = data
        return out

    def get_similar_sections(
        self, section_id: str, k: int = 3
    ) -> list[str]:  # pragma: no cover - heavy
        """Return up to ``k`` section IDs similar to ``section_id``."""

        if (
            section_id not in self.graph.nodes
            or self.graph.nodes[section_id].get("type") != "section"
        ):
            return []

        title = self.graph.nodes[section_id].get("title")
        if not title:
            return []

        indices = self.index.search(title, k=k + 1)
        neighbors: list[str] = []
        for idx in indices:
            sid = self.index.get_id(idx)
            if sid == section_id:
                continue
            if self.graph.nodes[sid].get("type") != "section":
                continue
            neighbors.append(sid)
            if len(neighbors) >= k:
                break
        return neighbors

    def get_similar_documents(
        self, doc_id: str, k: int = 3
    ) -> list[str]:  # pragma: no cover - heavy
        """Return up to ``k`` document IDs similar to ``doc_id``."""

        if (
            doc_id not in self.graph.nodes
            or self.graph.nodes[doc_id].get("type") != "document"
        ):
            return []

        text = self.graph.nodes[doc_id].get("text")
        if not text:
            return []

        indices = self.index.search(text, k=k + 1)
        neighbors: list[str] = []
        for idx in indices:
            did = self.index.get_id(idx)
            if did == doc_id:
                continue
            if self.graph.nodes[did].get("type") != "document":
                continue
            neighbors.append(did)
            if len(neighbors) >= k:
                break
        return neighbors

    def get_chunk_context(  # pragma: no cover - heavy
        self, chunk_id: str, before: int = 1, after: int = 1
    ) -> list[str]:
        """Return chunk IDs surrounding ``chunk_id`` including itself."""

        if (
            chunk_id not in self.graph.nodes
            or self.graph.nodes[chunk_id].get("type") != "chunk"
        ):
            return []

        context: list[str] = [chunk_id]

        current = chunk_id
        for _ in range(before):
            prev = self.get_previous_chunk(current)
            if prev is None:
                break
            context.insert(0, prev)
            current = prev

        current = chunk_id
        for _ in range(after):
            nxt = self.get_next_chunk(current)
            if nxt is None:
                break
            context.append(nxt)
            current = nxt

        return context

    def get_documents_for_entity(
        self, entity_id: str
    ) -> list[str]:  # pragma: no cover - heavy
        """Return document IDs where ``entity_id`` is mentioned."""

        docs: set[str] = set()
        # direct links from document to entity
        for src, _, data in self.graph.in_edges(entity_id, data=True):
            if (
                data.get("relation") == "mentions"
                and self.graph.nodes[src].get("type") == "document"
            ):
                docs.add(src)
        # links via chunks
        for chunk_id in self.get_chunks_for_entity(entity_id):
            for pred in self.graph.predecessors(chunk_id):
                if self.graph.edges[pred, chunk_id].get("relation") == "has_chunk":
                    docs.add(pred)
        return list(docs)

    def get_pages_for_entity(
        self, entity_id: str
    ) -> list[int]:  # pragma: no cover - heavy
        """Return page numbers mentioning ``entity_id`` via chunks."""

        pages: list[int] = []
        for cid in self.get_chunks_for_entity(entity_id):
            page = self.get_page_for_chunk(cid)
            if page is not None and page not in pages:
                pages.append(page)
        return pages

    # ------------------------------------------------------------------
    # Fact helpers
    # ------------------------------------------------------------------

    def find_facts(  # pragma: no cover - heavy
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
    ) -> list[str]:
        """Return fact IDs matching the provided components."""

        matches: list[str] = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") != "fact":
                continue
            if subject is not None and data.get("subject") != subject:
                continue
            if predicate is not None and data.get("predicate") != predicate:
                continue
            if object is not None and data.get("object") != object:
                continue
            matches.append(node)
        return matches

    def haa_link_score(
        self, driver: "Driver", a_id: str, b_id: str
    ) -> float | None:  # pragma: no cover - heavy
        """Return Hyper-Adamic–Adar score between two nodes from Neo4j.

        The Cypher query uses a composite index on ``(startNodeId, endNodeId)``
        for relationship ``SUGGESTED_HYPER_AA``. The index is only effective if
        start and end internal IDs are materialised first, hence the
        ``WITH id(a) AS id1, id(b) AS id2`` clause.
        """

        query = (
            "PROFILE MATCH (a {id:$u}) MATCH (b {id:$v}) "
            "WITH id(a) AS ida, id(b) AS idb "
            "WITH CASE WHEN ida <= idb THEN ida ELSE idb END AS id1, "
            "CASE WHEN ida <= idb THEN idb ELSE ida END AS id2 "
            "MATCH ()-[r:SUGGESTED_HYPER_AA]-() "
            "WHERE r.startNodeId = id1 AND r.endNodeId = id2 "
            "RETURN r.score AS score"
        )
        with driver.session() as session:
            rec = session.run(query, u=a_id, v=b_id).single()
            return float(rec["score"]) if rec else None

    def fact_confidence(  # pragma: no cover - heavy
        self, subject: str, predicate: str, object: str, *, max_hops: int = 3
    ) -> float:
        """Return a confidence score for ``subject`` ``predicate`` ``object``.

        The score is ``1.0`` when a direct edge or fact exists. If the entities
        are connected by a short path (``<= max_hops``) but not directly linked
        the score is ``0.4`` to indicate indirect evidence. Otherwise ``0.0`` is
        returned when no connecting path exists.
        """

        if self.graph.has_edge(subject, object) and (
            self.graph.edges[subject, object].get("relation") == predicate
        ):
            return 1.0

        if self.find_facts(subject=subject, predicate=predicate, object=object):
            return 1.0

        if not self.graph.has_node(subject) or not self.graph.has_node(object):
            return 0.0

        ug = self.graph.to_undirected()
        try:
            length = nx.shortest_path_length(ug, subject, object)
        except nx.NetworkXNoPath:
            return 0.0

        if length == 1:
            # direct relation exists but with a different predicate
            return 0.4

        return 0.4 if length <= max_hops else 0.0

    def verify_statements(  # pragma: no cover - heavy
        self, statements: Iterable[tuple[str, str, str]], *, max_hops: int = 3
    ) -> float:
        """Return average confidence over ``statements``.

        Each statement is a ``(subject, predicate, object)`` triple. The method
        calls :meth:`fact_confidence` for every triple and returns the mean
        confidence score. ``0.0`` is returned when no statements are supplied.
        """

        scores = [
            self.fact_confidence(s, p, o, max_hops=max_hops) for s, p, o in statements
        ]
        return float(sum(scores) / len(scores)) if scores else 0.0

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - heavy
        """Serialize the graph to a dictionary."""

        return {
            "nodes": [{"id": n, **data} for n, data in self.graph.nodes(data=True)],
            "edges": [
                {"source": u, "target": v, **data}
                for u, v, data in self.graph.edges(data=True)
            ],
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any]
    ) -> "KnowledgeGraph":  # pragma: no cover - heavy
        """Rebuild a :class:`KnowledgeGraph` from ``data``."""

        kg = cls()
        for node in data.get("nodes", []):
            node_id = node.pop("id")
            kg.graph.add_node(node_id, **node)
            if node.get("type") == "chunk" and "text" in node:
                kg.index.add(node_id, node["text"])
        for edge in data.get("edges", []):
            src = edge.pop("source")
            tgt = edge.pop("target")
            kg.graph.add_edge(src, tgt, **edge)
        kg.index.build()
        return kg

    def to_text(self) -> str:  # pragma: no cover - heavy
        """Return all chunk texts concatenated in document order."""

        parts: list[str] = []
        docs = [
            n for n, d in self.graph.nodes(data=True) if d.get("type") == "document"
        ]
        docs.sort()
        for doc_id in docs:
            chunks = self.get_chunks_for_document(doc_id)
            if chunks:
                for cid in chunks:
                    text = self.graph.nodes[cid].get("text")
                    if text:
                        parts.append(text)
            else:
                text = self.graph.nodes[doc_id].get("text")
                if text:
                    parts.append(text)
        return "\n\n".join(parts)

    def set_property(  # pragma: no cover - heavy
        self,
        key: str,
        value: Any,
        *,
        driver: Driver | None = None,
        dataset: str | None = None,
    ) -> None:
        """Store a global graph property and optionally persist to Neo4j."""

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid property name: {key}")

        self.graph.graph[key] = value

        if driver is not None and Driver is not None:
            query = f"MERGE (m:GraphMeta {{dataset:$ds}}) SET m.{key}=$val"
            with driver.session() as session:
                session.run(query, ds=dataset or "default", val=value)

    def create_fractal_index(self, driver: Driver) -> None:
        """Ensure full-text index on ``Subgraph.fractal_dim`` exists."""

        query = (
            "CALL db.index.fulltext.createNodeIndex("
            '"idx_fractal", ["Subgraph"], ["fractal_dim"])'
        )
        with driver.session() as session:
            session.run(query)

    # ------------------------------------------------------------------
    # Neo4j helpers
    # ------------------------------------------------------------------

    def to_neo4j(  # pragma: no cover - heavy
        self,
        driver: Driver,
        *,
        clear: bool = True,
        dataset: str | None = None,
    ) -> None:
        """Persist the graph to a Neo4j database.

        Parameters
        ----------
        driver: Driver
            Neo4j driver instance.
        clear: bool, optional
            Whether to remove existing nodes before writing.
        """

        def _write(tx):
            if dataset:
                if clear:
                    tx.run(
                        "MATCH (n {dataset:$dataset}) DETACH DELETE n",
                        dataset=dataset,
                    )
            elif clear:
                tx.run("MATCH (n) DETACH DELETE n")

            for n, data in self.graph.nodes(data=True):
                label = data.get("type", "Node").capitalize()
                props = {k: v for k, v in data.items() if k != "type"}
                if dataset:
                    props["dataset"] = dataset
                if label == "Document" and props.get("uid"):
                    tx.run(
                        f"MERGE (m:{label} {{uid:$uid{', dataset:$dataset' if dataset else ''}}}) "
                        "ON CREATE SET m.first_seen=timestamp() "
                        "SET m += $props, m.last_ingested=timestamp()",
                        uid=props["uid"],
                        **({"dataset": dataset} if dataset else {}),
                        props=props,
                    )
                else:
                    tx.run(
                        f"MERGE (m:{label} {{id:$id{', dataset:$dataset' if dataset else ''}}}) SET m += $props",
                        id=n,
                        **({"dataset": dataset} if dataset else {}),
                        props=props,
                    )
            for u, v, edata in self.graph.edges(data=True):
                rel = edata.get("relation", "RELATED_TO").upper()
                props = {k: v for k, v in edata.items() if k != "relation"}
                if dataset:
                    props["dataset"] = dataset
                tx.run(
                    f"MATCH (a {{id:$u{', dataset:$dataset' if dataset else ''}}}), (b {{id:$v{', dataset:$dataset' if dataset else ''}}}) MERGE (a)-[r:{rel}]->(b) SET r += $props",
                    u=u,
                    v=v,
                    **({"dataset": dataset} if dataset else {}),
                    props=props,
                )

        with driver.session() as session:
            session.execute_write(_write)

    @classmethod
    def from_neo4j(  # pragma: no cover - heavy
        cls,
        driver: Driver,
        *,
        dataset: str | None = None,
    ) -> "KnowledgeGraph":
        """Load a :class:`KnowledgeGraph` from a Neo4j database."""

        kg = cls()

        def _read(tx):
            if dataset:
                nodes = tx.run(
                    "MATCH (n {dataset:$dataset}) RETURN n, labels(n)[0] AS label",
                    dataset=dataset,
                )
            else:
                nodes = tx.run("MATCH (n) RETURN n, labels(n)[0] AS label")
            for record in nodes:
                props = record["n"]
                node_id = props.pop("id")
                node_type = props.pop("type", record["label"]).lower()
                kg.graph.add_node(node_id, type=node_type, **props)
                if node_type == "chunk" and "text" in props:
                    kg.index.add(node_id, props["text"])
            if dataset:
                edges = tx.run(
                    "MATCH (a {dataset:$dataset})-[r]->(b {dataset:$dataset}) RETURN a.id AS src, type(r) AS rel, b.id AS tgt, r as rel_props",
                    dataset=dataset,
                )
            else:
                edges = tx.run(
                    "MATCH (a)-[r]->(b) RETURN a.id AS src, type(r) AS rel, b.id AS tgt, r as rel_props"
                )
            for record in edges:
                props = dict(record["rel_props"])
                kg.graph.add_edge(
                    record["src"],
                    record["tgt"],
                    relation=record["rel"].lower(),
                    **props,
                )

        with driver.session() as session:
            session.execute_read(_read)
        kg.index.build()
        return kg

    def gds_quality_check(  # pragma: no cover - heavy
        self,
        driver: Driver,
        *,
        dataset: str | None = None,
        min_component_size: int | None = None,
        similarity_threshold: float | None = None,
        triangle_threshold: int | None = None,
        link_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Run Neo4j GDS quality checks and cleanup.

        Parameters
        ----------
        driver:
            Neo4j driver instance.
        dataset:
            Optional dataset label used to filter nodes.
        min_component_size:
            Components smaller than this size are removed. ``None`` loads the
            default from ``configs/default.yaml`` (``cleanup.k_min``) so autotuning
            can adapt it.
        similarity_threshold:
            Cosine similarity above which nodes are flagged as duplicates.
            ``None`` loads ``cleanup.sigma`` from the config.
        triangle_threshold:
            Nodes with fewer triangles than this value are incident to edges
            pruned during cleanup. ``None`` loads ``cleanup.tau`` from the config.
        link_threshold:
            Minimum score for a suggested link to be added to the in-memory
            graph.

        Returns
        -------
        dict
            Summary dictionary with lists of removed nodes, duplicate pairs,
            suggested links and hub nodes.
        """

        cfg_vals = get_cleanup_cfg()
        lp_sigma = float(cfg_vals.get("lp_sigma", 0.5))
        lp_topk = int(cfg_vals.get("lp_topk", 50))
        hub_deg = int(cfg_vals.get("hub_deg", 500))

        if min_component_size is None:
            min_component_size = int(cfg_vals.get("k_min", 2))
        if similarity_threshold is None:
            similarity_threshold = float(cfg_vals.get("sigma", 0.95))
        if triangle_threshold is None:
            triangle_threshold = int(cfg_vals.get("tau", 1))

        node_query = (
            "MATCH (n"
            + (" {dataset:$dataset}" if dataset else "")
            + ") RETURN id(n) AS id"
        )
        rel_query = (
            "MATCH (n"
            + (" {dataset:$dataset}" if dataset else "")
            + ")-[r]->(m"
            + (" {dataset:$dataset}" if dataset else "")
            + ") RETURN id(n) AS source, id(m) AS target"
        )
        params = {"dataset": dataset} if dataset else {}

        rel_proj = {
            "INSIDE": {"type": "HAS_CHUNK", "orientation": "NATURAL"},
            "NEXT": {"type": "NEXT_CHUNK", "orientation": "NATURAL"},
            "HYPER": {
                "type": "HYPER",
                "orientation": "UNDIRECTED",
                "aggregation": "MAX",
            },
        }

        with driver.session() as session:
            session.run("CALL gds.graph.drop('kg_qc', false)")
            session.run(
                "CALL gds.graph.project.cypher('kg_qc', $nodeQuery, $relQuery, {relationshipProjection:$relProj})",
                nodeQuery=node_query,
                relQuery=rel_query,
                relProj=rel_proj,
                **params,
            )

            comps = session.run(
                "CALL gds.wcc.stream('kg_qc') YIELD nodeId, componentId"
            )
            groups: Dict[int, List[int]] = {}
            for rec in comps:
                groups.setdefault(rec["componentId"], []).append(rec["nodeId"])
            removed: List[int] = []
            for nodes in groups.values():
                if len(nodes) < min_component_size:
                    for n in nodes:
                        session.run("MATCH (n) WHERE id(n)=$id DETACH DELETE n", id=n)
                        removed.append(n)

            duplicates = []
            sim = session.run(
                "CALL gds.nodeSimilarity.stream('kg_qc') YIELD node1, node2, similarity"
            )
            for rec in sim:
                if rec["similarity"] >= similarity_threshold:
                    duplicates.append((rec["node1"], rec["node2"], rec["similarity"]))

            links = session.run(
                "CALL gds.alpha.linkprediction.adamicAdar.stream('kg_qc') "
                "YIELD sourceNodeId, targetNodeId, score ORDER BY score DESC LIMIT 5"
            )
            id_cache: Dict[int, str] = {}

            def _name(nid: int) -> str | None:
                if nid not in id_cache:
                    rec = session.run(
                        "MATCH (n) WHERE id(n)=$id RETURN n.id AS name",
                        id=nid,
                    ).single()
                    id_cache[nid] = rec["name"] if rec else None
                return id_cache[nid]

            suggestions = []
            for rec in links:
                src = rec["sourceNodeId"]
                tgt = rec["targetNodeId"]
                score = float(rec["score"])
                suggestions.append((src, tgt, score))
                if score >= link_threshold:
                    u = _name(src)
                    v = _name(tgt)
                    if u and v and not self.graph.has_edge(u, v):
                        self.graph.add_edge(u, v, relation="suggested", score=score)

            pa_links = session.run(
                "CALL gds.alpha.linkprediction.preferentialAttachment.stream('kg_qc') "
                "YIELD sourceNodeId, targetNodeId, score ORDER BY score DESC LIMIT 5"
            )
            for rec in pa_links:
                src = rec["sourceNodeId"]
                tgt = rec["targetNodeId"]
                score = float(rec["score"])
                suggestions.append((src, tgt, score))
                if score >= link_threshold:
                    u = _name(src)
                    v = _name(tgt)
                    if u and v and not self.graph.has_edge(u, v):
                        self.graph.add_edge(u, v, relation="suggested", score=score)

            session.run(
                "CALL gds.alpha.hypergraph.linkprediction.adamicAdar.write("
                "'kg_qc', {writeRelationshipType:'SUGGESTED_HYPER_AA',"
                " writeProperty:'score', topK:$topk, relationshipProjection:$relProj})",
                topk=lp_topk,
                relProj=rel_proj,
            )
            # Canonical pair orientation avoids duplicates from reversed order
            session.run(
                "MATCH (a)-[r:SUGGESTED_HYPER_AA]->(b) "
                "WITH CASE WHEN r.startNodeId <= r.endNodeId THEN r.startNodeId ELSE r.endNodeId END AS s, "
                "CASE WHEN r.startNodeId <= r.endNodeId THEN r.endNodeId ELSE r.startNodeId END AS t, r "
                "SET r.startNodeId = s, r.endNodeId = t"
            )
            session.run(
                "MATCH ()-[r:SUGGESTED_HYPER_AA]->() "
                "WITH r.startNodeId AS s, r.endNodeId AS t, collect(r) AS rs "
                "WHERE size(rs) > 1 "
                "FOREACH (x IN tail(rs) | DELETE x)"
            )
            session.run(
                "CREATE INDEX haa_pair IF NOT EXISTS "
                "FOR ()-[r:SUGGESTED_HYPER_AA]-() "
                "ON (r.startNodeId, r.endNodeId)"
            )
            session.run(
                "CREATE INDEX haa_score_index IF NOT EXISTS FOR ()-[r:SUGGESTED_HYPER_AA]-() ON (r.score)"
            )
            session.run(
                "MATCH ()-[r:SUGGESTED_HYPER_AA]->() WHERE r.score <= $th DELETE r",
                th=lp_sigma,
            )
            count_rec = session.run(
                "MATCH ()-[r:SUGGESTED_HYPER_AA]->() RETURN count(r) AS c"
            ).single()
            try:
                from ..analysis.monitoring import haa_edges_total as _haa_counter

                if count_rec and _haa_counter is not None:
                    _haa_counter.inc(int(count_rec["c"]))
            except Exception:  # pragma: no cover - optional metrics
                pass

            deg_records = list(
                session.run("CALL gds.degree.stream('kg_qc') YIELD nodeId, score")
            )
            bet_records = list(
                session.run("CALL gds.betweenness.stream('kg_qc') YIELD nodeId, score")
            )
            tri_records = list(
                session.run(
                    "CALL gds.triangleCount.stream('kg_qc', {relationshipWeightProperty:'attention'}) "
                    "YIELD nodeId, triangleCount"
                )
            )
            tri_map = {r["nodeId"]: r["triangleCount"] for r in tri_records}
            hubs: List[int] = []
            hubs.extend(r["nodeId"] for r in deg_records if r["score"] >= hub_deg)
            hubs.extend(
                r["nodeId"]
                for r in bet_records
                if r["score"] >= hub_deg and r["nodeId"] not in hubs
            )

            weak_links: List[tuple[int, int]] = []
            triangles_removed = 0
            edge_records = list(
                session.run(
                    "MATCH (a)-[r]->(b) RETURN id(a) AS src, id(b) AS tgt, r.attention AS attention"
                    + (
                        ""
                        if not dataset
                        else " WHERE a.dataset=$dataset AND b.dataset=$dataset"
                    ),
                    **params,
                )
            )
            att_values = [
                rec.get("attention")
                for rec in edge_records
                if rec.get("attention") is not None
            ]
            median_att = float(np.median(att_values)) if att_values else 0.0
            for rec in edge_records:
                s = rec["src"]
                t = rec["tgt"]
                att = rec.get("attention", 0.0)
                if (
                    tri_map.get(s, 0) < triangle_threshold
                    or tri_map.get(t, 0) < triangle_threshold
                ) and att < median_att:
                    session.run(
                        "MATCH (a)-[r]->(b) WHERE id(a)=$s AND id(b)=$t DELETE r",
                        s=s,
                        t=t,
                    )
                    weak_links.append((s, t))
                    triangles_removed += 1

            session.run("CALL gds.graph.drop('kg_qc')")

        return {
            "removed_nodes": removed,
            "duplicates": duplicates,
            "suggested_links": suggestions,
            "hubs": hubs,
            "weak_links": weak_links,
            "triangles_removed": triangles_removed,
        }

    def node_similarity(  # pragma: no cover - heavy
        self,
        driver: Driver,
        node_id: str,
        *,
        dataset: str | None = None,
        threshold: float = 0.95,
    ) -> List[tuple[str, float]]:
        """Return nodes similar to ``node_id`` using Neo4j GDS.

        Parameters
        ----------
        driver:
            Active Neo4j :class:`Driver` instance used for queries.
        node_id:
            Identifier of the node whose neighbours are sought.
        dataset:
            Optional dataset label to scope the projection; when ``None`` all
            nodes in the database are considered.
        threshold:
            Minimum cosine/Jaccard similarity for a result to be returned.

        The method creates a temporary graph projection of the dataset and runs
        ``gds.nodeSimilarity``. Only pairs involving ``node_id`` with similarity
        greater than ``threshold`` are collected and returned.
        """

        node_query = (
            "MATCH (n" + (" {dataset:$dataset}" if dataset else "") + ") "
            "RETURN id(n) AS id, n.id AS name"
        )
        rel_query = (
            "MATCH (n"
            + (" {dataset:$dataset}" if dataset else "")
            + ")-[r]->(m"
            + (" {dataset:$dataset}" if dataset else "")
            + ") RETURN id(n) AS source, id(m) AS target"
        )
        params = {"dataset": dataset} if dataset else {}

        with driver.session() as session:
            session.run("CALL gds.graph.drop('kg_sim', false)")
            session.run(
                "CALL gds.graph.project.cypher('kg_sim', $nQuery, $rQuery)",
                nQuery=node_query,
                rQuery=rel_query,
                **params,
            )

            rec = session.run(
                "MATCH (n {id:$node_id"
                + (", dataset:$dataset" if dataset else "")
                + "}) "
                "RETURN id(n) AS nid",
                node_id=node_id,
                **params,
            ).single()
            if not rec:
                session.run("CALL gds.graph.drop('kg_sim')")
                return []
            nid = rec["nid"]

            results = session.run(
                "CALL gds.nodeSimilarity.stream('kg_sim') "
                "YIELD node1, node2, similarity "
                "WHERE (node1=$nid OR node2=$nid) AND similarity >= $thres "
                "RETURN node1, node2, similarity",
                nid=nid,
                thres=threshold,
            )
            matches: List[tuple[str, float]] = []
            for r in results:
                other = r["node2"] if r["node1"] == nid else r["node1"]
                node_rec = session.run(
                    "MATCH (n) WHERE id(n)=$id RETURN n.id AS name",
                    id=other,
                ).single()
                if node_rec:
                    matches.append((node_rec["name"], r["similarity"]))

            session.run("CALL gds.graph.drop('kg_sim')")

        return matches
