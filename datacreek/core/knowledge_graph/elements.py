"""Graph element management mixin."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

if TYPE_CHECKING:
    from ...utils.retrieval import EmbeddingIndex

logger = logging.getLogger(__name__)

try:  # optional dependency for graph operations
    import networkx as nx
except Exception:  # pragma: no cover - minimal stub when networkx missing
    import types

    class _GraphStub:
        def __init__(self, *a, **k) -> None:
            pass

    nx = types.SimpleNamespace(DiGraph=_GraphStub, Graph=_GraphStub)  # type: ignore

try:
    from neo4j import Driver
except Exception:  # pragma: no cover - optional dependency for tests
    Driver = object  # type: ignore


class ElementMixin:
    """Mixin for managing graph elements (nodes and edges)."""

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
            from ...utils.text import detect_language

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
        """Create a relation between ``node_id`` and ``entity_id``."""

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
        sequence: int | None = None,
    ) -> None:
        """Insert an audio node linked to ``doc_id``."""

        if source is None:
            source = self.graph.nodes[doc_id].get("source")
        if self.graph.has_node(audio_id):
            raise ValueError(f"Audio already exists: {audio_id}")

        audios = self.get_audios_for_document(doc_id)
        if sequence is None:
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
                from ...utils.text import detect_language

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
        """Insert a simplex node linked to ``node_ids``."""

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
                logger.exception("Failed to delete document %s from Neo4j", doc_id)

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

    def clean_chunk_texts(self) -> int:  # pragma: no cover - heavy
        """Normalize text in all chunk nodes.

        Returns
        -------
        int
            Number of chunks modified.
        """
        try:
            from ...utils.text import clean_text
        except ImportError:
            # Fallback if utils not available
            def clean_text(t): return " ".join(t.split())

        count = 0
        for n, data in self.graph.nodes(data=True):
            if data.get("type") != "chunk":
                continue
            text = data.get("text")
            if not text:
                continue
            
            cleaned = clean_text(text)
            if cleaned != text:
                self.graph.nodes[n]["text"] = cleaned
                count += 1
        
        return count
