"""Search and retrieval mixin."""
from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from neo4j import Driver
    import numpy as np

    from ...utils.retrieval import EmbeddingIndex

logger = logging.getLogger(__name__)

try:
    from ...analysis.monitoring import ann_backend, ann_latency
except ImportError:
    ann_backend = None
    ann_latency = None


class SearchMixin:
    """Mixin for search, retrieval and similarity operations."""

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
                    if hasattr(self, "get_chunks_for_document"):
                        doc_chunks = self.get_chunks_for_document(doc)
                        if nid in doc_chunks:
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
        """Return node IDs by combining lexical and embedding search."""

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
                    "document": getattr(self, "get_document_for_chunk", lambda x: None)(
                        nid
                    ),
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
                        "document": getattr(
                            self, "get_document_for_chunk", lambda x: None
                        )(nid),
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
        get_prev = getattr(self, "get_previous_chunk", None)
        get_next = getattr(self, "get_next_chunk", None)

        if get_prev:
            current = chunk_id
            for _ in range(before):
                prev = get_prev(current)
                if prev is None:
                    break
                context.insert(0, prev)
                current = prev

        if get_next:
            current = chunk_id
            for _ in range(after):
                nxt = get_next(current)
                if nxt is None:
                    break
                context.append(nxt)
                current = nxt

        return context

    # ------------------------------------------------------------------
    # Hybrid Search and FAISS
    # ------------------------------------------------------------------

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
        """Return a hybrid similarity score between ``src`` and ``tgt``."""

        from ...analysis import hybrid_score as _hs

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

    def build_faiss_index(  # pragma: no cover - heavy
        self, node_attr: str = "embedding", *, method: str = "flat"
    ) -> None:
        """Build a FAISS index from node embeddings."""

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
            if ann_backend is not None:
                if method == "faiss_gpu_ivfpq":
                    ann_backend.set(3)
                elif method == "hnsw":
                    ann_backend.set(2)
                else:
                    ann_backend.set(1)
        except Exception as exc:
            logger.debug("ANN backend unavailable for %s: %s", method, exc)

    def search_faiss(  # pragma: no cover - heavy
        self,
        vector: Iterable[float],
        k: int = 5,
        *,
        adaptive: bool = False,
        latency_threshold: float = 0.1,
    ) -> list[str]:
        """Return ``k`` nearest nodes using the FAISS index."""

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
        """Return top ``k`` nodes by the multi-view similarity using ANN."""

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
            from ...analysis import hybrid_score as _hs

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
        """Return Cypher query results seeded by ANN search."""

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
        """Return mean recall@k using hybrid similarity search."""

        from ...analysis.autotune import recall_at_k as _recall

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
        """Return chunk IDs related to a query and expand via graph links."""

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
        """Return enriched search results expanding through graph links."""

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
