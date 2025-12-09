"""Embedding-based retrieval utilities using TF-IDF."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

try:
    import hnswlib  # type: ignore
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover - optional dependency
    np = None
    TfidfVectorizer = None
    NearestNeighbors = None
    hnswlib = None  # type: ignore


class EmbeddingIndex:
    """Maintain a simple in-memory retrieval index."""

    def __init__(self, *, use_hnsw: bool = False) -> None:
        self.texts: List[str] = []
        self.ids: List[str] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix: Optional[np.ndarray] = None
        self._nn: Optional[NearestNeighbors] = None
        self._hnsw: Optional["hnswlib.Index"] = None
        self.use_hnsw = use_hnsw and hnswlib is not None

    def add(self, chunk_id: str, text: str) -> None:
        self.ids.append(chunk_id)
        self.texts.append(text)

    def remove(self, chunk_id: str) -> None:
        """Remove a chunk from the index."""
        if chunk_id not in self.ids:
            return
        idx = self.ids.index(chunk_id)
        del self.ids[idx]
        del self.texts[idx]
        self._vectorizer = None
        self._matrix = None
        self._nn = None
        self._hnsw = None

    def build(self) -> None:
        if not self.texts:
            return
        global TfidfVectorizer, NearestNeighbors, np
        if TfidfVectorizer is None or NearestNeighbors is None or np is None:
            try:
                import numpy as _np
                from sklearn.feature_extraction.text import TfidfVectorizer as _TF
                from sklearn.neighbors import NearestNeighbors as _NN

                np = _np
                TfidfVectorizer = _TF
                NearestNeighbors = _NN
            except Exception as exc:
                # Fall back to simple count vectors if scikit-learn is missing
                np = _np  # type: ignore[name-defined]
                vocab = sorted({w for t in self.texts for w in t.lower().split()})
                vectors = []
                for txt in self.texts:
                    counts = Counter(txt.lower().split())
                    vectors.append([counts.get(w, 0) for w in vocab])
                self._matrix = np.array(vectors, dtype=float)
                self._vectorizer = None
                self._nn = None
                if NearestNeighbors is not None:
                    self._nn = NearestNeighbors(metric="cosine")
                    self._nn.fit(self._matrix)
            else:
                self._vectorizer = TfidfVectorizer().fit(self.texts)
                self._matrix = self._vectorizer.transform(self.texts)
                self._nn = NearestNeighbors(metric="cosine")
                self._nn.fit(self._matrix)
        else:
            self._vectorizer = TfidfVectorizer().fit(self.texts)
            self._matrix = self._vectorizer.transform(self.texts)
            if self.use_hnsw:
                self._hnsw = hnswlib.Index(space="cosine", dim=self._matrix.shape[1])
                self._hnsw.init_index(
                    max_elements=len(self.ids), ef_construction=100, M=16
                )
                self._hnsw.add_items(self._matrix.toarray(), list(range(len(self.ids))))
                self._hnsw.set_ef(50)
                self._nn = None
            else:
                self._nn = NearestNeighbors(metric="cosine")
                self._nn.fit(self._matrix)
                self._hnsw = None
            return
        if self.use_hnsw and self._matrix is not None:
            self._hnsw = hnswlib.Index(space="cosine", dim=self._matrix.shape[1])
            self._hnsw.init_index(max_elements=len(self.ids), ef_construction=100, M=16)
            self._hnsw.add_items(
                (
                    self._matrix
                    if isinstance(self._matrix, np.ndarray)
                    else self._matrix.toarray()
                ),
                list(range(len(self.ids))),
            )
            self._hnsw.set_ef(50)
            self._nn = None
        elif self._matrix is not None and self._nn is not None:
            self._nn.fit(self._matrix)
            self._hnsw = None

    def _ensure_index(self) -> None:
        """Build the index if it has not been constructed yet."""
        if self._vectorizer is None or self._matrix is None:
            self.build()
        elif self.use_hnsw and self._hnsw is None:
            self.build()
        elif not self.use_hnsw and self._nn is None:
            self.build()

    def nearest_neighbors(
        self, k: int = 3, return_distances: bool = False
    ) -> Dict[str, List[tuple[str, float]] | List[str]]:
        """Return the ``k`` nearest neighbors for each text ID.

        Parameters
        ----------
        k:
            Number of neighbors to return.
        return_distances:
            If ``True``, return a list of ``(id, similarity)`` tuples for each
            chunk. Similarity is ``1 - cosine_distance``.
        """

        self._ensure_index()
        if not self.texts or self._matrix is None:
            return {}

        if self.use_hnsw and self._hnsw is not None:
            indices, distances = self._hnsw.knn_query(self._matrix.toarray(), k + 1)
            distances = distances.tolist()
            indices = indices.tolist()
        else:
            distances, indices = self._nn.kneighbors(
                self._matrix, n_neighbors=min(k + 1, len(self.ids))
            )
        neighbors: Dict[str, List[tuple[str, float]] | List[str]] = {}
        for idx, (dist_row, neigh_row) in enumerate(zip(distances, indices)):
            items: List[tuple[str, float]] | List[str] = []
            for dist, i in zip(dist_row, neigh_row):
                if i == idx:
                    continue
                if len(items) >= k:
                    break
                if return_distances:
                    items.append((self.ids[i], 1 - float(dist)))
                else:
                    items.append(self.ids[i])
            neighbors[self.ids[idx]] = items
        return neighbors

    def search(self, query: str, k: int = 5) -> List[int]:
        if not self.texts:
            return []
        if self._vectorizer is None:
            self.build()
        query_vec = self._vectorizer.transform([query])
        if self.use_hnsw and self._hnsw is not None:
            indices, _ = self._hnsw.knn_query(query_vec.toarray(), k)
            return indices[0].tolist()
        distances, indices = self._nn.kneighbors(
            query_vec, n_neighbors=min(k, len(self.ids))
        )
        return indices[0].tolist()

    def get_text(self, idx: int) -> str:
        return self.texts[idx]

    def get_id(self, idx: int) -> str:
        return self.ids[idx]

    # ------------------------------------------------------------------
    # New helpers
    # ------------------------------------------------------------------

    def transform(self, texts: List[str]) -> np.ndarray:
        """Return embeddings for ``texts`` using the internal vectorizer."""

        if self._vectorizer is None:
            self.build()
        if self._vectorizer is None:  # no texts indexed
            return np.empty((0, 0))
        return self._vectorizer.transform(texts).toarray()

    def embed(self, text: str) -> np.ndarray:
        """Return the embedding vector for ``text``."""

        mat = self.transform([text])
        return mat[0] if len(mat) else np.zeros(0)
