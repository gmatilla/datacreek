"""Hybrid ANN retrieval combining HNSW and IVFPQ."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from datacreek.backend import get_xp

xp = get_xp()

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - faiss optional
    faiss = None  # type: ignore

__all__ = ["rerank_pq", "search_hnsw_pq"]


def expected_recall(nprobe: int, n_cells: int, tables: int) -> float:
    r"""Return approximate recall with multi-probing.

    Parameters
    ----------
    nprobe:
        Number of coarse centroids probed per table.
    n_cells:
        Total number of centroids in the IVF index.
    tables:
        Number of product quantisation tables.

    Notes
    -----
    The estimate follows

    .. math::

       1 - \left(1 - \frac{n_{\text{probe}}}{N_{\text{cells}}}\right)^{L}

    where ``L`` is ``tables``.
    """

    if n_cells == 0:
        return 0.0
    return 1.0 - (1.0 - nprobe / n_cells) ** tables


def choose_nprobe_multi(n_cells: int, tables: int) -> int:
    """Return a heuristic ``nprobe_multi`` based on index size.

    ``nprobe_multi`` is chosen as the square root of ``n_cells`` per
    subquantizer so that ``nprobe_multi=32`` when ``n_cells=2**14`` and
    ``tables=16`` (as in an index with 16 sub-quantizers).
    """

    if n_cells <= 0 or tables <= 0:
        return 1
    return max(1, int(round((n_cells / tables) ** 0.5)))


def load_ivfpq_cpu(
    path: str, base_nprobe: int
) -> object:  # pragma: no cover - requires faiss
    """Load an IVFPQ index on CPU and set ``nprobe_multi``.

    Parameters
    ----------
    path:
        Path to the index file created by FAISS.
    base_nprobe:
        Number of probes per subquantizer.

    Returns
    -------
    object
        Loaded FAISS index with ``nprobe_multi`` configured.
    """

    if faiss is None:
        raise RuntimeError("faiss not installed")
    index = faiss.read_index(path)
    pq = getattr(index, "pq", None)
    n_subquantizers = getattr(index, "nsq", pq.M if pq is not None else 1)
    index.nprobe_multi = [base_nprobe] * int(n_subquantizers)
    return index


def rerank_pq(  # pragma: no cover - requires faiss
    xb: np.ndarray,
    xq: np.ndarray,
    *,
    k: int = 10,
    gpu: bool = True,
    n_subprobe: int = 1,
) -> np.ndarray:
    """Return ``k`` nearest indices using IVFPQ on CPU or GPU.

    Parameters
    ----------
    xb:
        Database vectors of shape ``(n, d)`` used for training and search.
    xq:
        Query vectors of shape ``(m, d)``.
    k:
        Number of neighbours to return.
    gpu:
        Whether to run the search on GPU when available.
    n_subprobe:
        Number of additional FAISS multi-probes. ``nprobe`` becomes
        ``base * n_subprobe`` where ``base`` is ``max(1, nlist // 4)``.

    Returns
    -------
    ``np.ndarray``
        Indices of size ``(m, k)`` referring to ``xb``.
    """
    if faiss is None:
        raise RuntimeError("faiss not installed")

    d = xb.shape[1]
    quantizer = faiss.IndexFlatIP(d)

    if xb.shape[0] < 256:
        index = faiss.IndexFlatIP(d)
        index.add(xb)
    else:
        nlist = max(4, int(xp.sqrt(xb.shape[0])))
        nbits = 8 if xb.shape[0] >= 1000 else 6
        index = faiss.IndexIVFPQ(quantizer, d, nlist, 8, int(nbits))

        if (
            gpu
            and getattr(faiss, "StandardGpuResources", None)
            and faiss.get_num_gpus() > 0
        ):
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.train(xb)
        index.add(xb)
        base = max(1, nlist // 4)
        index.nprobe = base * max(1, int(n_subprobe))
        if not gpu and hasattr(index, "nprobe_multi"):
            n_subquantizers = getattr(index, "nsq", index.pq.M)
            base_multi = choose_nprobe_multi(nlist, n_subquantizers)
            index.nprobe_multi = [base_multi] * n_subquantizers

    _, rerank = index.search(xq, k)
    return rerank


def search_hnsw_pq(  # pragma: no cover - requires faiss
    xb: np.ndarray,
    xq: np.ndarray,
    *,
    k: int = 10,
    prefetch: int = 50,
    n_subprobe: int = 1,
) -> Sequence[int]:
    """Return ``k`` neighbours using an HNSW stage then PQ reranking.

    Parameters
    ----------
    xb:
        Database vectors of shape ``(n, d)``.
    xq:
        Query vectors of shape ``(1, d)``.
    k:
        Number of neighbours to return.
    prefetch:
        Size of candidate set retrieved with HNSW before reranking.
    n_subprobe:
        Number of sub-probes used during PQ reranking.

    Returns
    -------
    Sequence[int]
        Indices of the nearest neighbours.
    """
    if faiss is None:
        raise RuntimeError("faiss not installed")

    d = xb.shape[1]
    hnsw = faiss.IndexHNSWFlat(d, 32)
    hnsw.hnsw.efSearch = max(prefetch * 2, 64)
    hnsw.add(xb)
    _, idx = hnsw.search(xq, prefetch)
    candidates = idx[0]

    rerank = rerank_pq(
        xb[candidates],
        xq,
        k=k,
        gpu=True,
        n_subprobe=n_subprobe,
    )
    return [int(candidates[i]) for i in rerank[0]]
