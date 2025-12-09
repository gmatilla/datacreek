# pragma: no cover - requires GPU libraries
"""GPU-accelerated GraphWave utilities."""

from __future__ import annotations

import math
from typing import Iterable

import networkx as nx
import numpy as np

from datacreek.backend import get_xp

xp = get_xp()


def estimate_stream_memory(n: int, block: int, *, dtype_size: int = 4) -> int:
    """Return estimated bytes for streaming kernel.

    Parameters
    ----------
    n:
        Number of nodes in the graph.
    block:
        Active Chebyshev blocks stored on the device.
    dtype_size:
        Size in bytes of the floating point representation. ``4`` for float32.

    Notes
    -----
    The memory footprint is approximately ``2 * n * dtype_size * block`` bytes
    as only two ``(n, block)`` matrices are allocated at any time.
    """

    return 2 * n * dtype_size * block


def choose_stream_block(
    n: int,
    *,
    limit_gb: float = 8.0,
    order: int | None = None,
    dtype_size: int = 4,
) -> int:
    r"""Return block size so usage fits within ``limit_gb`` VRAM.

    Parameters
    ----------
    n:
        Number of nodes in the graph.
    limit_gb:
        Available memory budget in gigabytes.
    order:
        Total number of Chebyshev coefficients to compute. When provided,
        ``block`` is chosen according to

        .. math::

           b = \left\lceil \frac{m}{\lceil V / 5\,\text{GB}\rceil} \right\rceil

        where :math:`m` is ``order`` and :math:`V` is ``limit_gb``.
    dtype_size:
        Size of a single coefficient in bytes.
    """

    if order is not None:
        denom = max(1, int(math.ceil(limit_gb / 5.0)))
        return int(max(1, math.ceil(order / denom)))

    bytes_limit = int(limit_gb * (1024**3))
    block = max(1, bytes_limit // (2 * n * dtype_size))
    return int(block)


try:  # pragma: no cover - optional dependency
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse import csr_matrix as csr_gpu
except Exception:  # pragma: no cover - cupy not installed
    cp = None  # type: ignore
    csp = None  # type: ignore
    csr_gpu = None  # type: ignore

__all__ = [
    "chebyshev_heat_kernel_gpu",
    "chebyshev_heat_kernel_gpu_batch",
    "chebyshev_heat_kernel_gpu_stream",
    "graphwave_embedding_gpu",
]


def chebyshev_heat_kernel_gpu_batch(  # pragma: no cover - requires GPU
    L: np.ndarray | "csp.spmatrix", ts: Iterable[float], m: int = 7
) -> list[np.ndarray]:
    """Return GPU approximation of ``exp(-t L)`` for multiple scales.

    The Laplacian is normalised once, then the Chebyshev polynomials are
    reused for every ``t`` which limits the number of sparse matrix-vector
    multiplications. This is more efficient when many diffusion scales are
    evaluated.

    Parameters
    ----------
    L:
        Laplacian matrix (CSR) on CPU or GPU.
    ts:
        Iterable of diffusion scales.
    m:
        Order of the approximation.

    Returns
    -------
    list[np.ndarray]
        Heat kernel matrices on the host in the same order as ``ts``.
    """
    if cp is None:
        raise RuntimeError("cupy is required for GPU GraphWave")

    arr = L
    if not isinstance(arr, csr_gpu):
        arr = csp.csr_matrix(L)

    n = arr.shape[0]
    # Estimate lmax with power iteration on GPU
    x = xp.random.rand(n, dtype=xp.float32)
    x /= xp.linalg.norm(x)
    for _ in range(5):
        x = arr @ x
        norm = xp.linalg.norm(x)
        if norm == 0:
            break
        x /= norm
    lmax = xp.float32(x.T @ (arr @ x))
    if float(lmax) == 0.0:
        lmax = xp.float32(1.0)

    L_norm = (2.0 / lmax) * arr - csp.identity(
        n,
        format="csr",
        dtype=xp.float32,
    )

    # Pre-compute Chebyshev polynomials
    Tk_minus = csp.identity(n, format="csr", dtype=xp.float32)
    Tk = L_norm
    polys = [Tk_minus, Tk]
    for _ in range(2, m + 1):
        Tk_plus = 2 * L_norm.dot(Tk) - Tk_minus
        polys.append(Tk_plus)
        Tk_minus, Tk = Tk, Tk_plus

    from cupyx.scipy.special import iv

    results = []
    for t in ts:
        coeffs = [2 * iv(k, t * lmax / 2.0) for k in range(m + 1)]
        psi = csp.csr_matrix((n, n), dtype=xp.float32)
        for c, T in zip(coeffs, polys):
            if float(c) != 0.0:
                psi = psi + c * T
        psi = xp.exp(-t * lmax / 2.0, dtype=xp.float32) * psi
        results.append(
            cp.asnumpy(psi.toarray())  # noqa: E501
            if cp is not None
            else xp.asarray(psi.toarray())
        )

    return results


def chebyshev_heat_kernel_gpu(  # pragma: no cover - requires GPU
    L: np.ndarray | "csp.spmatrix", t: float, m: int = 7
) -> np.ndarray:
    """Return GPU approximation of ``exp(-t L)``.

    Uses Chebyshev polynomials for efficiency.
    """

    return chebyshev_heat_kernel_gpu_batch(L, [t], m=m)[0]


def chebyshev_heat_kernel_gpu_stream(  # pragma: no cover - requires GPU
    L: np.ndarray | "csp.spmatrix",
    t: float,
    *,
    order: int = 7,
    block: int | None = None,
    limit_gb: float = 8.0,
) -> np.ndarray:
    """Return GPU approximation of ``exp(-t L)`` streaming Chebyshev blocks.

    The polynomial order ``order`` is processed in chunks of size ``block`` so
    that at most two ``(n, block)`` matrices are stored on the device.  The
    memory usage follows ``M = 2 * n * 4 * block`` bytes where ``n`` is the
    number of nodes and ``4`` the size of ``float32``.

    If ``block`` is ``None``, it is computed via :func:`choose_stream_block` to
    honour ``limit_gb`` of available device memory (defaults to ``8`` GB).
    """

    if cp is None:
        raise RuntimeError("cupy is required for GPU GraphWave")

    arr = L if isinstance(L, csr_gpu) else csp.csr_matrix(L)
    n = arr.shape[0]
    if block is None:
        block = choose_stream_block(n, limit_gb=limit_gb)

    # Estimate lmax via power iteration
    x = xp.random.rand(n, dtype=xp.float32)
    x /= xp.linalg.norm(x)
    for _ in range(5):
        x = arr @ x
        norm = xp.linalg.norm(x)
        if norm == 0:
            break
        x /= norm
    lmax = xp.float32(x.T @ (arr @ x))
    if float(lmax) == 0.0:
        lmax = xp.float32(1.0)

    L_norm = (2.0 / lmax) * arr - csp.identity(
        n,
        format="csr",
        dtype=xp.float32,
    )

    from cupyx.scipy.special import iv

    coeffs = [2 * iv(k, t * lmax / 2.0) for k in range(order + 1)]
    coeffs[0] /= 2.0

    result = csp.csr_matrix((n, n), dtype=xp.float32)
    Tk_minus = csp.identity(n, format="csr", dtype=xp.float32)
    Tk = L_norm
    result = result + coeffs[0] * Tk_minus + coeffs[1] * Tk

    k = 2
    while k <= order:
        for _ in range(block):
            if k > order:
                break
            Tk_plus = 2.0 * L_norm.dot(Tk) - Tk_minus
            result = result + coeffs[k] * Tk_plus
            Tk_minus, Tk = Tk, Tk_plus
            k += 1

    result = xp.exp(-t * lmax / 2.0, dtype=xp.float32) * result
    return (
        cp.asnumpy(result.toarray())
        if cp is not None
        else xp.asarray(result.toarray())  # noqa: E501
    )


def graphwave_embedding_gpu(  # pragma: no cover - requires GPU
    graph: nx.Graph,
    scales: Iterable[float],
    *,
    num_points: int = 10,
    order: int = 7,
) -> dict[int, np.ndarray]:
    """Return GraphWave embeddings using :mod:`cupy` for acceleration."""
    if cp is None:
        raise RuntimeError("cupy is required for GPU GraphWave")

    nodes = list(graph.nodes())
    lap = nx.to_scipy_sparse_array(graph, format="csr")
    ts = np.linspace(0, 2 * np.pi, num_points)
    emb: dict[int, list[float]] = {n: [] for n in nodes}

    heats = chebyshev_heat_kernel_gpu_batch(lap, list(scales), m=order)
    for heat in heats:
        for idx, node in enumerate(nodes):
            coeffs = np.exp(1j * np.outer(ts, heat[idx, :])).sum(axis=1)
            emb[node].extend(coeffs.real)
            emb[node].extend(coeffs.imag)

    return {n: np.asarray(v, dtype=float) for n, v in emb.items()}
