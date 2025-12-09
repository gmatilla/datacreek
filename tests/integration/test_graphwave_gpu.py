import importlib

import networkx as nx
import numpy as np
import pytest

spec = importlib.util.find_spec("cupy")


@pytest.mark.skipif(spec is None, reason="cupy required")
def test_chebyshev_heat_kernel_gpu_matches_cpu():

    from datacreek.analysis.fractal import chebyshev_heat_kernel
    from datacreek.analysis.graphwave_cuda import chebyshev_heat_kernel_gpu

    g = nx.path_graph(4)
    L = nx.normalized_laplacian_matrix(g).tocsr()
    h_cpu = chebyshev_heat_kernel(L, 0.5, m=3)
    h_gpu = chebyshev_heat_kernel_gpu(L, 0.5, m=3)
    assert np.allclose(h_cpu, h_gpu, atol=1e-4)  # noqa: S101


@pytest.mark.skipif(spec is None, reason="cupy required")
def test_chebyshev_heat_kernel_gpu_batch():

    from datacreek.analysis.fractal import chebyshev_heat_kernel
    from datacreek.analysis.graphwave_cuda import (  # noqa: E501
        chebyshev_heat_kernel_gpu_batch,
    )

    g = nx.path_graph(4)
    L = nx.normalized_laplacian_matrix(g).tocsr()
    cpu = [chebyshev_heat_kernel(L, t, m=3) for t in (0.25, 0.5)]
    gpu = chebyshev_heat_kernel_gpu_batch(L, [0.25, 0.5], m=3)
    for c, gk in zip(cpu, gpu):
        assert np.allclose(c, gk, atol=1e-4)  # noqa: S101


@pytest.mark.skipif(spec is None, reason="cupy required")
def test_chebyshev_heat_kernel_gpu_stream():
    """Streaming kernel should match full computation."""
    from datacreek.analysis.graphwave_cuda import (
        chebyshev_heat_kernel_gpu,
        chebyshev_heat_kernel_gpu_stream,
    )

    g = nx.path_graph(6)
    L = nx.normalized_laplacian_matrix(g).tocsr()
    ref = chebyshev_heat_kernel_gpu(L, 0.5, m=5)
    streaming = chebyshev_heat_kernel_gpu_stream(L, 0.5, order=5, block=2)
    assert np.allclose(ref, streaming, atol=1e-4)  # noqa: S101


@pytest.mark.skipif(spec is None, reason="cupy required")
def test_graphwave_runner_gpu(monkeypatch):
    from datacreek.core.knowledge_graph import KnowledgeGraph
    from datacreek.core.runners import GraphWaveRunner

    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")
    monkeypatch.setattr(kg, "graphwave_entropy", lambda: 0.0)
    monkeypatch.setattr(
        kg,
        "compute_graphwave_embeddings",
        lambda *a, **k: None,
    )
    runner = GraphWaveRunner(kg)
    runner.run(scales=[0.5], gpu=True)


@pytest.mark.gpu
@pytest.mark.skipif(spec is None, reason="cupy required")
def test_graphwave_embedding_gpu_precision():
    """GPU embeddings should match CPU implementation within 1e-5."""
    from datacreek.analysis.fractal import graphwave_embedding_chebyshev
    from datacreek.analysis.graphwave_cuda import graphwave_embedding_gpu

    g = nx.path_graph(1000)
    scales = [0.3]

    cpu = graphwave_embedding_chebyshev(g, scales, num_points=3, order=5)
    gpu = graphwave_embedding_gpu(g, scales, num_points=3, order=5)

    arr_cpu = np.stack([cpu[n] for n in sorted(g.nodes())])
    arr_gpu = np.stack([gpu[n] for n in sorted(g.nodes())])

    rel_err = np.linalg.norm(arr_gpu - arr_cpu) / np.linalg.norm(arr_cpu)
    assert rel_err < 1e-5  # noqa: S101


def test_stream_memory_estimation():
    from datacreek.analysis.graphwave_cuda import estimate_stream_memory

    mem = estimate_stream_memory(10_000_000, 64)
    assert mem < 5 * 1024**3  # noqa: S101


def test_stream_block_selection():
    from datacreek.analysis.graphwave_cuda import (
        choose_stream_block,
        estimate_stream_memory,
    )

    b = choose_stream_block(10_000_000, limit_gb=5)
    assert estimate_stream_memory(10_000_000, b) <= 5 * 1024**3  # noqa: S101


def test_stream_block_formula():
    from datacreek.analysis.graphwave_cuda import choose_stream_block

    assert choose_stream_block(10, order=64, limit_gb=10) == 32  # noqa: S101


@pytest.mark.skipif(spec is None, reason="cupy required")
def test_stream_performance_within_baseline():
    """Streaming implementation should be within 10% of full version."""
    import time

    from datacreek.analysis.graphwave_cuda import (
        chebyshev_heat_kernel_gpu,
        chebyshev_heat_kernel_gpu_stream,
    )

    g = nx.path_graph(200)
    L = nx.normalized_laplacian_matrix(g).tocsr()
    start = time.perf_counter()
    full = chebyshev_heat_kernel_gpu(L, 0.5, m=7)
    t_full = time.perf_counter() - start

    start = time.perf_counter()
    streamed = chebyshev_heat_kernel_gpu_stream(L, 0.5, order=7, block=4)
    t_stream = time.perf_counter() - start

    assert np.allclose(full, streamed, atol=1e-4)  # noqa: S101
    assert t_stream <= t_full * 1.1  # noqa: S101
