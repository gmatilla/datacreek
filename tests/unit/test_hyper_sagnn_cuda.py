import importlib

import numpy as np
import pytest

spec = importlib.util.find_spec("torch")


def test_stream_matches_numpy(monkeypatch):
    from datacreek.analysis.hyper_sagnn_cuda import hyper_sagnn_embeddings_stream
    from datacreek.analysis.hypergraph import hyper_sagnn_embeddings

    edges = [[0, 1], [1, 2, 3]]
    feats = np.eye(4)

    cpu = hyper_sagnn_embeddings(edges, feats, embed_dim=2, seed=0)
    gpu = hyper_sagnn_embeddings_stream(
        edges, feats, embed_dim=2, stream_batch=1, device="cpu", seed=0
    )

    assert np.allclose(cpu, gpu, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(spec is None, reason="torch required")
def test_stream_gpu_available(monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    edges = [[0, 1, 2], [2, 3]]
    feats = np.random.rand(4, 3).astype(np.float32)

    res = hyper_sagnn_embeddings_stream(edges, feats, embed_dim=3)
    assert res.shape == (2, 3)
