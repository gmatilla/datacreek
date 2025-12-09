import pytest

faiss = pytest.importorskip("faiss")


from datacreek.core.knowledge_graph import KnowledgeGraph


@pytest.mark.faiss_gpu
def test_faiss_ivfpq_backend_metric(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_node("a", embedding=[1.0, 0.0])
    kg.graph.add_node("b", embedding=[0.0, 1.0])

    metrics = {"value": 0}

    class DummyGauge:
        def set(self, value):
            metrics["value"] = value

    monkeypatch.setattr(
        "datacreek.analysis.monitoring.ann_backend", DummyGauge(), raising=False
    )
    try:
        kg.build_faiss_index(method="faiss_gpu_ivfpq")
    except RuntimeError:
        pytest.skip("faiss not installed")
    assert metrics["value"] == 3
    assert kg.faiss_index_type == "faiss_gpu_ivfpq"


@pytest.mark.faiss_gpu
def test_faiss_ivfpq_cpu_fallback(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_node("a", embedding=[1.0, 0.0])
    kg.graph.add_node("b", embedding=[0.0, 1.0])

    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 0)
    kg.build_faiss_index(method="faiss_gpu_ivfpq")
    assert isinstance(kg.faiss_index, faiss.IndexIVFPQ)
