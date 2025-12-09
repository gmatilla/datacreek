import sys

import networkx as nx
import numpy as np
import pytest

from datacreek.generators import kg_generator, vqa_generator


class DummyClient:
    def __init__(self):
        self.config = {}

    def chat_completion(self, *args, **kwargs):
        return "reply"


class DummyIndex:
    def __init__(self, result):
        self._result = result

    def transform(self, texts):
        return self._result


class FakeKG:
    def __init__(self, num_facts, embeddings):
        self.graph = nx.DiGraph()
        for i in range(num_facts):
            self.graph.add_node(
                f"f{i}",
                type="fact",
                subject="s",
                predicate="p",
                object=str(i),
            )
        self.index = DummyIndex(embeddings)


def test_select_facts_small_graph():
    kg = FakeKG(2, np.zeros((0, 0)))
    gen = kg_generator.KGGenerator(DummyClient())
    selected = gen._select_facts(kg, 5)
    assert set(selected) == {"f0", "f1"}


def test_select_facts_no_embeddings_sort_by_degree():
    kg = FakeKG(3, np.empty((0, 0)))
    kg.graph.add_edge("f0", "f1")
    kg.graph.add_edge("f0", "f2")
    gen = kg_generator.KGGenerator(DummyClient())
    result = gen._select_facts(kg, 2)
    assert result == ["f0", "f1"]


def test_select_facts_requires_kmeans(monkeypatch):
    kg = FakeKG(4, np.ones((4, 2)))
    monkeypatch.setattr(kg_generator, "KMeans", None)
    gen = kg_generator.KGGenerator(DummyClient())
    with pytest.raises(ImportError):
        gen._select_facts(kg, 2)


def test_check_optional_deps_missing(monkeypatch):
    """_check_optional_deps should raise when libraries are absent."""
    monkeypatch.setitem(sys.modules, "datasets", None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError):
        vqa_generator._check_optional_deps()


def test_encode_image_base64(monkeypatch):
    from PIL import Image

    monkeypatch.setattr(vqa_generator, "_check_optional_deps", lambda: None)
    gen = vqa_generator.VQAGenerator(DummyClient())
    img = Image.new("RGB", (1, 1), color="red")
    data = gen.encode_image_base64(img)
    assert isinstance(data, str) and len(data) > 0


def test_process_graph_basic(monkeypatch):
    kg = FakeKG(1, np.ones((1, 2)))

    def get_chunks(fid):
        return []

    kg.get_chunks_for_fact = lambda fid: get_chunks(fid)
    kg.fact_confidence = lambda s, p, o: 0.5
    monkeypatch.setattr(
        kg_generator,
        "get_prompt",
        lambda cfg, name: "{facts}" if name == "kg_question" else "answer",
    )
    gen = kg_generator.KGGenerator(DummyClient())
    result = gen.process_graph(kg, num_pairs=1)
    assert result["qa_pairs"][0]["facts"] == ["f0"]


def test_process_graph_multi_answer(monkeypatch):
    kg = FakeKG(2, np.ones((2, 2)))
    for fid in ["f0", "f1"]:
        kg.graph.add_node(f"c{fid}", text=f"chunk {fid}")
    kg.get_chunks_for_fact = lambda fid: [f"c{fid}"]
    kg.fact_confidence = lambda s, p, o: 0.2
    monkeypatch.setattr(
        kg_generator,
        "get_prompt",
        lambda cfg, name: "{facts}" if name == "kg_question" else "A {question}",
    )
    gen = kg_generator.KGGenerator(DummyClient())
    out = gen.process_graph(kg, num_pairs=3, multi_answer=True)
    assert len(out["qa_pairs"]) >= 2
