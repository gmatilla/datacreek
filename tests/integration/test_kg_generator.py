import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.generators.kg_generator import KGGenerator


class DummyClient:
    def __init__(self):
        self.calls = []
        self.config = {
            "prompts": {"kg_question": "{facts}", "kg_answer": "{question} {facts}"},
            "generation": {"temperature": 0.1, "max_tokens": 50},
        }

    def chat_completion(self, messages, *, temperature=None, max_tokens=None):
        self.calls.append(messages[0]["content"])
        if len(self.calls) % 2 == 1:
            return "What?"
        return "Because"


def test_kg_generator_basic():
    kg = KnowledgeGraph()
    fid = kg.add_fact("Earth", "is", "round")
    gen = KGGenerator(DummyClient())
    result = gen.process_graph(kg, num_pairs=1)
    assert result == {
        "qa_pairs": [{"question": "What?", "answer": "Because", "facts": [fid]}]
    }
    assert len(gen.client.calls) == 2


def test_kg_generator_multi_answer():
    kg = KnowledgeGraph()
    fid = kg.add_fact("Sky", "is", "blue")
    gen = KGGenerator(DummyClient())
    res = gen.process_graph(kg, num_pairs=2, multi_answer=True)
    assert len(res["qa_pairs"]) == 2
    assert res["qa_pairs"][0]["question"] == "What?"
    assert len(gen.client.calls) == 3


def test_kg_generator_select_limit():
    kg = KnowledgeGraph()
    for i in range(5):
        kg.add_fact(f"S{i}", "is", f"O{i}")

    gen = KGGenerator(DummyClient())
    res = gen.process_graph(kg, num_pairs=2)
    assert len(res["qa_pairs"]) == 2


def test_kg_generator_confidence():
    kg = KnowledgeGraph()
    fid = kg.add_fact("A", "related", "B")
    gen = KGGenerator(DummyClient())
    res = gen.process_graph(kg, num_pairs=1)
    pair = res["qa_pairs"][0]
    assert pair["facts"] == [fid]
    assert "confidence" in pair
