from datacreek.core.knowledge_graph import KnowledgeGraph


def build_graph():
    kg = KnowledgeGraph()
    kg.add_document("d", "src")
    kg.add_chunk("d", "c", "text")
    kg.add_entity("e1", "Alice")
    kg.add_entity("e2", "Bob")
    kg.add_fact("e1", "knows", "e2", fact_id="f1")
    kg.link_entity("c", "e1")
    return kg


def test_fact_confidence_and_verify():
    kg = build_graph()
    assert kg.fact_confidence("e1", "knows", "e2") == 1.0
    assert kg.fact_confidence("e1", "likes", "e2") == 0.4
    score = kg.verify_statements([("e1", "knows", "e2"), ("e1", "likes", "e2")])
    assert round(score, 2) == 0.70
