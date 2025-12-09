from datacreek.models.qa import QAPair


def test_to_dict_full():
    pair = QAPair(
        question="q",
        answer="a",
        rating=0.5,
        confidence=0.9,
        chunk="c",
        source="s",
        facts=["f1", "f2"],
    )
    expected = {
        "question": "q",
        "answer": "a",
        "rating": 0.5,
        "confidence": 0.9,
        "chunk": "c",
        "source": "s",
        "facts": ["f1", "f2"],
    }
    assert pair.to_dict() == expected


def test_to_dict_partial():
    pair = QAPair(question="q", answer="a")
    assert pair.to_dict() == {"question": "q", "answer": "a"}
