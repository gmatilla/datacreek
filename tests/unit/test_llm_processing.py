import sys

import pytest

from datacreek.models.qa import QAPair
from datacreek.utils import llm_processing


def test_parse_qa_pairs_json():
    text = '[{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]'
    pairs = llm_processing.parse_qa_pairs(text)
    assert [p.to_dict() for p in pairs] == [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
    ]


def test_parse_qa_pairs_regex():
    text = 'some text "question": "Q1", "answer": "A1" and again "question": "Q2", "answer": "A2"'
    pairs = llm_processing.parse_qa_pairs(text)
    assert len(pairs) == 2
    assert pairs[0].question == "Q1" and pairs[0].answer == "A1"
    assert pairs[1].question == "Q2" and pairs[1].answer == "A2"


def test_parse_ratings_json_array():
    text = '[{"question": "Q1", "answer": "A1", "rating": 0.5}]'
    result = llm_processing.parse_ratings(text)
    assert len(result) == 1
    r = result[0]
    assert r.question == "Q1" and r.answer == "A1" and r.rating == 0.5


def test_parse_ratings_json_object():
    text = '{"question": "Q1", "answer": "A1", "rating": 1}'
    res = llm_processing.parse_ratings(text)
    assert len(res) == 1 and res[0].rating == 1


def test_parse_ratings_code_block():
    block = """```json\n[{\"question\": \"Q\", \"answer\": \"A\", \"rating\": 2}]```"""
    res = llm_processing.parse_ratings(block)
    assert res[0].rating == 2


def test_parse_ratings_pattern_match():
    items = [QAPair("QX", "AX")]
    text = 'QX said "rating": 3 in the text'
    res = llm_processing.parse_ratings(text, [i.to_dict() for i in items])
    assert res and res[0].rating == 3


def test_parse_ratings_regex_fallback():
    # extra brace invalidates JSON so regex path is used
    txt = '{"question": "Q1", "answer": "A1", "rating": 4}}'
    res = llm_processing.parse_ratings(txt)
    assert res and res[0].rating == 4


def test_parse_ratings_json5(monkeypatch):
    class Dummy:
        @staticmethod
        def loads(s):
            return {"question": "Q", "answer": "A", "rating": 6}

    monkeypatch.setitem(sys.modules, "json5", Dummy)
    txt = "nonsense"
    res = llm_processing.parse_ratings(txt)
    assert res[0].rating == 6


def test_parse_ratings_multiple_items_with_meta():
    txt = (
        '[{"question": "Q1", "answer": "A1", "rating": 1},'
        ' {"question": "Q2", "answer": "A2", "rating": 2}]'
    )
    original = [
        QAPair("Q1", "A1", chunk="c1", source="s1"),
        {"question": "Q2", "answer": "A2", "chunk": "c2", "source": "s2"},
    ]
    res = llm_processing.parse_ratings(txt, original)
    assert [r.rating for r in res] == [1, 2]
    assert res[0].chunk == "c1" and res[1].source == "s2"


def test_parse_ratings_invalid_item_array():
    txt = '[{"question": "Q", "answer": "A"}]'
    assert llm_processing.parse_ratings(txt) == []


def test_parse_ratings_line_by_line():
    items = [QAPair("QQ", "AA")]
    txt = 'something QQ "rating": 5 more text'
    res = llm_processing.parse_ratings(txt, [i.to_dict() for i in items])
    assert res[0].rating == 5


def test_parse_ratings_error():
    with pytest.raises(ValueError):
        llm_processing.parse_ratings("invalid")


def test_parse_qa_pairs_no_match(caplog):
    caplog.set_level("ERROR")
    assert llm_processing.parse_qa_pairs("nothing here") == []
    assert any("Failed to parse QA pairs" in r.message for r in caplog.records)


def test_parse_ratings_missing_meta():
    txt = '[{"question": "Q1", "answer": "A1", "rating": 1}, {"question": "Q2", "answer": "A2", "rating": 2}]'
    # only one original item so second should have None metadata
    orig = [QAPair("Q1", "A1", chunk="c1", source="s1")]
    res = llm_processing.parse_ratings(txt, orig)
    assert res[0].chunk == "c1" and res[1].chunk is None and res[1].source is None


def test_convert_to_conversation_format():
    pairs = [QAPair("Q1", "A1"), {"question": "Q2", "answer": "A2"}]
    conv = llm_processing.convert_to_conversation_format(pairs, system_prompt="hi")
    assert conv == [
        [
            {"role": "system", "content": "hi"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ],
        [
            {"role": "system", "content": "hi"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ],
    ]


def test_qa_pairs_to_records_modify():
    pairs = [QAPair("Q", "A", chunk="c", source="s")]

    def add_tag(conv, pair):
        conv.append({"role": "tag", "content": pair.question})

    recs = llm_processing.qa_pairs_to_records(pairs, modify=add_tag)
    assert recs[0]["chunk"] == "c" and recs[0]["source"] == "s"
    assert recs[0]["conversations"][-1]["role"] == "tag"
