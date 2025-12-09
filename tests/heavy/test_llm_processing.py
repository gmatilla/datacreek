import sys

import pytest

from datacreek.models.qa import QAPair
from datacreek.utils import llm_processing


@pytest.mark.heavy
def test_parse_qa_pairs_json():
    text = '[{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]'
    pairs = llm_processing.parse_qa_pairs(text)
    assert [p.to_dict() for p in pairs] == [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
    ]


@pytest.mark.heavy
def test_parse_ratings_array():
    text = '[{"question": "Q1", "answer": "A1", "rating": 0.5}]'
    out = llm_processing.parse_ratings(text)
    assert len(out) == 1 and out[0].rating == 0.5


@pytest.mark.heavy
def test_parse_ratings_error():
    with pytest.raises(ValueError):
        llm_processing.parse_ratings("bad")


@pytest.mark.heavy
def test_conversion_helpers():
    pairs = [QAPair("Q", "A")]
    conv = llm_processing.convert_to_conversation_format(pairs, system_prompt="hi")
    assert conv == [
        [
            {"role": "system", "content": "hi"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
    ]
    recs = llm_processing.qa_pairs_to_records(pairs, system_prompt="hi")
    assert recs[0]["conversations"][0]["content"] == "hi" and recs[0]["chunk"] is None


@pytest.mark.heavy
def test_parse_ratings_regex_fallback():
    txt = '{"question": "Q1", "answer": "A1", "rating": 4}}'
    res = llm_processing.parse_ratings(txt)
    assert res and res[0].rating == 4


@pytest.mark.heavy
def test_parse_ratings_json5(monkeypatch):
    class Dummy:
        @staticmethod
        def loads(s):
            return {"question": "Q", "answer": "A", "rating": 6}

    monkeypatch.setitem(sys.modules, "json5", Dummy)
    res = llm_processing.parse_ratings("nonsense")
    assert res[0].rating == 6


@pytest.mark.heavy
def test_parse_ratings_line_by_line():
    items = [QAPair("QQ", "AA")]
    txt = 'something QQ "rating": 5 more text'
    res = llm_processing.parse_ratings(txt, [i.to_dict() for i in items])
    assert res[0].rating == 5


@pytest.mark.heavy
def test_qa_pairs_to_records_modify():
    pairs = [QAPair("Q", "A", chunk="c", source="s")]

    def add_tag(conv, pair):
        conv.append({"role": "tag", "content": pair.question})

    recs = llm_processing.qa_pairs_to_records(pairs, modify=add_tag)
    assert recs[0]["chunk"] == "c" and recs[0]["conversations"][-1]["role"] == "tag"


@pytest.mark.heavy
def test_parse_qa_pairs_regex():
    text = (
        'some "question": "Q1", "answer": "A1" another "question": "Q2", "answer": "A2"'
    )
    pairs = llm_processing.parse_qa_pairs(text)
    assert len(pairs) == 2 and pairs[1].answer == "A2"


@pytest.mark.heavy
def test_parse_qa_pairs_no_match(caplog):
    caplog.set_level("ERROR")
    assert llm_processing.parse_qa_pairs("nothing") == []
    assert any("Failed to parse QA pairs" in r.message for r in caplog.records)


@pytest.mark.heavy
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


@pytest.mark.heavy
def test_parse_ratings_invalid_item_array():
    txt = '[{"question": "Q", "answer": "A"}]'
    assert llm_processing.parse_ratings(txt) == []
