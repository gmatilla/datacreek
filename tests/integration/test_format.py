import fakeredis
import pytest

from datacreek.core.save_as import convert_format


def test_convert_format_invalid():
    with pytest.raises(ValueError):
        convert_format({"qa_pairs": []}, "bad")


def test_convert_format_jsonl_roundtrip():
    qa = {"qa_pairs": [{"question": "q", "answer": "a"}]}
    client = fakeredis.FakeStrictRedis()
    key = "qa:test"
    res = convert_format(qa, "jsonl", redis_client=client, redis_key=key)
    assert res == key
    assert client.get(key).decode().strip() == '{"question": "q", "answer": "a"}'


def test_convert_format_in_memory():
    qa = {"qa_pairs": [{"question": "q", "answer": "a"}]}
    res = convert_format(qa, "alpaca")
    assert "instruction" in res
