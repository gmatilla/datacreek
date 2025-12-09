import datacreek.utils.redis_helpers as rh


def test_decode_hash_basic():
    data = {b"key1": b"42", b"key2": "value"}
    assert rh.decode_hash(data) == {"key1": 42, "key2": "value"}


def test_decode_hash_invalid_json():
    data = {"k": b"invalid{"}
    assert rh.decode_hash(data) == {"k": "invalid{"}
