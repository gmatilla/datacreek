import pytest

from datacreek.utils.crypto import (
    decrypt_pii_fields,
    encrypt_pii_fields,
    xor_decrypt,
    xor_encrypt,
)


def test_xor_encrypt_decrypt_roundtrip():
    text = "hello world"
    key = "secret"
    token = xor_encrypt(text, key)
    assert isinstance(token, str)
    assert xor_decrypt(token, key) == text


def test_encrypt_decrypt_pii_fields():
    record = {"name": "Alice", "age": 30, "email": "alice@example.com"}
    key = "k"
    fields = ["name", "email"]
    encrypted = encrypt_pii_fields(record.copy(), key, fields)
    # values should change for encrypted fields but not for others
    assert encrypted["name"] != "Alice"
    assert encrypted["age"] == 30
    assert encrypted["email"] != "alice@example.com"
    # decrypt to recover original
    decrypted = decrypt_pii_fields(encrypted, key, fields)
    assert decrypted == record


def test_encrypt_skip_missing_or_none():
    record = {"id": 1, "name": None}
    key = "secret"
    fields = ["name", "missing"]
    encrypted = encrypt_pii_fields(record.copy(), key, fields)
    # unchanged due to None or missing
    assert encrypted["name"] is None
    assert "missing" not in encrypted
