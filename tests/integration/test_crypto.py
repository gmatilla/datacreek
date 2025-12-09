import pytest

from datacreek.utils.crypto import (
    decrypt_pii_fields,
    encrypt_pii_fields,
    xor_decrypt,
    xor_encrypt,
)


def test_xor_roundtrip():
    text = "secret"
    key = "k"
    enc = xor_encrypt(text, key)
    assert enc != text
    dec = xor_decrypt(enc, key)
    assert dec == text


def test_encrypt_pii_fields():
    record = {"author": "Alice", "organization": "Org"}
    key = "k"
    encrypt_pii_fields(record, key, ["author", "organization"])
    assert record["author"] != "Alice"
    decrypt_pii_fields(record, key, ["author", "organization"])
    assert record["author"] == "Alice"
    assert record["organization"] == "Org"
