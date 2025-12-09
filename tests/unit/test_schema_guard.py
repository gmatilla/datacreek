"""Tests for the LakeFS schema compatibility hook."""

from scripts.check_schema_break import is_breaking_change


def test_addition_is_allowed():
    """Adding new fields should not be considered breaking."""
    prev = {"fields": [{"name": "id", "type": "string"}]}
    curr = {"fields": prev["fields"] + [{"name": "extra", "type": "int"}]}
    assert not is_breaking_change(curr, prev)


def test_removal_is_breaking():
    """Removing a field must trigger a breaking change."""
    prev = {
        "fields": [
            {"name": "id", "type": "string"},
            {"name": "value", "type": "int"},
        ]
    }
    curr = {"fields": [{"name": "id", "type": "string"}]}
    assert is_breaking_change(curr, prev)


def test_type_change_is_breaking():
    """Changing the type of a field is breaking."""
    prev = {"fields": [{"name": "id", "type": "string"}]}
    curr = {"fields": [{"name": "id", "type": "int"}]}
    assert is_breaking_change(curr, prev)
