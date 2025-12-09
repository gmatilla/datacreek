import pytest
from pydantic import ValidationError

from datacreek.utils.config import load_config


def test_schema_defaults():
    cfg = load_config("configs/default.yaml")
    assert cfg["pid"]["Kp"] == 0.4
    assert cfg["pid"]["Ki"] == 0.05
    assert cfg["gpu"]["enabled"] is False


def test_schema_validation_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("pid:\n  Kp: 1.5\n  Ki: 0.05\n")
    with pytest.raises(ValidationError):
        load_config(str(bad))
