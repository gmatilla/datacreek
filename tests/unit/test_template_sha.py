"""Tests for the template SHA256 validation hook."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.validate_template_sha import validate_paths


def _write_template(tmp_path: Path, body: str, digest: str | None = None) -> Path:
    """Utility to create a temporary template with an optional digest."""
    if digest is None:
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    tmpl = tmp_path / "tmp.jinja"
    tmpl.write_text(f"# sha256: {digest}\n{body}")
    return tmpl


def test_validate_template_passes(tmp_path: Path) -> None:
    tmpl = _write_template(tmp_path, "Hello")
    # Should not raise an exception for valid template
    validate_paths([tmpl])


def test_validate_template_fails_on_mismatch(tmp_path: Path) -> None:
    tmpl = _write_template(tmp_path, "Hello", digest="0" * 64)
    with pytest.raises(SystemExit):
        validate_paths([tmpl])
