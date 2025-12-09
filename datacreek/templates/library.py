from __future__ import annotations

"""Prompt template library with validation utilities."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import jsonschema

# Template definitions live under ``specs/`` to avoid gitignore "data/" rules
TEMPLATE_DIR = Path(__file__).resolve().parent / "specs"


@dataclass
class PromptTemplate:
    """Metadata describing a prompt/response format."""

    name: str
    schema: dict
    max_length: int
    regex: str

    def validate(self, output: str) -> bool:
        """Return ``True`` if ``output`` respects length and regex constraints."""
        if len(output) > self.max_length:
            return False
        if self.regex and not re.fullmatch(self.regex, output.strip(), re.DOTALL):
            return False
        try:
            jsonschema.validate(json.loads(output), self.schema)
        except Exception:
            return False
        return True


def load_templates() -> Dict[str, PromptTemplate]:
    """Load templates from the ``data`` directory."""
    templates: Dict[str, PromptTemplate] = {}
    if not TEMPLATE_DIR.exists():
        return templates
    for path in TEMPLATE_DIR.glob("*.json"):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        tmpl = PromptTemplate(
            name=path.stem,
            schema=data.get("schema", {}),
            max_length=int(data.get("max_length", 4096)),
            regex=data.get("regex", r".*"),
        )
        templates[tmpl.name] = tmpl
    return templates


TEMPLATES = load_templates()


def get_template(name: str) -> PromptTemplate:
    """Return template by ``name``."""
    if name not in TEMPLATES:
        raise KeyError(f"Unknown template: {name}")
    return TEMPLATES[name]


def validate_output(template_name: str, text: str) -> bool:
    """Validate ``text`` using template constraints."""
    tmpl = get_template(template_name)
    return tmpl.validate(text)
