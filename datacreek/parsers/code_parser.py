from __future__ import annotations

from .base import BaseParser


class CodeParser(BaseParser):
    """Simple parser for code files."""

    def parse(self, file_path: str) -> str:
        """Return the code as text."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
