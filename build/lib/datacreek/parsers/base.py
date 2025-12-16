class BaseParser:
    """Abstract parser interface."""

    def parse(self, file_path: str) -> str:
        """Return raw text from the given resource."""
        raise NotImplementedError

    def save(self, content: str, output_path: str) -> None:  # pragma: no cover - legacy
        """Deprecated: parsers no longer write files to disk."""
        raise RuntimeError("save() is deprecated; parsers operate in memory only")
