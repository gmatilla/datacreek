from enum import Enum


class ExportFormat(str, Enum):
    """Supported dataset export formats."""

    JSONL = "jsonl"
    ALPACA = "alpaca"
    FT = "ft"
    CHATML = "chatml"
    DELTA = "delta"
