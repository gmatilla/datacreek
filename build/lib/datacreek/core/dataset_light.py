from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from ..pipelines import DatasetType
from ..utils import push_metrics
from .context import AppContext  # minimal dependency

MAX_NAME_LENGTH = 64
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass
class HistoryEvent:
    operation: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    params: Optional[Dict[str, Any]] | None = None


class FakeGraph:
    def __init__(self) -> None:
        self.graph = {}

    def __getattr__(self, name: str):
        def method(*args, **kwargs):
            return None

        return method


@dataclass
class Policy:
    loops: int = 0


@dataclass
class DatasetBuilder:
    dataset_type: Any
    name: Optional[str] = None
    graph: Any = field(default_factory=FakeGraph)
    history: List[str] = field(default_factory=list)
    events: List[HistoryEvent] = field(default_factory=list)
    policy: Policy = field(default_factory=Policy)

    @staticmethod
    def validate_name(name: str) -> str:
        if len(name) > MAX_NAME_LENGTH or " " in name:
            raise ValueError(f"Invalid dataset name: {name}")
        return name

    def _record_event(self, op: str, msg: str, **params: Any) -> None:
        evt = HistoryEvent(op, msg, params=params or None)
        self.events.append(evt)
        self.history.append(msg)

    def log_cycle_metrics(self) -> None:
        for k in ["sigma_db", "coverage_frac"]:
            self.graph.__getattr__(k)()

    def add_document(
        self, doc_id: str, source: str, *, text: str | None = None
    ) -> None:
        self.graph.add_document(doc_id, source, text=text)
        self._record_event("add_document", doc_id, source=source)

    def add_section(self, doc_id: str, section_id: str) -> None:
        self.graph.add_section(doc_id, section_id)
        self._record_event("add_section", section_id)

    def add_chunk(self, doc_id: str, chunk_id: str, text: str) -> None:
        self.graph.add_chunk(doc_id, chunk_id, text)
        self._record_event("add_chunk", chunk_id)

    def export_prompts(self) -> List[Dict[str, Any]]:
        push_metrics({"prompts_exported": 1.0})
        self._record_event("export_prompts", "done")
        return [{"tag": "inferred"}]
