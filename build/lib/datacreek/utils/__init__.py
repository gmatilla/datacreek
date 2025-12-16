"""Utility helpers for datacreek."""

from contextlib import contextmanager

from .audio_vad import split_on_silence
from .backpressure import acquire_slot as acquire_ingest_slot
from .backpressure import has_capacity as ingest_has_capacity
from .backpressure import release_slot as release_ingest_slot

try:  # optional dependency on redis and pydantic
    from .cache import cache_l1
except Exception:  # pragma: no cover - fallback when dependencies missing
    cache_l1 = None  # type: ignore
from .chunking import chunk_by_sentences, chunk_by_tokens

try:  # optional dependency on pydantic/yaml for configuration helpers
    from .config import (
        get_curate_config,
        get_curate_settings,
        get_format_config,
        get_format_settings,
        get_generation_config,
        get_llm_settings,
        get_openai_config,
        get_openai_settings,
        get_prompt,
        get_vllm_config,
        get_vllm_settings,
        load_config,
        merge_configs,
    )
except Exception:  # pragma: no cover - fallback when heavy deps missing

    def _missing(*_a, **_k):
        raise RuntimeError("configuration utilities require pydantic and yaml")

    get_curate_config = get_curate_settings = get_format_config = (
        get_format_settings
    ) = get_generation_config = get_llm_settings = get_openai_config = (
        get_openai_settings
    ) = get_prompt = get_vllm_config = get_vllm_settings = load_config = (
        merge_configs
    ) = _missing  # type: ignore
from .crypto import decrypt_pii_fields, encrypt_pii_fields, xor_decrypt, xor_encrypt
from .dataset_cleanup import deduplicate_pairs
from .dataset_export import snapshot_template, snapshot_tokenizer
from .delta_export import delta_optimize, delta_vacuum, export_delta, lakefs_commit
from .gitinfo import get_commit_hash
from .kafka_queue import enqueue_ingest
from .rate_limit import consume_token
from .schema_evolution import add_column_if_missing

try:  # optional dependency on networkx
    from .graph_text import graph_to_text, neighborhood_to_sentence, subgraph_to_text
except Exception:  # pragma: no cover - lightweight fallback when networkx missing
    graph_to_text = neighborhood_to_sentence = subgraph_to_text = None  # type: ignore[assignment]
from .llm_processing import (
    convert_to_conversation_format,
    parse_qa_pairs,
    parse_ratings,
    qa_pairs_to_records,
)
from .metrics import push_metrics

try:  # optional dependency on rich
    # Only load real progress helpers if ``rich.progress`` is already imported.
    # This allows tests to provide a stub module before importing ``datacreek``.
    import sys

    if "rich.progress" not in sys.modules:
        raise ImportError
    import rich.progress as _rp  # noqa: F401

    # Skip using rich if a test stub injected a fake module.
    if getattr(_rp.Progress, "__module__", "rich.progress").startswith("test_"):
        raise ImportError

    def create_progress(*args, **kwargs):
        """Load :func:`create_progress` lazily to allow test stubs."""

        from .progress import create_progress as _create

        return _create(*args, **kwargs)

    @contextmanager
    def progress_context(*args, **kwargs):
        """Load :func:`progress_context` lazily to allow test stubs."""

        from .progress import progress_context as _context

        with _context(*args, **kwargs) as ctx:
            yield ctx

except Exception:  # pragma: no cover - fallback when rich is missing

    def create_progress(*_a, **_k):  # type: ignore[return-type]
        return None, 0

    @contextmanager  # type: ignore[misc]
    def progress_context(*_a, **_k):
        yield None, 0


from .audio_validator import AudioValidator
from .redis_helpers import decode_hash
from .redis_pid import get_current_ttl, start_pid_controller
from .text import clean_text, extract_json_from_text, normalize_units, split_into_chunks
from .toolformer import execute_tool_calls, insert_tool_calls


def __getattr__(name: str):
    """Lazily import heavy utilities to avoid circular imports."""
    if name == "extract_facts":
        from .fact_extraction import extract_facts as func

        return func
    if name == "extract_entities":
        from .entity_extraction import extract_entities as func

        return func
    if name in {
        "propose_merge_split",
        "record_feedback",
        "fine_tune_from_feedback",
    }:
        from .curation_agent import fine_tune_from_feedback as _ft
        from .curation_agent import propose_merge_split as _ps
        from .curation_agent import record_feedback as _rf

        mapping = {
            "propose_merge_split": _ps,
            "record_feedback": _rf,
            "fine_tune_from_feedback": _ft,
        }

        return mapping[name]
    raise AttributeError(name)


__all__ = [
    "load_config",
    "get_vllm_config",
    "get_vllm_settings",
    "get_openai_config",
    "get_openai_settings",
    "get_llm_settings",
    "get_generation_config",
    "get_curate_config",
    "get_curate_settings",
    "get_format_config",
    "get_format_settings",
    "get_prompt",
    "merge_configs",
    "split_into_chunks",
    "chunk_by_tokens",
    "chunk_by_sentences",
    "split_on_silence",
    "extract_json_from_text",
    "clean_text",
    "normalize_units",
    "parse_qa_pairs",
    "parse_ratings",
    "create_progress",
    "progress_context",
    "convert_to_conversation_format",
    "qa_pairs_to_records",
    "deduplicate_pairs",
    "get_commit_hash",
    "extract_facts",
    "extract_entities",
    "decode_hash",
    "neighborhood_to_sentence",
    "subgraph_to_text",
    "graph_to_text",
    "insert_tool_calls",
    "execute_tool_calls",
    "xor_encrypt",
    "xor_decrypt",
    "encrypt_pii_fields",
    "decrypt_pii_fields",
    "push_metrics",
    "cache_l1",
    "start_pid_controller",
    "get_current_ttl",
    "acquire_ingest_slot",
    "release_ingest_slot",
    "ingest_has_capacity",
    "consume_token",
    "enqueue_ingest",
    "add_column_if_missing",
    "propose_merge_split",
    "record_feedback",
    "fine_tune_from_feedback",
    "snapshot_tokenizer",
    "snapshot_template",
    "AudioValidator",
]
