# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Logic for saving file format
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import redis
except Exception:  # pragma: no cover - optional dependency missing

    class _RedisStub:
        Redis = object

    redis = _RedisStub()  # type: ignore

from datacreek.storage import StorageBackend

logger = logging.getLogger(__name__)
from datacreek.utils.format_converter import (
    to_alpaca,
    to_chatml,
    to_fine_tuning,
    to_hf_dataset,
    to_jsonl,
)

FORMAT_DISPATCH = {
    "jsonl": to_jsonl,
    "alpaca": to_alpaca,
    "ft": to_fine_tuning,
    "chatml": to_chatml,
}


def _format_pairs(qa_pairs: List[Dict[str, str]], fmt: str) -> Any:
    """Return formatted data for in-memory conversion."""

    if fmt not in FORMAT_DISPATCH:
        raise ValueError(f"Unknown format type: {fmt}")

    # Use dispatch table to build JSON string if no output path is provided
    if fmt == "jsonl":
        return "\n".join(json.dumps(p, ensure_ascii=False) for p in qa_pairs)
    if fmt == "alpaca":
        return json.dumps(
            [
                {"instruction": p["question"], "input": "", "output": p["answer"]}
                for p in qa_pairs
            ],
            indent=2,
        )

    messages_key = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": p["question"]},
            {"role": "assistant", "content": p["answer"]},
        ]
        for p in qa_pairs
    ]
    return json.dumps([{"messages": m} for m in messages_key], indent=2)


def convert_format(
    input_data: Any,
    format_type: str,
    config: Optional[Dict[str, Any]] = None,
    storage_format: str = "json",
    *,
    redis_client: "redis.Redis" | None = None,
    redis_key: str | None = None,
    backend: "StorageBackend" | None = None,
) -> Any:
    """Convert data to different formats

    Args:
        input_path: Path to the input file or JSON string
        format_type: Output format (jsonl, alpaca, ft, chatml)
        config: Configuration dictionary
        storage_format: Storage format, either "json" or "hf" (Hugging Face dataset)
        redis_client: Optional Redis connection for in-memory storage
        redis_key: Key used when storing the result in Redis

    Returns:
        The formatted data or the Redis key when using ``redis_client``
    """
    if format_type not in FORMAT_DISPATCH:
        raise ValueError(f"Unknown format type: {format_type}")

    if isinstance(input_data, str):
        if os.path.exists(input_data):
            with open(input_data, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(input_data)
    else:
        data = input_data

    # Extract data based on known structures
    # Try to handle the case where we have QA pairs or conversations
    if "qa_pairs" in data:
        qa_pairs = data.get("qa_pairs", [])
    elif "filtered_pairs" in data:
        qa_pairs = data.get("filtered_pairs", [])
    elif "conversations" in data:
        conversations = data.get("conversations", [])
        qa_pairs = []
        for conv in conversations:
            if (
                len(conv) >= 3
                and conv[1]["role"] == "user"
                and conv[2]["role"] == "assistant"
            ):
                qa_pairs.append(
                    {"question": conv[1]["content"], "answer": conv[2]["content"]}
                )
    else:
        # If the file is just an array of objects, check if they look like QA pairs
        if isinstance(data, list):
            qa_pairs = []
            for item in data:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    qa_pairs.append(item)
        else:
            raise ValueError(
                "Unrecognized data format - expected QA pairs or conversations"
            )

    # When using HF dataset storage format
    if storage_format == "hf":
        formatted_pairs = (
            qa_pairs
            if format_type == "jsonl"
            else json.loads(_format_pairs(qa_pairs, format_type))
        )
        if backend is not None and redis_key is not None:
            return to_hf_dataset(formatted_pairs, redis_key)
        raise ValueError("HF dataset export requires a storage backend and redis_key")

    # Standard JSON file storage format
    formatted = _format_pairs(qa_pairs, format_type)

    if backend is not None and redis_key is not None:
        try:
            return backend.save(redis_key, formatted)
        except Exception:
            logger.exception("Failed to save formatted data via backend")
            raise
    if redis_client and redis_key:
        try:
            redis_client.set(redis_key, formatted)
            return redis_key
        except Exception:
            logger.exception("Failed to save formatted data to Redis")
            raise
    return formatted
