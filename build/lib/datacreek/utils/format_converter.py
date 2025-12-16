# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Utils for format conversions
import json
from datetime import datetime
from typing import Any, Dict, List, Optional


def to_jsonl(data: List[Dict[str, Any]]) -> str:
    """Return data in JSONL format as a string."""
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in data)


def to_alpaca(qa_pairs: List[Dict[str, str]]) -> str:
    """Return QA pairs in Alpaca format as a JSON string."""
    alpaca_data = [
        {"instruction": p["question"], "input": "", "output": p["answer"]}
        for p in qa_pairs
    ]
    return json.dumps(alpaca_data, indent=2)


def to_fine_tuning(qa_pairs: List[Dict[str, str]]) -> str:
    """Return QA pairs in the OpenAI fine-tuning format."""
    ft_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p["question"]},
                {"role": "assistant", "content": p["answer"]},
            ]
        }
        for p in qa_pairs
    ]
    return json.dumps(ft_data, indent=2)


def to_chatml(qa_pairs: List[Dict[str, str]]) -> str:
    """Return QA pairs in ChatML JSONL format."""
    lines = []
    for pair in qa_pairs:
        chat = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]},
        ]
        lines.append(json.dumps({"messages": chat}))
    return "\n".join(lines)


def to_hf_dataset(qa_pairs: List[Dict[str, str]]) -> str:
    """
    Convert QA pairs to a Hugging Face dataset encoded as JSON.

    Args:
        qa_pairs: List of question-answer dictionaries

    Returns:
        JSON string representing the dataset
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for HF dataset format. "
            "Install it with: pip install datasets"
        )

    # Convert list of dicts to dict of lists for Dataset.from_dict()
    dict_of_lists = {}
    for key in qa_pairs[0].keys():
        dict_of_lists[key] = [item.get(key, "") for item in qa_pairs]

    # Create dataset
    dataset = Dataset.from_dict(dict_of_lists)
    return dataset.to_json()
