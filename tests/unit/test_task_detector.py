"""Tests for :mod:`training.task_detector`."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure repository root on sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from training.task_detector import detect_task, format_classif, format_rlhf, format_sft


class DummyDataset:
    """Minimal dataset stub exposing ``info`` and ``column_names``."""

    def __init__(self, records, metadata=None):
        self.data = records
        self.info = SimpleNamespace(metadata=metadata or {})
        self.column_names = list(records[0].keys()) if records else []


def test_detect_task_prefers_metadata():
    ds = DummyDataset([{"text": "hello"}], metadata={"task": "rlhf_ppo"})
    assert detect_task(ds) == "rlhf_ppo"


def test_detect_task_heuristics():
    ds = DummyDataset([{"prompt": "p", "chosen": "a", "rejected": "b"}])
    assert detect_task(ds) == "rlhf_dpo"

    ds = DummyDataset([{"question": "q", "answer": "a"}])
    assert detect_task(ds) == "qa"

    ds = DummyDataset([{"text": "t", "label": 1}])
    assert detect_task(ds) == "classification"

    ds = DummyDataset([{"prompt": "free"}])
    assert detect_task(ds) == "generation"


def test_format_sft_builds_prompt_and_text():
    sample = {"question": "Q?", "answer": "A"}
    out = format_sft(sample, eos_token="</s>")
    assert out["prompt"] == "Q?"
    assert out["text"] == "Q?\nA</s>"


def test_format_classif_appends_label_and_eos():
    sample = {"text": "news", "label": "sport"}
    out = format_classif(sample, eos_token="</s>")
    assert out["prompt"] == "news"
    assert out["labels"] == "sport"
    assert out["text"] == "news\nsport</s>"


def test_format_rlhf_returns_chosen_and_rejected():
    sample = {"prompt": "p", "chosen": "c", "rejected": "r"}
    out = format_rlhf(sample, eos_token="</s>")
    assert out["prompt"] == "p"
    assert out["chosen"] == "c</s>"
    assert out["rejected"] == "r</s>"
