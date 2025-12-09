"""Tests for Accelerate ZeRO-3 configuration and gradient accumulation."""

from training.accelerate_utils import (
    compute_gradient_accumulation_steps,
    load_accelerate_config,
)


def test_gradient_accumulation_formula() -> None:
    assert compute_gradient_accumulation_steps(32, 16) == 2
    assert compute_gradient_accumulation_steps(33, 16) == 3


def test_zero3_enabled() -> None:
    cfg = load_accelerate_config()
    assert cfg["distributed_type"].upper() == "DEEPSPEED"
    assert cfg["deepspeed_config"]["zero_stage"] == 3
