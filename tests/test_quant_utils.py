"""Tests for SmoothQuant scale computation and bfloat4 quantization."""

import pytest
import torch

from training.quant_utils import quantize_bfloat4, smoothquant_group_scales


def test_smoothquant_scales_clip_outliers() -> None:
    """Scales above the 99th percentile should be clipped."""
    group_size = 4
    # Create 100 groups: 99 normal, 1 outlier with 10x larger values
    weights = torch.ones(100 * group_size)
    weights = weights.reshape(100, group_size) * 127
    weights[-1] = 1270  # last group has scale 10
    raw_scales = weights.abs().max(dim=-1).values / 127.0
    p99 = torch.quantile(raw_scales, 0.99)
    clipped = smoothquant_group_scales(weights, group_size=group_size)
    assert torch.all(clipped <= p99)
    assert clipped[-1].item() == pytest.approx(p99.item())


def test_quantize_bfloat4_range() -> None:
    """Quantized values should lie within the signed 4-bit range."""
    torch.manual_seed(0)
    weights = torch.randn(2, 8) * 0.1
    q, scales = quantize_bfloat4(weights, group_size=4)
    assert q.dtype == torch.int8
    assert q.min() >= -8 and q.max() <= 7
    assert scales.shape[-1] == weights.shape[-1] // 4
