"""Tests for NF4 quantization utilities."""

import importlib.machinery
import importlib.util
import subprocess
import sys
from pathlib import Path

import torch

# Ensure the lightweight scikit-learn stub installed by ``tests.conftest`` has a
# module specification so ``importlib.find_spec('sklearn')`` succeeds.
if "sklearn" in sys.modules:
    sys.modules["sklearn"].__spec__ = importlib.machinery.ModuleSpec(
        "sklearn", loader=None
    )

# Load the module directly to avoid executing ``training.__init__`` which has
# heavy side effects during test collection.
spec = importlib.util.spec_from_file_location(
    "quant_utils", (Path(__file__).resolve().parents[2] / "training" / "quant_utils.py")
)
quant_utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(quant_utils)


def test_merge_lora_weights_adds_delta():
    base = {"w": torch.zeros(2, 64)}
    lora = {"w": torch.ones(2, 64)}
    merged = quant_utils.merge_lora_weights(base, lora)
    assert torch.allclose(merged["w"], torch.ones_like(base["w"]))


def test_nf4_quantization_roundtrip(tmp_path):
    base = {"linear.weight": torch.randn(64, 64)}
    lora = {"linear.weight": torch.randn(64, 64) * 0.01}
    base_path = tmp_path / "base.pt"
    lora_path = tmp_path / "lora.pt"
    torch.save(base, base_path)
    torch.save(lora, lora_path)

    out_path = tmp_path / "model.gguf"
    subprocess.run(
        [
            sys.executable,
            "scripts/merge_lora_and_quantize.py",
            "--base",
            str(base_path),
            "--lora",
            str(lora_path),
            "--out",
            str(out_path),
        ],
        check=True,
    )

    assert out_path.exists()
    with open(out_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF"
        qstate = torch.load(f, weights_only=False)
    assert "linear.weight" in qstate
    param = qstate["linear.weight"]
    assert param.qweight.dtype == torch.uint8
    assert not torch.isnan(param.scale).any()

    # Dequantize and ensure values are finite
    recovered = quant_utils.dequantize_state_dict_nf4(qstate)["linear.weight"]
    assert torch.isfinite(recovered).all()


def test_overflow_guard_prevents_nan():
    huge = {"w": torch.full((1, 64), 1e6)}
    qstate = quant_utils.quantize_state_dict_nf4(huge, blocksize=64)
    recovered = quant_utils.dequantize_state_dict_nf4(qstate)["w"]
    assert not torch.isnan(recovered).any()


def test_no_nan_after_many_quant_steps():
    """Stress-test quantization/dequantization for 10k iterations.

    The NF4 overflow guard should keep the values finite even after many
    quantize/dequantize cycles, mirroring 10k optimization steps.
    """

    state = {"w": torch.randn(1, 8)}
    for _ in range(10000):
        qstate = quant_utils.quantize_state_dict_nf4(state, blocksize=8)
        state = {"w": quant_utils.dequantize_state_dict_nf4(qstate)["w"]}

    assert torch.isfinite(state["w"]).all()
