import numpy as np

from datacreek.analysis.compression import fp8_dequantize, fp8_quantize


def test_fp8_roundtrip():
    x = np.array([0.1, -0.25, 0.0], dtype=np.float32)
    q, s = fp8_quantize(x)
    restored = fp8_dequantize(q, s)
    assert np.allclose(restored, x, atol=1e-2)
