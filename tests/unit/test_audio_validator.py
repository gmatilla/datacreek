import math

import numpy as np

from datacreek.analysis import monitoring
from datacreek.utils.audio_validator import AudioValidator


def _noise(duration=1600):
    return (np.random.randn(duration).astype(np.int16)).tobytes()


def _tone(duration=1600, snr_db=20.0, sr=1600):
    signal = np.ones(duration) * 10000
    noise = (10 ** (-snr_db / 20.0)) * np.random.randn(duration) * 10000
    pcm = signal + noise
    return np.asarray(pcm, dtype=np.int16).tobytes()


def test_audio_validator_updates_metric(monkeypatch):
    called = {}

    def fake_update(name, value, labels=None):
        called["name"] = name
        called["value"] = value

    monkeypatch.setattr(monitoring, "update_metric", fake_update)
    v = AudioValidator(window=10)
    pcm = _tone()
    assert v.validate(pcm)
    assert called["name"] == "snr_dynamic_thr"
    assert math.isclose(called["value"], 6.0, rel_tol=1e-5)


def test_audio_validator_rejects_noise():
    v = AudioValidator(window=50)
    false_pos = 0
    for _ in range(100):
        pcm = _noise()
        if v.validate(pcm):
            false_pos += 1
    assert false_pos < 2
