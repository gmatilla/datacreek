import math

import numpy as np
from PIL import Image, ImageFilter

from datacreek.utils.quality_metrics import (
    audio_metrics,
    audio_snr,
    blur_score,
    dynamic_snr_threshold,
    image_dimensions,
    snr_soft_gate,
    snr_std,
    text_entropy,
)


def _make_grid(path):
    img = Image.new("L", (8, 8))
    for i in range(8):
        for j in range(8):
            img.putpixel((i, j), 0 if (i + j) % 2 == 0 else 255)
    img.save(path)


def test_blur_score(tmp_path):
    sharp_path = tmp_path / "sharp.png"
    blur_path = tmp_path / "blur.png"
    _make_grid(sharp_path)
    Image.open(sharp_path).filter(ImageFilter.GaussianBlur(radius=1)).save(blur_path)
    assert blur_score(str(blur_path)) > blur_score(str(sharp_path))


def test_text_entropy():
    assert text_entropy("aaaa") == 0.0
    assert math.isclose(text_entropy("abcd"), 2.0, rel_tol=1e-5)


def test_audio_snr():
    sr = 8000
    t = np.arange(sr, dtype=float) / sr
    signal = 10000 * np.sin(2 * math.pi * 440 * t)
    noise = 1000 * np.random.randn(sr)
    pcm = np.asarray(signal + noise, dtype=np.int16).tobytes()
    assert audio_snr(pcm) > 10.0


def test_audio_snr_constant():
    pcm = (np.ones(1600, dtype=np.int16) * 1000).tobytes()
    assert audio_snr(pcm) == float("inf")


def test_image_dimensions(tmp_path):
    img_path = tmp_path / "img.png"
    _make_grid(img_path)
    assert image_dimensions(str(img_path)) == (8, 8)


def test_audio_metrics(tmp_path):
    wav = tmp_path / "snd.wav"
    sr = 8000
    t = np.arange(sr, dtype=float) / sr
    tone = np.asarray(10000 * np.sin(2 * math.pi * 440 * t), dtype=np.int16)
    import wave

    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(tone.tobytes())

    sr_out, dur, snr = audio_metrics(str(wav))
    assert sr_out == sr
    assert math.isclose(dur, 1.0, rel_tol=1e-3)
    assert snr > 10.0


def test_dynamic_snr_threshold_and_gate():
    history = [10.0, 12.0, 11.0, 9.0, 10.0]
    sigma = snr_std(history)
    assert math.isclose(sigma, 1.0198039, rel_tol=1e-5)
    thr = dynamic_snr_threshold(history)
    assert math.isclose(thr, 6 + 0.5 * sigma, rel_tol=1e-5)
    assert snr_soft_gate(thr + 0.1, history)
    assert not snr_soft_gate(thr - 0.1, history)
