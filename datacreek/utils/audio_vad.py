"""Audio chunking based on voice activity detection."""

from __future__ import annotations

from typing import List, Tuple

try:
    import webrtcvad
except Exception:  # pragma: no cover - optional dependency
    webrtcvad = None  # type: ignore


__all__ = ["split_on_silence"]


def split_on_silence(
    pcm: bytes,
    sample_rate: int,
    *,
    frame_ms: int = 30,
    join_ms: int = 300,
) -> List[Tuple[float, float]]:
    """Return start/end times of speech segments using VAD.

    Parameters
    ----------
    pcm:
        Raw 16-bit mono PCM audio data.
    sample_rate:
        Sample rate of the input audio in Hz.
    frame_ms:
        Frame size for VAD analysis in milliseconds.
    join_ms:
        Minimum silence duration required to split segments. Shorter gaps are
        merged back together.
    """

    if webrtcvad is None:  # pragma: no cover - optional dependency
        raise ImportError("webrtcvad is required for audio chunking")

    vad = webrtcvad.Vad(3)
    frame_len = int(sample_rate * frame_ms / 1000) * 2  # bytes per frame
    frames = [pcm[i : i + frame_len] for i in range(0, len(pcm), frame_len)]
    segments: List[Tuple[int, int]] = []
    start: int | None = None
    silence = 0
    for idx, frame in enumerate(frames):
        voiced = vad.is_speech(frame, sample_rate)
        if voiced:
            if start is None:
                start = idx
            silence = 0
        else:
            if start is not None:
                silence += 1
                if silence * frame_ms >= join_ms:
                    end = idx - silence + 1
                    segments.append((start, end))
                    start = None
                    silence = 0
    if start is not None:
        segments.append((start, len(frames)))

    return [(s * frame_ms / 1000.0, e * frame_ms / 1000.0) for s, e in segments]
