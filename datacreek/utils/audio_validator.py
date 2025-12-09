from __future__ import annotations

"""Audio validator with adaptive SNR threshold."""

from collections import deque
from typing import Deque

from datacreek.analysis import monitoring

from .quality_metrics import audio_snr, dynamic_snr_threshold

__all__ = ["AudioValidator"]


class AudioValidator:
    r"""Validate audio clips using a dynamic SNR gate.

    The validator maintains a rolling window of the last :math:`N` signal-to-noise
    ratio (SNR) measurements.  The adaptive threshold is computed as

    .. math::
        \mathrm{thr} = 6 + 0.5\,\sigma_{SNR},

    where :math:`\sigma_{SNR}` is the standard deviation of the collected SNR
    values.  A higher variance thus raises the bar for accepting new audio
    clips.

    Parameters
    ----------
    window:
        Maximum number of recent SNR values to keep in the history (default
        500).
    """

    def __init__(self, window: int = 500) -> None:
        self.history: Deque[float] = deque(maxlen=window)

    def validate(self, pcm: bytes) -> bool:
        """Return ``True`` if ``pcm`` passes the dynamic SNR threshold.

        The current threshold is exported via the Prometheus metric
        ``snr_dynamic_thr`` when the monitoring subsystem is available.
        ``pcm`` is expected to contain 16-bit mono PCM samples.
        """

        snr = audio_snr(pcm)
        threshold = dynamic_snr_threshold(self.history)
        try:  # optional Prometheus metric
            monitoring.update_metric("snr_dynamic_thr", threshold)
        except Exception:  # pragma: no cover - metrics optional
            pass
        passed = snr >= threshold
        self.history.append(snr)
        return passed
