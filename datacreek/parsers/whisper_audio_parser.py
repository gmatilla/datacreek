"""Audio parser based on OpenAI Whisper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

try:  # optional dependency used for CPU int8 inference
    import torch
except Exception:  # pragma: no cover - dependency missing
    torch = None  # type: ignore[misc]

from .base import BaseParser

try:  # optional quantised matmul
    from bitsandbytes.functional import matmul_8bit as _matmul
except Exception:  # pragma: no cover - dependency missing
    _matmul = None

if TYPE_CHECKING:  # pragma: no cover - used only for type hints
    import torch as _torch

matmul_8bit: Optional[Callable[..., "_torch.Tensor"]] = _matmul


class WhisperAudioParser(BaseParser):  # type: ignore[misc]
    """Parse audio files using the Whisper model.

    The audio is first segmented using :func:`~datacreek.utils.audio_vad.split_on_silence`
    with ``webrtcvad.mode=3`` and short pauses (``<=300`` ms) are merged back.
    Each chunk is then transcribed independently and the results concatenated.
    This reduces the word error rate on long recordings.
    """

    def _chunk_audio(self, file_path: str) -> list[str]:
        """Return temporary WAV files for each speech segment."""

        try:
            import wave
            from tempfile import NamedTemporaryFile

            from datacreek.utils.audio_vad import split_on_silence
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "wave and webrtcvad are required for audio chunking"
            ) from exc

        with wave.open(file_path, "rb") as w:
            frames = w.readframes(w.getnframes())
            rate = w.getframerate()
            chans = w.getnchannels()
            width = w.getsampwidth()
        segments = split_on_silence(frames, rate, join_ms=300)
        if not segments:
            segments = [(0.0, len(frames) / (rate * chans * width))]

        paths: list[str] = []
        for start, end in segments:
            start_idx = int(start * rate) * chans * width
            end_idx = int(end * rate) * chans * width
            buf = frames[start_idx:end_idx]
            tmp = NamedTemporaryFile(delete=False, suffix=".wav")
            with wave.open(tmp, "wb") as out:
                out.setnchannels(chans)
                out.setsampwidth(width)
                out.setframerate(rate)
                out.writeframes(buf)
            paths.append(tmp.name)
        return paths

    def parse(self, file_path: str) -> str:
        """Return transcription of ``file_path``."""

        try:
            from datacreek.utils.whisper_batch import transcribe_audio_batch
        except Exception:
            transcribe_audio_batch = None

        try:
            chunk_paths = self._chunk_audio(file_path)
        except Exception:  # pragma: no cover - fallback when chunking fails
            chunk_paths = [file_path]

        if transcribe_audio_batch is not None:
            texts = transcribe_audio_batch(chunk_paths)
            return " ".join(str(t) for t in texts if t)

        try:
            import whisper
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The whisper library is required for audio parsing"
            ) from exc

        model = whisper.load_model("base")
        use_int8 = (
            matmul_8bit is not None
            and torch is not None
            and not torch.cuda.is_available()
        )
        if use_int8:
            orig_matmul = torch.matmul
            torch.matmul = matmul_8bit
        try:
            texts = []
            for path in chunk_paths:
                res = model.transcribe(path)
                texts.append(res.get("text", ""))
        finally:
            if use_int8:
                torch.matmul = orig_matmul
        return " ".join(t for t in texts if t)

    def save(self, content: str, output_path: str) -> None:
        # pragma: no cover - legacy
        """Persist ``content`` to ``output_path`` using parent helper."""
        super().save(content, output_path)
