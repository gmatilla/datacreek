from .base import BaseParser


class AudioParser(BaseParser):
    """Parser for audio files using SpeechRecognition."""

    def parse(self, file_path: str) -> str:
        try:
            import speech_recognition as sr
        except Exception as exc:  # pragma: no cover - dependency missing
            raise ImportError(
                "SpeechRecognition and pocketsphinx are required for audio parsing."
            ) from exc
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        try:
            return recognizer.recognize_sphinx(audio)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:  # pragma: no cover - unlikely
            raise RuntimeError("Sphinx request error") from exc

    def save(self, content: str, output_path: str) -> None:  # pragma: no cover - legacy
        super().save(content, output_path)
