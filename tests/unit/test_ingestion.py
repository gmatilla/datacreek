import os
from pathlib import Path

import datacreek.analysis.ingestion as ing


def test_partition_files_to_atoms(tmp_path: Path):
    """Ensure text lines are returned when unstructured is unavailable."""
    path = tmp_path / "sample.txt"
    path.write_text("line1\n\nline2\n")
    atoms = ing.partition_files_to_atoms(str(path))
    assert atoms == ["line1", "line2"]


def test_transcribe_audio_no_dep(tmp_path: Path):
    """Without whisper installed an empty string is returned."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00\x00")
    assert ing.transcribe_audio(str(audio)) == ""


def test_blip_caption_image_no_dep(tmp_path: Path):
    """Without BLIP dependencies an empty caption is returned."""
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert ing.blip_caption_image(str(img)) == ""


def test_parse_code_to_atoms_success(tmp_path: Path):
    """Parse Python functions and classes from a file."""
    code = """\
class A:
    pass

def func():
    return 1
"""
    file_path = tmp_path / "code.py"
    file_path.write_text(code)
    atoms = ing.parse_code_to_atoms(str(file_path))
    assert any("class A" in a for a in atoms)
    assert any("def func" in a for a in atoms)


def test_parse_code_to_atoms_fallback(tmp_path: Path, monkeypatch):
    """Fallback returns raw text when parsing fails."""
    file_path = tmp_path / "bad.py"
    file_path.write_text("not python")
    monkeypatch.setattr(ing, "ast", None, raising=False)
    atoms = ing.parse_code_to_atoms(str(file_path))
    assert atoms == ["not python"]
