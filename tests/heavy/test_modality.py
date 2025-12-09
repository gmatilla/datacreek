import os
import tempfile

import pytest

from datacreek.utils import modality


@pytest.mark.heavy
def test_detect_modality_files(tmp_path):
    text_file = tmp_path / "a.txt"
    audio_file = tmp_path / "b.mp3"
    code_file = tmp_path / "c.py"
    image_file = tmp_path / "d.png"
    for f in [text_file, audio_file, code_file, image_file]:
        f.write_text("x")
    assert modality.detect_modality(str(text_file)) == "TEXT"
    assert modality.detect_modality(str(audio_file)) == "AUDIO"
    assert modality.detect_modality(str(code_file)) == "CODE"
    assert modality.detect_modality(str(image_file)) == "IMAGE"


@pytest.mark.heavy
def test_detect_modality_text():
    assert modality.detect_modality("um yeah") == "spoken"
    assert modality.detect_modality("hello world.") == "written"
    long_spoken = "word " * 10
    assert modality.detect_modality(long_spoken) == "spoken"
