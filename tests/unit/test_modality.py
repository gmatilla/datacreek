from datacreek.utils import modality


def test_detect_modality_files(tmp_path):
    img = tmp_path / "x.png"
    img.write_text("")
    assert modality.detect_modality(str(img)) == "IMAGE"
    aud = tmp_path / "y.mp3"
    aud.write_text("")
    assert modality.detect_modality(str(aud)) == "AUDIO"
    code = tmp_path / "z.py"
    code.write_text("")
    assert modality.detect_modality(str(code)) == "CODE"
    txt = tmp_path / "t.txt"
    txt.write_text("")
    assert modality.detect_modality(str(txt)) == "TEXT"


def test_detect_modality_text():
    assert modality.detect_modality("um hello there") == "spoken"
    assert modality.detect_modality("Hello world.") == "written"
