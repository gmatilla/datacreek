from pathlib import Path

from datacreek.utils.checksum import md5_file


def test_md5_file(tmp_path):
    p = tmp_path / "file.txt"
    p.write_text("hello")
    assert md5_file(str(p)) == "5d41402abc4b2a76b9719d911017c592"
