import hashlib
import os

from datacreek.utils.checksum import md5_file


def test_md5_file(tmp_path):
    file_path = tmp_path / "f.txt"
    data = b"hello world"
    file_path.write_bytes(data)
    expected = hashlib.md5(data).hexdigest()
    assert md5_file(str(file_path)) == expected
