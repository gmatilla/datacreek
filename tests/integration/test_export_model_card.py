import json
import subprocess
import sys


def test_export_model_card(tmp_path):
    json_path = tmp_path / "card.json"
    html_path = tmp_path / "card.html"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/export_model_card.py",
            str(json_path),
            "--html",
            str(html_path),
        ]
    )
    data = json.loads(json_path.read_text())
    assert "sigma_db" in data
    assert "code_commit" in data
    assert html_path.exists()
