import json
from pathlib import Path


def test_gpu_dashboard_contains_metric() -> None:
    """Ensure the GPU dashboard references the ``gpu_minutes_total`` metric."""
    path = Path(__file__).resolve().parents[2] / "docs" / "grafana" / "gpu_minutes.json"
    data = json.loads(path.read_text())
    text = json.dumps(data)
    assert "gpu_minutes_total" in text
    assert "{{tenant}}" in text
