import os
import sys

from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL", "sqlite:///test_openapi.db")
os.environ.setdefault("DATACREEK_REQUIRE_PERSISTENCE", "0")
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)

from datacreek.api import app  # noqa: E402


def test_explain_route_in_openapi():
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    assert "/explain/{node}" in schema["paths"]  # noqa: S101


def test_explain_examples_in_openapi():
    """Ensure API documentation exposes usage snippets."""
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    desc = schema["paths"]["/explain/{node}"]["get"]["description"]
    assert "curl" in desc  # noqa: S101
    assert "fetch(" in desc  # noqa: S101
