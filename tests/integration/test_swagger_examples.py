import importlib
import json

from datacreek.api import app


def test_swagger_examples_present():
    spec = app.openapi()
    explain = spec["paths"]["/explain/{node}"]["get"]
    params = {p["name"]: p for p in explain["parameters"]}
    assert "dataset" in params
    assert "example" in params["dataset"]

    search = spec["paths"]["/vector/search"]["post"]
    ref = search["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    name = ref.split("/")[-1]
    body = spec["components"]["schemas"][name]
    assert "example" in body["properties"]["dataset"]
    assert "example" in body["properties"]["query"]
