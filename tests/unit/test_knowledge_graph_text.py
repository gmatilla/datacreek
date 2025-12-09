import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


class DummySession:
    def __init__(self):
        self.last_query = None

    def run(self, q, **params):
        self.last_query = (q, params)
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyDriver:
    def __init__(self):
        self.session_obj = DummySession()

    def session(self):
        return self.session_obj


def build_kg():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src", text="doc text")
    kg.graph.nodes["d1"]["label"] = "document"
    kg.add_chunk("d1", "c1", "hello")
    kg.add_chunk("d1", "c2", "world")
    kg.index.build()
    return kg


def test_has_label_and_to_text():
    kg = build_kg()
    assert kg.has_label("document")
    assert not kg.has_label("unknown")
    text = kg.to_text()
    assert text == "hello\n\nworld"


def test_text_helpers_and_auto_tools(monkeypatch):
    kg = build_kg()
    kg.graph.add_edge("c1", "c2", relation="rel")
    assert "rel" in kg.path_to_text(["c1", "c2"])
    assert "rel" in kg.subgraph_to_text(["c1", "c2"])
    out = kg.graph_to_text()
    assert "hello" in out and "world" in out

    monkeypatch.setattr(
        "datacreek.utils.toolformer.insert_tool_calls", lambda t, p: t + "!"
    )
    res = kg.auto_tool_calls("c1", [("t", "h")])
    assert res.endswith("!")
    all_res = kg.auto_tool_calls_all([("t", "h")])
    assert {"c1", "c2"}.issubset(all_res.keys())
    assert kg.graph.nodes["c1"]["text"].endswith("!")


def test_set_property(monkeypatch):
    kg = build_kg()
    driver = DummyDriver()
    kg.set_property("prop", 1, driver=driver, dataset="ds")
    q, params = driver.session_obj.last_query
    assert "MERGE" in q and params["val"] == 1 and params["ds"] == "ds"

    with pytest.raises(ValueError):
        kg.set_property("1bad", 2)
