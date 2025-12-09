import types

import pytest

from datacreek.core.dataset_light import MAX_NAME_LENGTH, DatasetBuilder
from datacreek.pipelines import DatasetType


class DummyMetrics:
    def __init__(self):
        self.data = None

    def __call__(self, metrics):
        self.data = metrics


class DummyGraph:
    def __init__(self):
        self.docs = []

    def add_document(self, doc_id, source, text=None):
        self.docs.append((doc_id, source, text))

    def add_section(self, doc_id, section_id):
        self.docs.append((doc_id, section_id))

    def add_chunk(self, doc_id, chunk_id, text):
        self.docs.append((chunk_id, text))

    def __getattr__(self, name):
        def f(*a, **k):
            return name

        return f


def test_builder_add_and_metrics(monkeypatch):
    dummy_metrics = DummyMetrics()
    monkeypatch.setattr("datacreek.core.dataset_light.push_metrics", dummy_metrics)
    builder = DatasetBuilder(dataset_type=DatasetType.QA, graph=DummyGraph())
    builder.add_document("d1", "src", text="hello")
    builder.add_section("d1", "s1")
    builder.add_chunk("d1", "c1", "chunktext")
    builder.log_cycle_metrics()
    out = builder.export_prompts()
    assert out == [{"tag": "inferred"}]
    assert dummy_metrics.data == {"prompts_exported": 1.0}
    assert len(builder.events) == 4


def test_validate_name_errors():
    builder = DatasetBuilder(dataset_type=DatasetType.QA)
    good = "a" * MAX_NAME_LENGTH
    assert builder.validate_name(good) == good
    for bad in ["with space", "a" * (MAX_NAME_LENGTH + 1)]:
        with pytest.raises(ValueError):
            builder.validate_name(bad)
