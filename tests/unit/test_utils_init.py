import sys
import types

import pytest

import datacreek.utils as utils


def test_getattr_dynamic(monkeypatch):
    fake_facts = types.SimpleNamespace(extract_facts=lambda x: "facts")
    monkeypatch.setitem(sys.modules, "datacreek.utils.fact_extraction", fake_facts)
    assert utils.__getattr__("extract_facts")("txt") == "facts"

    fake_curation = types.SimpleNamespace(
        propose_merge_split=lambda: "ps",
        record_feedback=lambda: "rf",
        fine_tune_from_feedback=lambda: "ft",
    )
    monkeypatch.setitem(sys.modules, "datacreek.utils.curation_agent", fake_curation)
    assert utils.__getattr__("propose_merge_split") is fake_curation.propose_merge_split
    assert utils.__getattr__("record_feedback") is fake_curation.record_feedback
    assert (
        utils.__getattr__("fine_tune_from_feedback")
        is fake_curation.fine_tune_from_feedback
    )

    with pytest.raises(AttributeError):
        utils.__getattr__("unknown")
