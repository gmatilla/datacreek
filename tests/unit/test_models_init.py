import pytest

import datacreek.models as m


def test_getattr_models():
    assert m.LLMClient.__name__ == "LLMClient"
    assert m.LLMService.__name__ == "LLMService"
    assert m.QAPair.__name__ == "QAPair"
    with pytest.raises(AttributeError):
        m.__getattr__("missing")


def test_all_exports():
    for name in m.__all__:
        assert hasattr(m, name)
