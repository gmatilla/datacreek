import sys
import types

import pytest

import datacreek.generators as gens


def test_getattr_dynamic_import(monkeypatch):
    names = [
        ("QAGenerator", "qa_generator"),
        ("KGGenerator", "kg_generator"),
        ("ConversationGenerator", "conversation_generator"),
        ("COTGenerator", "cot_generator"),
        ("MultiToolGenerator", "multi_tool_generator"),
        ("PrefListGenerator", "pref_generator"),
        ("PrefPairGenerator", "pref_generator"),
        ("ToolCallGenerator", "tool_generator"),
        ("VQAGenerator", "vqa_generator"),
    ]
    for cls_name, module_name in names:
        mod = types.ModuleType(f"datacreek.generators.{module_name}")
        dummy_cls = type(cls_name, (), {})
        setattr(mod, cls_name, dummy_cls)
        sys.modules[f"datacreek.generators.{module_name}"] = mod
        assert gens.__getattr__(cls_name) is dummy_cls

    with pytest.raises(AttributeError):
        gens.__getattr__("Missing")
