# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Factory helpers for various data generation pipelines."""
# Generator logic for both QA
__all__ = [
    "QAGenerator",
    "COTGenerator",
    "VQAGenerator",
    "KGGenerator",
    "ToolCallGenerator",
    "ConversationGenerator",
    "MultiToolGenerator",
    "PrefPairGenerator",
    "PrefListGenerator",
]


def __getattr__(name: str):
    if name == "ConversationGenerator":
        from .conversation_generator import ConversationGenerator as cls

        return cls
    if name == "COTGenerator":
        from .cot_generator import COTGenerator as cls

        return cls
    if name == "KGGenerator":
        from .kg_generator import KGGenerator as cls

        return cls
    if name == "MultiToolGenerator":
        from .multi_tool_generator import MultiToolGenerator as cls

        return cls
    if name == "PrefListGenerator":
        from .pref_generator import PrefListGenerator as cls

        return cls
    if name == "PrefPairGenerator":
        from .pref_generator import PrefPairGenerator as cls

        return cls
    if name == "QAGenerator":
        from .qa_generator import QAGenerator as cls

        return cls
    if name == "ToolCallGenerator":
        from .tool_generator import ToolCallGenerator as cls

        return cls
    if name == "VQAGenerator":
        from .vqa_generator import VQAGenerator as cls

        return cls
    raise AttributeError(name)
