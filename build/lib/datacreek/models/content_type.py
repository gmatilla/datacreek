from enum import Enum


class ContentType(str, Enum):
    """Supported data generation content types."""

    QA = "qa"
    SUMMARY = "summary"
    COT = "cot"
    COT_ENHANCE = "cot-enhance"
    VQA_ADD_REASONING = "vqa_add_reasoning"
    FROM_KG = "from_kg"
    TOOL_CALL = "tool_call"
    CONVERSATION = "conversation"
    MULTI_TOOL = "multi_tool"
    PREF_PAIR = "pref_pair"
    PREF_LIST = "pref_list"
