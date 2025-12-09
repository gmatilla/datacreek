"""Extract factual triples from text using LLMs or regex."""

import json
import re
from typing import Dict, List, Optional

from datacreek.models.llm_client import LLMClient
from datacreek.utils.config import get_prompt, load_config


def extract_facts(
    text: str,
    client: Optional[LLMClient] = None,
    *,
    config: Optional[dict] = None,
) -> List[Dict[str, str]]:
    """Extract atomic facts from ``text`` using an LLM if provided.

    The returned items follow the schema ``{"subject", "predicate", "object"}``.
    When ``client`` is ``None`` a naive regex extractor is used for testing.
    """
    if client is None:
        pattern = r"([A-Za-z ]+) is (?:the )?([A-Za-z ]+)\."
        pattern2 = r"([A-Za-z ]+) was born in ([A-Za-z ]+)\."
        facts = []
        for m in re.finditer(pattern, text):
            subj = m.group(1).strip()
            obj = m.group(2).strip()
            facts.append({"subject": subj, "predicate": "is", "object": obj})
        for m in re.finditer(pattern2, text):
            subj = m.group(1).strip()
            obj = m.group(2).strip()
            facts.append({"subject": subj, "predicate": "born_in", "object": obj})
        return facts

    cfg = config or load_config()
    prompt = get_prompt(cfg, "fact_extraction")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    response = client.chat_completion(messages, temperature=0)
    try:
        data = json.loads(response)
    except Exception:
        return []
    out = []
    if isinstance(data, list):
        for item in data:
            if not all(k in item for k in ("subject", "predicate", "object")):
                continue
            out.append(
                {
                    "subject": item["subject"],
                    "predicate": item["predicate"],
                    "object": item["object"],
                }
            )
    return out
