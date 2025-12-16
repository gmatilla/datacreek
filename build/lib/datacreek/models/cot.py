from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class COTExample:
    """Representation of a chain-of-thought example."""

    question: str
    reasoning: str
    answer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "reasoning": self.reasoning,
            "answer": self.answer,
        }
