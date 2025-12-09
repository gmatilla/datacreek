"""Simple Named Entity Recognition helpers."""

from __future__ import annotations

import re
from typing import List, Optional


def extract_entities(text: str, model: str | None = "en_core_web_sm") -> List[str]:
    """Return entity strings found in ``text``.

    When ``model`` is provided and spaCy is available, it will be used for
    extraction. Otherwise a naive capitalized word pattern is applied.
    """

    if model:
        try:
            import spacy  # type: ignore

            nlp = spacy.load(model)  # type: ignore[arg-type]
        except Exception:
            nlp = None
    else:
        nlp = None

    if nlp is not None:
        doc = nlp(text)
        return [ent.text for ent in doc.ents]

    pattern = r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b"
    return list({m.group(0) for m in re.finditer(pattern, text)})
