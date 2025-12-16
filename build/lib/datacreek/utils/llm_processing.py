# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Output utilities
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Union

from datacreek.models.qa import QAPair

logger = logging.getLogger(__name__)


def parse_qa_pairs(text: str) -> List[QAPair]:
    """Parse QA pairs from LLM output with enhanced error handling"""
    verbose = logger.isEnabledFor(logging.DEBUG)

    if verbose:
        logger.debug("Parsing response of length %d", len(text))  # pragma: no cover

    try:
        # Try direct JSON parsing
        if "[" in text and "]" in text:
            # Find the first [ and last ]
            start = text.find("[")
            end = text.rfind("]") + 1
            json_text = text[start:end]

            # Try to clean up the JSON to fix common issues
            cleaned_text = re.sub(
                r"(\n\s*|\r\s*)", " ", json_text
            )  # Remove newlines and extra spaces
            cleaned_text = re.sub(
                r",(\s*\}|\s*\])", r"\1", cleaned_text
            )  # Remove trailing commas

            try:
                pairs = json.loads(cleaned_text)
                if verbose:
                    logger.debug(
                        "Successfully parsed %d QA pairs", len(pairs)
                    )  # pragma: no cover
                return [
                    QAPair(question=p.get("question", ""), answer=p.get("answer", ""))
                    for p in pairs
                    if isinstance(p, dict) and "question" in p and "answer" in p
                ]
            except json.JSONDecodeError as e:  # pragma: no cover
                if verbose:
                    logger.debug(
                        "Direct JSON parsing failed: %s", e
                    )  # pragma: no cover
                    logger.debug(
                        "Attempted to parse: %s...", cleaned_text[:200]
                    )  # pragma: no cover
    except Exception as e:  # pragma: no cover
        if verbose:
            logger.debug("Error during JSON extraction: %s", e)  # pragma: no cover

    # Fallback to regex pattern matching
    if verbose:
        logger.debug("Falling back to regex pattern matching")  # pragma: no cover
    qa_pattern = (
        r'"question":\s*"((?:[^"\\]|\\.)*)"\s*,\s*"answer":\s*"((?:[^"\\]|\\.)*)"\s*'
    )
    pairs = []

    for match in re.finditer(qa_pattern, text):
        try:
            q = match.group(1).replace('\\"', '"')
            a = match.group(2).replace('\\"', '"')
            pairs.append(QAPair(question=q, answer=a))
        except Exception as e:  # pragma: no cover
            if verbose:
                logger.debug("Error extracting pair: %s", e)  # pragma: no cover

    if pairs:
        if verbose:
            logger.debug(
                "Extracted %d QA pairs with regex", len(pairs)
            )  # pragma: no cover
    else:
        if verbose:
            logger.debug(
                "No QA pairs extracted. Check the model output format."
            )  # pragma: no cover
        else:
            logger.error(
                "Failed to parse QA pairs from response: %s", text[:100]
            )  # pragma: no cover

    return pairs


def parse_ratings(
    text: str, original_items: List[Dict[str, str]] = None
) -> List[QAPair]:
    """Parse rated items from LLM output

    Attempts to parse JSON from LLM response. Will raise an exception if
    parsing fails. Never adds default ratings - either the model returns valid
    ratings or the function will crash.

    Args:
        text: LLM response text to parse
        original_items: Original QA pairs (ignored - no defaults used)

    Returns:
        List of items with ratings from the LLM

    Raises:
        ValueError: If the response cannot be parsed as valid JSON
    """
    verbose = logger.isEnabledFor(logging.DEBUG)

    def _meta(idx: int) -> Dict[str, Any]:
        if not original_items:
            return {"chunk": None, "source": None}
        try:
            item = original_items[idx]
        except Exception:  # pragma: no cover
            return {"chunk": None, "source": None}
        if isinstance(item, QAPair):
            return {"chunk": item.chunk, "source": item.source}
        if isinstance(item, dict):
            return {"chunk": item.get("chunk"), "source": item.get("source")}
        return {"chunk": None, "source": None}

    if verbose:
        logger.debug(
            "Parsing ratings response of length %d", len(text)
        )  # pragma: no cover
        logger.debug("Raw response: %r", text[:500])  # pragma: no cover

    # The multiple passes are to for edge cases that emerge when using 8B or smaller models for generating synthetic data. This is to make a comprehensive parser for faster protoyping.
    # With 70B or bigger model, `json.load()` should "just work"
    try:
        # Handle the common case of indented JSON with newlines
        # First, remove any markdown or text before/after the JSON
        # Look for standard JSON start/end markers
        json_content = text.strip()

        # Try to normalize escape sequences
        json_content = (
            json_content.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
        )

        # Check if we have a JSON object
        if "{" in json_content and "}" in json_content:
            start_idx = json_content.find("{")
            end_idx = json_content.rfind("}") + 1
            json_text = json_content[start_idx:end_idx]

            # Clean up the JSON string to handle common issues
            # First, convert newlines to spaces in JSON
            json_text = re.sub(r"\s*\n\s*", " ", json_text)

            # Now, try to parse it
            try:
                parsed = json.loads(json_text)
                if isinstance(parsed, dict) and "rating" in parsed:
                    if verbose:
                        logger.debug(
                            "Successfully parsed single JSON object"
                        )  # pragma: no cover
                    meta = _meta(0)
                    return [
                        QAPair(
                            question=parsed.get("question", ""),
                            answer=parsed.get("answer", ""),
                            rating=float(parsed["rating"]),
                            chunk=meta["chunk"],
                            source=meta["source"],
                        )
                    ]
            except json.JSONDecodeError as e:
                if verbose:
                    logger.debug(
                        "JSON parse error for object: %s", e
                    )  # pragma: no cover

        # Check if we have a JSON array
        if "[" in json_content and "]" in json_content:
            start_idx = json_content.find("[")
            end_idx = json_content.rfind("]") + 1
            json_text = json_content[start_idx:end_idx]

            # Clean up the JSON string
            json_text = re.sub(r"\s*\n\s*", " ", json_text)

            try:
                parsed = json.loads(json_text)
                if isinstance(parsed, list):
                    for item in parsed:
                        if not isinstance(item, dict) or "rating" not in item:
                            if verbose:
                                logger.debug(
                                    "Array contains invalid item: %s", item
                                )  # pragma: no cover
                            return []
                    if verbose:
                        logger.debug(
                            "Successfully parsed %d items in JSON array", len(parsed)
                        )  # pragma: no cover
                    result = []
                    for idx, item in enumerate(parsed):
                        meta = _meta(idx)
                        result.append(
                            QAPair(
                                question=item.get("question", ""),
                                answer=item.get("answer", ""),
                                rating=float(item["rating"]),
                                chunk=meta["chunk"],
                                source=meta["source"],
                            )
                        )
                    return result
            except json.JSONDecodeError as e:  # pragma: no cover
                if verbose:
                    logger.debug(
                        "JSON parse error for array: %s", e
                    )  # pragma: no cover

    except Exception as e:  # pragma: no cover
        if verbose:
            logger.debug("Error in primary parsing approach: %s", e)  # pragma: no cover

    # Fallback to more specific methods
    # Method 1: Code block extraction
    try:
        code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
        if code_blocks:
            for block in code_blocks:
                try:
                    # Clean up newlines in the code block
                    clean_block = re.sub(r"\s*\n\s*", " ", block.strip())
                    parsed = json.loads(clean_block)
                    if isinstance(parsed, dict) and "rating" in parsed:
                        if verbose:
                            logger.debug(
                                "Successfully parsed from code block (single object)"
                            )  # pragma: no cover
                        meta = _meta(0)
                        return [
                            QAPair(
                                question=parsed.get("question", ""),
                                answer=parsed.get("answer", ""),
                                rating=float(parsed["rating"]),
                                chunk=meta["chunk"],
                                source=meta["source"],
                            )
                        ]
                    elif isinstance(parsed, list):
                        valid_items = True
                        for item in parsed:
                            if not isinstance(item, dict) or "rating" not in item:
                                valid_items = False
                                break
                        if valid_items and len(parsed) > 0:
                            if verbose:
                                logger.debug(  # pragma: no cover
                                    "Successfully parsed %d items from code block",
                                    len(parsed),
                                )
                            result = []
                            for idx, item in enumerate(parsed):
                                meta = _meta(idx)
                                result.append(
                                    QAPair(
                                        question=item.get("question", ""),
                                        answer=item.get("answer", ""),
                                        rating=float(item["rating"]),
                                        chunk=meta["chunk"],
                                        source=meta["source"],
                                    )
                                )
                            return result
                except json.JSONDecodeError:  # pragma: no cover
                    pass
    except Exception as e:  # pragma: no cover
        if verbose:
            logger.debug("Error in code block extraction: %s", e)  # pragma: no cover

    result = _parse_regex(text, _meta, verbose)
    if result is not None:
        return result

    # Method 3: Try using json5 if available (more lenient parser)
    result = _parse_json5(text, _meta, verbose)
    if result is not None:
        return result

    result = _parse_line_by_line(text, original_items)
    if result is not None:
        return result

    # If we reach here, we couldn't extract valid JSON
    if verbose:
        logger.debug("All parsing methods failed")  # pragma: no cover

    # Instead of a generic error message, include part of the response
    error_snippet = text[:100] if len(text) > 100 else text
    raise ValueError(f"Could not parse JSON with ratings: {error_snippet}")


def _parse_line_by_line(
    text: str, original_items, verbose: bool = False
):  # pragma: no cover
    """Slow fallback that searches for ratings next to questions."""
    try:
        if original_items and len(original_items) > 0:
            found_items = []
            for item in original_items:
                question_escaped = re.escape(item.get("question", ""))
                pattern = f'.*{question_escaped}.*"rating"\\s*:\\s*(\\d+(?:\\.\\d+)?)'
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    try:
                        rating = float(match.group(1))
                        meta = {
                            "chunk": (
                                item.get("chunk")
                                if isinstance(item, dict)
                                else getattr(item, "chunk", None)
                            ),
                            "source": (
                                item.get("source")
                                if isinstance(item, dict)
                                else getattr(item, "source", None)
                            ),
                        }
                        found_items.append(
                            QAPair(
                                question=item.get("question", ""),
                                answer=item.get("answer", ""),
                                rating=rating,
                                chunk=meta["chunk"],
                                source=meta["source"],
                            )
                        )
                        if verbose:
                            logger.debug(  # pragma: no cover
                                "Found rating %s for question: %s...",
                                rating,
                                item.get("question", "")[:30],
                            )
                    except Exception:
                        pass
            if found_items:
                if verbose:
                    logger.debug(
                        "Extracted %d ratings using pattern matching", len(found_items)
                    )  # pragma: no cover
                return found_items
    except Exception as e:
        if verbose:
            logger.debug("Error in final extraction attempt: %s", e)  # pragma: no cover
    return None


def _parse_regex(text: str, _meta, verbose: bool):  # pragma: no cover
    try:
        json_patterns = [
            r'(\{\s*"question"\s*:\s*"[^"]*"\s*,\s*"answer"\s*:\s*"[^"]*"\s*,\s*"rating"\s*:\s*\d+(?:\.\d+)?\s*\})',
            r'(\[\s*\{\s*"question"\s*:.*"rating"\s*:\s*\d+(?:\.\d+)?\s*\}\s*\])',
        ]
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        clean_match = re.sub(r"\s*\n\s*", " ", match)
                        parsed = json.loads(clean_match)
                        if isinstance(parsed, dict) and "rating" in parsed:
                            if verbose:
                                logger.debug(
                                    "Successfully parsed using regex (single object)"
                                )  # pragma: no cover
                            meta = _meta(0)
                            return [
                                QAPair(
                                    question=parsed.get("question", ""),
                                    answer=parsed.get("answer", ""),
                                    rating=float(parsed["rating"]),
                                    chunk=meta["chunk"],
                                    source=meta["source"],
                                )
                            ]
                        elif isinstance(parsed, list) and all(
                            "rating" in item for item in parsed
                        ):
                            if verbose:
                                logger.debug(
                                    "Successfully parsed %d items using regex",
                                    len(parsed),
                                )  # pragma: no cover
                            result = []
                            for idx, item in enumerate(parsed):
                                meta = _meta(idx)
                                result.append(
                                    QAPair(
                                        question=item.get("question", ""),
                                        answer=item.get("answer", ""),
                                        rating=float(item["rating"]),
                                        chunk=meta["chunk"],
                                        source=meta["source"],
                                    )
                                )
                            return result
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        if verbose:
            logger.debug("Error in regex extraction: %s", e)  # pragma: no cover
    return None


def _parse_json5(text: str, _meta, verbose: bool):  # pragma: no cover
    try:
        import json5

        try:
            parsed = json5.loads(text)
            if isinstance(parsed, dict) and "rating" in parsed:
                if verbose:
                    logger.debug(
                        "Successfully parsed using json5 (single object)"
                    )  # pragma: no cover
                meta = _meta(0)
                return [
                    QAPair(
                        question=parsed.get("question", ""),
                        answer=parsed.get("answer", ""),
                        rating=float(parsed["rating"]),
                        chunk=meta["chunk"],
                        source=meta["source"],
                    )
                ]
            elif isinstance(parsed, list) and all("rating" in item for item in parsed):
                if verbose:
                    logger.debug(
                        "Successfully parsed %d items using json5", len(parsed)
                    )  # pragma: no cover
                result = []
                for idx, item in enumerate(parsed):
                    meta = _meta(idx)
                    result.append(
                        QAPair(
                            question=item.get("question", ""),
                            answer=item.get("answer", ""),
                            rating=float(item["rating"]),
                            chunk=meta["chunk"],
                            source=meta["source"],
                        )
                    )
                return result
        except Exception:
            pass
    except ImportError:
        if verbose:
            logger.debug("json5 not available")  # pragma: no cover
    return None


def convert_to_conversation_format(
    qa_pairs: List[Union[Dict[str, str], QAPair]],
    system_prompt: Optional[str] = None,
) -> List[List[Dict[str, str]]]:
    """Convert QA pairs to conversation format.

    ``qa_pairs`` can contain either raw dictionaries or :class:`QAPair` objects.
    """
    if system_prompt is None:
        system_prompt = (
            "You are a helpful AI assistant that provides accurate, detailed responses."
        )

    conversations = []
    for pair in qa_pairs:
        if isinstance(pair, QAPair):
            q = pair.question
            a = pair.answer
        else:
            q = pair.get("question", "")
            a = pair.get("answer", "")

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        conversations.append(conversation)

    return conversations


def qa_pairs_to_records(
    qa_pairs: List[QAPair],
    *,
    system_prompt: Optional[str] = None,
    modify: Optional[Callable[[List[Dict[str, str]], QAPair], None]] = None,
) -> List[Dict[str, Any]]:
    """Return conversation records with metadata for ``qa_pairs``."""

    convs = convert_to_conversation_format(qa_pairs, system_prompt)
    records: List[Dict[str, Any]] = []
    for pair, conv in zip(qa_pairs, convs):
        if modify:
            modify(conv, pair)
        records.append(
            {"conversations": conv, "chunk": pair.chunk, "source": pair.source}
        )
    return records
