import logging
import os

import pytest

import datacreek.utils  # ensure utils loaded to avoid circular import
from datacreek.models.llm_client import LLMClient, logger


class DummyResponse:
    def __init__(self):
        self.choices = [{"message": {"content": "ok"}}]


class DummyOpenAI:
    class Chat:
        class Completions:
            @staticmethod
            def create(**kwargs):
                return DummyResponse()

        completions = Completions()

    chat = Chat()


class DummyLLMClient:
    def __init__(self):
        self.provider = "api-endpoint"
        self.config = {"generation": {}}
        self.model = "gpt"
        self.max_retries = 1
        self.retry_delay = 0
        self.openai_client = DummyOpenAI()


def test_debug_mode_uses_logging(monkeypatch):
    dummy = DummyLLMClient()
    called = {}

    def fake_openai_chat_completion(self, *args, **kwargs):
        called["debug"] = logger.isEnabledFor(logging.DEBUG)
        return "ok"

    monkeypatch.setattr(
        LLMClient, "_openai_chat_completion", fake_openai_chat_completion
    )
    dummy._openai_chat_completion = fake_openai_chat_completion.__get__(
        dummy, DummyLLMClient
    )

    old_level = logger.level
    os.environ["SDK_DEBUG"] = "true"

    logger.setLevel(logging.INFO)
    LLMClient.chat_completion(dummy, [{"role": "user", "content": "hi"}])
    assert called["debug"] is False

    logger.setLevel(logging.DEBUG)
    LLMClient.chat_completion(dummy, [{"role": "user", "content": "hi"}])
    assert called["debug"] is True

    logger.setLevel(old_level)
