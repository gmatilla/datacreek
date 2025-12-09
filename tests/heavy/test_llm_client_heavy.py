import types

import pytest

import datacreek.analysis.generation as gen
import datacreek.models.llm_client as llm


class DummyResp:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code != 200:
            raise llm.requests.exceptions.RequestException("bad")


def _minimal_config():
    return {
        "llm": {"provider": "vllm"},
        "vllm": {
            "api_base": "http://x",
            "model": "m",
            "max_retries": 2,
            "retry_delay": 0,
        },
        "generation": {
            "temperature": 0.1,
            "max_tokens": 5,
            "top_p": 0.9,
            "batch_size": 2,
        },
    }


def _patch_config(monkeypatch):
    monkeypatch.setattr(
        llm, "load_config_with_overrides", lambda *a, **k: _minimal_config()
    )
    monkeypatch.setattr(llm, "get_llm_provider", lambda cfg: "vllm")
    monkeypatch.setattr(
        llm,
        "get_vllm_settings",
        lambda cfg: types.SimpleNamespace(
            api_base="http://x", model="m", max_retries=2, retry_delay=0
        ),
    )


@pytest.mark.heavy
def test_llm_check_and_chat(monkeypatch):
    _patch_config(monkeypatch)
    get_resps = iter(
        [
            DummyResp(200, {"ok": True}),
            DummyResp(200, {"ok": True}),
        ]
    )
    post_resps = iter(
        [
            DummyResp(500),
            DummyResp(200, {"choices": [{"message": {"content": "hi"}}]}),
        ]
    )
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: next(get_resps))
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: next(post_resps))
    monkeypatch.setattr(llm, "time", types.SimpleNamespace(sleep=lambda *_: None))
    called = {}
    monkeypatch.setattr(
        gen,
        "apply_logit_bias",
        lambda data, loc, glob: called.setdefault("bias", True),
    )
    client = llm.LLMClient()
    ok, info = client._check_vllm_server()
    assert ok and info == {"ok": True}
    out = client._vllm_chat_completion(
        [{"role": "user", "content": "hi"}],
        0.1,
        5,
        0.9,
        0,
        0,
        None,
        False,
        loc_hist=[1],
        glob_hist=[2],
        logits=[3],
    )
    assert out == "hi" and called.get("bias")


@pytest.mark.heavy
def test_llm_batch(monkeypatch):
    _patch_config(monkeypatch)
    monkeypatch.setattr(llm.LLMClient, "_check_vllm_server", lambda self: (True, {}))
    responses = iter(
        [
            DummyResp(200, {"choices": [{"message": {"content": "one"}}]}),
            DummyResp(200, {"choices": [{"message": {"content": "two"}}]}),
        ]
    )
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: next(responses))
    client = llm.LLMClient()
    result = client._vllm_batch_completion(
        [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
        0.1,
        5,
        0.9,
        1,
        0,
        0,
        None,
        False,
    )
    assert result == ["one", "two"]
