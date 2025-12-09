import builtins
import types

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
            "max_retries": 1,
            "retry_delay": 0,
        },
        "generation": {
            "temperature": 0.1,
            "max_tokens": 10,
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
            api_base="http://x", model="m", max_retries=1, retry_delay=0
        ),
    )


def test_check_vllm_server(monkeypatch):
    _patch_config(monkeypatch)
    client = llm.LLMClient
    monkeypatch.setattr(
        llm.requests, "get", lambda *a, **k: DummyResp(200, {"ok": True})
    )
    c = llm.LLMClient()
    assert c._check_vllm_server() == (True, {"ok": True})

    def raise_exc(*a, **k):
        raise llm.requests.exceptions.RequestException("fail")

    monkeypatch.setattr(llm.requests, "get", raise_exc)
    assert c._check_vllm_server()[0] is False


def test_vllm_chat_and_batch(monkeypatch):
    _patch_config(monkeypatch)
    monkeypatch.setattr(llm.LLMClient, "_check_vllm_server", lambda self: (True, {}))
    import datacreek.analysis.generation as gen

    called = {}
    monkeypatch.setattr(
        gen,
        "apply_logit_bias",
        lambda data, loc, glob: called.setdefault("bias", True),
    )
    responses = iter(
        [
            DummyResp(200, {"choices": [{"message": {"content": "first"}}]}),
            DummyResp(200, {"choices": [{"message": {"content": "second"}}]}),
            DummyResp(200, {"choices": [{"message": {"content": "third"}}]}),
        ]
    )

    def fake_post(url, headers=None, data=None, timeout=180):
        return next(responses)

    monkeypatch.setattr(llm.requests, "post", fake_post)
    client = llm.LLMClient()
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
    assert out == "first"
    batch = client._vllm_batch_completion(
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
    assert batch == ["second", "third"]
    assert called.get("bias")
