import os

import pytest

from datacreek.config_models import (
    CurateSettings,
    CurateSettingsModel,
    FormatSettings,
    FormatSettingsModel,
    GenerationSettings,
    GenerationSettingsModel,
    LLMSettings,
    LLMSettingsModel,
    OpenAISettings,
    OpenAISettingsModel,
    VLLMSettings,
    VLLMSettingsModel,
)
from datacreek.utils.config import (
    get_curate_settings,
    get_format_settings,
    get_generation_config,
    get_llm_settings,
    get_openai_settings,
    get_vllm_settings,
    load_config,
)


def test_env_override(monkeypatch):
    cfg = load_config()
    monkeypatch.setenv("GEN_TEMPERATURE", "0.5")
    gen_cfg = get_generation_config(cfg)
    assert gen_cfg.temperature == 0.5


def test_openai_env_override(monkeypatch):
    cfg = load_config()
    monkeypatch.setenv("LLM_API_BASE", "http://override")
    monkeypatch.setenv("API_ENDPOINT_KEY", "secret")
    oa = get_openai_settings(cfg)
    assert oa.api_base == "http://override"
    assert oa.api_key == "secret"


def test_vllm_env_override(monkeypatch):
    cfg = load_config()
    monkeypatch.setenv("LLM_API_BASE", "http://vllm")
    monkeypatch.setenv("LLM_MODEL", "mymodel")
    vllm = get_vllm_settings(cfg)
    assert vllm.api_base == "http://vllm"
    assert vllm.model == "mymodel"


def test_generation_settings_model_validation():
    model = GenerationSettingsModel(temperature=0.4)
    settings = model.to_settings()
    assert isinstance(settings, GenerationSettings)
    assert settings.temperature == 0.4
    with pytest.raises(Exception):
        GenerationSettingsModel(temperature="hot")


def test_curate_settings_model_validation():
    cfg = load_config()
    cur = get_curate_settings(cfg)
    assert isinstance(cur, CurateSettings)
    model = CurateSettingsModel(threshold=6.5)
    settings = model.to_settings()
    assert settings.threshold == 6.5


def test_format_settings_model_and_loader():
    cfg = load_config()
    fmt = get_format_settings(cfg)
    assert isinstance(fmt, FormatSettings)
    model = FormatSettingsModel(default="json")
    settings = model.to_settings()
    assert settings.default == "json"


def test_llm_and_provider_models():
    cfg = load_config()
    llm = get_llm_settings(cfg)
    assert isinstance(llm, LLMSettings)
    assert llm.provider in {"vllm", "api-endpoint"}

    vllm_settings = get_vllm_settings(cfg)
    assert isinstance(vllm_settings, VLLMSettings)

    oa_settings = get_openai_settings(cfg)
    assert isinstance(oa_settings, OpenAISettings)

    model = VLLMSettingsModel(api_base="http://example.com")
    assert model.to_settings().api_base == "http://example.com"

    oa_model = OpenAISettingsModel(model="gpt-4")
    assert oa_model.to_settings().model == "gpt-4"

    llm_model = LLMSettingsModel(provider="api-endpoint")
    assert llm_model.to_settings().provider == "api-endpoint"
