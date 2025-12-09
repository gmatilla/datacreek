import importlib
import os
from pathlib import Path

import pytest

import datacreek.utils.config as cfg


def test_load_config_relative_alt_path(tmp_path, monkeypatch):
    # change directory so relative path doesn't exist
    monkeypatch.chdir(tmp_path)
    data = cfg.load_config("configs/default.yaml")
    assert isinstance(data, dict)


def test_env_var_conversion_error(monkeypatch):
    config = {"vllm": {}}
    monkeypatch.setenv("LLM_MAX_RETRIES", "bad")
    vllm = cfg.get_vllm_settings(config)
    assert vllm.max_retries == "bad"


def test_generation_env_error(monkeypatch):
    monkeypatch.setenv("GEN_RETRIEVAL_TOP_K", "5")
    gen = cfg.get_generation_config({})
    assert gen.retrieval_top_k == 5


def test_get_prompt_and_missing():
    data = {"prompts": {"greet": "hi"}}
    assert cfg.get_prompt(data, "greet") == "hi"
    with pytest.raises(ValueError):
        cfg.get_prompt(data, "bye")


def test_merge_configs_nested():
    base = {"a": 1, "b": {"x": 2}}
    over = {"b": {"y": 3}}
    merged = cfg.merge_configs(base, over)
    assert merged == {"a": 1, "b": {"x": 2, "y": 3}}


def test_get_model_profile(tmp_path):
    cfg_data = {"models": {"m1": {"foo": 1}}}
    assert cfg.get_model_profile(cfg_data, "m1") == {"foo": 1}
    with pytest.raises(KeyError):
        cfg.get_model_profile(cfg_data, "m2")


def test_config_get(monkeypatch, tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("pid:\n  Kp: 0.5\n")
    monkeypatch.setenv(cfg.CONFIG_PATH_ENV, str(path))
    importlib.reload(cfg)
    data = cfg.Config.get()
    assert data["pid"]["Kp"] == 0.5
