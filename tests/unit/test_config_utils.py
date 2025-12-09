import os

import pytest

from datacreek.utils import config


def create_config_file(tmpdir):
    path = os.path.join(tmpdir, "cfg.yaml")
    with open(path, "w") as f:
        f.write(
            """
llm:
  provider: api-endpoint
api-endpoint:
  api_base: http://api
  api_key: token
  model: gpt-4o
vllm:
  api_base: http://localhost:8001/v1
curate:
  batch_size: 10
format:
  include_metadata: false
models:
  small: {dim: 128}
databases:
  redis:
    host: 127.0.0.1
    port: 6380
  neo4j:
    uri: bolt://host:7687
    user: test
    password: test
"""
        )
    return path


def test_load_and_provider(monkeypatch, tmp_path):
    cfg_file = create_config_file(tmp_path)
    monkeypatch.setenv("DATACREEK_CONFIG", str(cfg_file))
    cfg = config.load_config()
    assert cfg["api-endpoint"]["api_base"] == "http://api"
    assert config.get_llm_provider(cfg) == "api-endpoint"


def test_vllm_and_openai_settings(monkeypatch, tmp_path):
    cfg_file = create_config_file(tmp_path)
    monkeypatch.setenv("DATACREEK_CONFIG", str(cfg_file))
    cfg = config.load_config()
    monkeypatch.setenv("LLM_MODEL", "new-model")
    vllm = config.get_vllm_settings(cfg)
    assert vllm.model == "new-model"
    monkeypatch.setenv("API_ENDPOINT_KEY", "k")
    openai = config.get_openai_settings(cfg)
    assert openai.api_key == "k"


def test_generation_and_curate(monkeypatch):
    monkeypatch.setenv("GEN_BATCH_SIZE", "12")
    gen_cfg = config.get_generation_config({})
    assert gen_cfg.batch_size == 12
    cur = config.get_curate_config({})
    cur_set = config.get_curate_settings({})
    assert cur["batch_size"] == 8
    assert cur_set.batch_size == 32


def test_format_and_model_profiles(tmp_path, monkeypatch):
    cfg_file = create_config_file(tmp_path)
    monkeypatch.setenv("DATACREEK_CONFIG", str(cfg_file))
    cfg = config.load_config()
    fs = config.get_format_settings(cfg)
    assert fs.include_metadata is False
    profile = config.get_model_profile(cfg, "small")
    assert profile == {"dim": 128}
    with pytest.raises(KeyError):
        config.get_model_profile(cfg, "missing")


def test_redis_and_neo4j_config(tmp_path, monkeypatch):
    cfg_file = create_config_file(tmp_path)
    monkeypatch.setenv("DATACREEK_CONFIG", str(cfg_file))
    cfg = config.load_config()
    redis_cfg = config.get_redis_config(cfg)
    neo_cfg = config.get_neo4j_config(cfg)
    assert redis_cfg["port"] == 6380
    assert neo_cfg["user"] == "test"


def test_merge_and_overrides(tmp_path, monkeypatch):
    cfg_file = create_config_file(tmp_path)
    monkeypatch.setenv("DATACREEK_CONFIG", str(cfg_file))
    base = config.load_config()
    override = {"curate": {"batch_size": 20}}
    merged = config.merge_configs(base, override)
    assert merged["curate"]["batch_size"] == 20
    loaded = config.load_config_with_overrides(cfg_file, override)
    assert loaded["curate"]["batch_size"] == 20


def test_get_and_reload(monkeypatch, tmp_path):
    cfg_file = create_config_file(tmp_path)
    monkeypatch.setattr(config, "_config_data", {"x": 1})
    assert config.Config.get() == {"x": 1}

    flag = {}

    def fake_load(path=None):
        flag["called"] = True
        return {"y": 2}

    monkeypatch.setattr(config, "load_config", fake_load)
    config.Config.reload()
    assert flag == {"called": True}
    assert config._config_data == {"y": 2}

    events = []
    monkeypatch.setattr(config.Config, "reload", lambda: events.append("r"))
    handler = config._ConfigHandler(cfg_file)
    event = type("Evt", (), {"src_path": str(cfg_file)})()
    handler.on_modified(event)
    assert events == ["r"]


def test_load_config_errors(tmp_path):
    path = tmp_path / "missing.yml"
    with pytest.raises(FileNotFoundError):
        config.load_config(str(path))


def test_manual_parse_and_prompt(monkeypatch, tmp_path):
    """Ensure manual YAML parser works when PyYAML and json fail."""
    cfg = tmp_path / "m.yml"
    cfg.write_text("""\nllm:\n  provider: vllm\nflag: true\nnum: 5\n""")
    monkeypatch.setattr(config, "yaml", None)
    import json as json_mod

    monkeypatch.setattr(
        json_mod, "load", lambda f: (_ for _ in ()).throw(ValueError()), raising=False
    )
    loaded = config.load_config(str(cfg))
    assert loaded["flag"] is True and loaded["num"] == 5
    with pytest.raises(ValueError):
        config.get_prompt(loaded, "missing")
    loaded.setdefault("prompts", {"p": "hi"})
    assert config.get_prompt(loaded, "p") == "hi"


def test_validation_error(monkeypatch, tmp_path):
    """Validation failures surface to the caller."""
    cfg = tmp_path / "b.yml"
    cfg.write_text("a: 1\n")
    monkeypatch.setattr(config, "yaml", None)

    class DummyError(Exception):
        pass

    monkeypatch.setattr(config, "ValidationError", DummyError)
    monkeypatch.setattr(
        config.ConfigSchema,
        "model_validate",
        lambda data: (_ for _ in ()).throw(DummyError()),
    )
    import json as json_mod

    monkeypatch.setattr(json_mod, "load", lambda f: {}, raising=False)
    with pytest.raises(DummyError):
        config.load_config(str(cfg))


def test_watcher_and_env_override(monkeypatch, tmp_path):
    """start and stop the config watcher and check env overrides."""
    cfg_file = create_config_file(tmp_path)
    monkeypatch.setenv("DATACREEK_CONFIG", str(cfg_file))
    # ensure no existing observer
    config._config_observer = None
    config.start_config_watcher(cfg_file)
    assert config._config_observer is not None
    config.stop_config_watcher()
    assert config._config_observer is None

    monkeypatch.setenv("GEN_TEST", "42")
    assert config._env_override("test") == "42"
    monkeypatch.delenv("GEN_TEST")
    assert config._env_override("test") is None
