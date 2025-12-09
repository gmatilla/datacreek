# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Config Utilities
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency missing
    yaml = None

try:  # optional dependency for live config reloads
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except Exception:  # pragma: no cover - fallback when watchdog is absent
    FileSystemEventHandler = object  # type: ignore[misc]

    class _DummyObserver:  # pragma: no cover - lightweight stub
        def schedule(self, *a, **k):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def join(self, timeout=None):
            pass

    Observer = _DummyObserver  # type: ignore[assignment]

try:  # optional dependency
    from pydantic import ValidationError
except Exception:  # pragma: no cover - fallback when pydantic missing

    class ValidationError(Exception):
        """Fallback validation error used when pydantic is absent."""

        pass


from datacreek.config.schema import ConfigSchema
from datacreek.config_models import (
    CurateSettings,
    FormatSettings,
    GenerationSettings,
    LLMSettings,
    OpenAISettings,
    VLLMSettings,
)

# Default config location relative to the project root
ORIGINAL_CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "configs",
        "default.yaml",
    )
)

# Fallback path within the source tree for bundled deployments
SOURCE_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "default.yaml")
)

# Use the bundled path as default
DEFAULT_CONFIG_PATH = SOURCE_CONFIG_PATH
# Environment variable pointing to a config file location
CONFIG_PATH_ENV = "DATACREEK_CONFIG"

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration file.

    When ``config_path`` is not provided, the :data:`DATACREEK_CONFIG` environment
    variable is consulted before falling back to the built-in defaults.
    """
    if config_path is None:
        config_path = os.getenv(CONFIG_PATH_ENV)

    if config_path is None:
        # Try each path in order until one exists
        for path in [SOURCE_CONFIG_PATH, ORIGINAL_CONFIG_PATH]:
            if os.path.exists(path):
                config_path = path
                break
        else:
            # If none exists, use the default (which will likely fail, but with a clear error)
            config_path = DEFAULT_CONFIG_PATH

    if not os.path.exists(config_path):
        # Support relative paths when tests run from temporary directories
        if not os.path.isabs(config_path):
            pkg_root = Path(__file__).resolve().parents[2]
            alt_path = pkg_root / config_path
            if os.path.exists(alt_path):
                config_path = str(alt_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

    logger.info("Loading config from: %s", config_path)
    with open(config_path, "r") as f:
        if yaml is not None:
            config = yaml.safe_load(f)
        else:  # pragma: no cover - lightweight fallback
            import json

            try:
                config = json.load(f)
            except Exception:  # pragma: no cover - simple fallback parser
                f.seek(0)
                config = {}
                stack = [config]
                indents = [0]
                for line in f:
                    if not line.strip() or line.lstrip().startswith("#"):
                        continue
                    indent = len(line) - len(line.lstrip())
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.split("#", 1)[0].strip()
                    while len(indents) > 1 and indent <= indents[-1]:
                        stack.pop()
                        indents.pop()
                    parent = stack[-1]
                    if val == "":
                        node = {}
                        parent[key] = node
                        stack.append(node)
                        indents.append(indent)
                    else:
                        if val.lower() in {"true", "false"}:
                            parsed = val.lower() == "true"
                        else:
                            try:
                                parsed = int(val)
                            except ValueError:
                                try:
                                    parsed = float(val)
                                except ValueError:
                                    parsed = val.strip("\"'")
                        parent[key] = parsed

    # Validate against typed schema to catch malformed values early
    try:
        ConfigSchema.model_validate(config)
    except ValidationError:
        logger.exception("configuration validation failed")
        raise

    # Debug: Print LLM provider if it exists
    if "llm" in config and "provider" in config["llm"]:
        logger.info("Config has LLM provider set to: %s", config["llm"]["provider"])
    else:
        logger.info("Config does not have LLM provider set")

    return config


def get_llm_provider(config: Dict[str, Any]) -> str:
    """Get the selected LLM provider

    Returns:
        String with provider name: 'vllm' or 'api-endpoint'
    """
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "vllm")
    logger.debug("get_llm_provider returning: %s", provider)
    if (
        provider != "api-endpoint"
        and "llm" in config
        and "provider" in config["llm"]
        and config["llm"]["provider"] == "api-endpoint"
    ):
        logger.warning("Config has 'api-endpoint' but returning '%s'", provider)
    return provider


def get_llm_settings(config: Dict[str, Any]) -> LLMSettings:
    """Return general LLM configuration as :class:`LLMSettings`."""

    llm_cfg = config.get("llm", {})
    defaults = {"provider": "vllm"}
    defaults.update(llm_cfg)
    return LLMSettings.from_dict(defaults)


def get_vllm_settings(config: Dict[str, Any]) -> VLLMSettings:
    """Return VLLM configuration as :class:`VLLMSettings`."""

    defaults = config.get(
        "vllm",
        {
            "api_base": "http://localhost:8000/v1",
            "port": 8000,
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "max_retries": 3,
            "retry_delay": 1.0,
        },
    ).copy()
    for field_name in VLLMSettings.__dataclass_fields__:
        defaults.setdefault(field_name, getattr(VLLMSettings(), field_name))

    env_map = {
        "api_base": "LLM_API_BASE",
        "model": "LLM_MODEL",
        "max_retries": "LLM_MAX_RETRIES",
        "retry_delay": "LLM_RETRY_DELAY",
    }
    for field, env in env_map.items():
        if env_val := os.getenv(env):
            try:
                defaults[field] = type(defaults[field])(env_val)
            except Exception:
                defaults[field] = env_val

    return VLLMSettings.from_dict(defaults)


def get_vllm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Backwards compatible wrapper returning a plain dictionary."""

    return get_vllm_settings(config).__dict__


def get_openai_settings(config: Dict[str, Any]) -> OpenAISettings:
    """Return OpenAI/API endpoint configuration as :class:`OpenAISettings`."""

    defaults = config.get(
        "api-endpoint",
        {
            "api_base": None,
            "api_key": None,
            "model": "gpt-4o",
            "max_retries": 3,
            "retry_delay": 1.0,
        },
    ).copy()
    for field_name in OpenAISettings.__dataclass_fields__:
        defaults.setdefault(field_name, getattr(OpenAISettings(), field_name))

    env_map = {
        "api_base": "LLM_API_BASE",
        "api_key": "API_ENDPOINT_KEY",
        "model": "LLM_MODEL",
        "max_retries": "LLM_MAX_RETRIES",
        "retry_delay": "LLM_RETRY_DELAY",
    }
    for field, env in env_map.items():
        if env_val := os.getenv(env):
            try:
                defaults[field] = type(defaults[field])(env_val)
            except Exception:
                defaults[field] = env_val

    return OpenAISettings.from_dict(defaults)


def get_openai_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Backwards compatible wrapper returning a plain dictionary."""

    return get_openai_settings(config).__dict__


def _env_override(key: str) -> Optional[str]:  # pragma: no cover - simple helper
    """Helper to fetch environment variable overrides."""
    env_key = f"GEN_{key.upper()}"
    return os.environ.get(env_key)


def get_generation_config(config: Dict[str, Any]) -> GenerationSettings:
    """Return generation configuration as :class:`GenerationSettings`."""

    defaults = config.get("generation", {}).copy()

    # Fill in defaults from dataclass definition
    for field_name, field_def in GenerationSettings.__dataclass_fields__.items():
        defaults.setdefault(field_name, getattr(GenerationSettings(), field_name))

    # Apply environment variable overrides if present for all known keys
    env_overrides = {
        k: type(defaults.get(k))(v)  # type: ignore
        for k in GenerationSettings.__dataclass_fields__.keys()
        if (v := _env_override(k)) is not None
    }

    cfg = {**defaults, **env_overrides}
    return GenerationSettings.from_dict(cfg)


def get_curate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get curation configuration"""
    return config.get("curate", {"threshold": 7.0, "batch_size": 8, "temperature": 0.1})


def get_curate_settings(config: Dict[str, Any]) -> CurateSettings:
    """Return curation configuration as :class:`CurateSettings`."""
    defaults = config.get("curate", {}).copy()
    for field_name in CurateSettings.__dataclass_fields__:
        defaults.setdefault(field_name, getattr(CurateSettings(), field_name))
    return CurateSettings.from_dict(defaults)


def get_format_settings(config: Dict[str, Any]) -> FormatSettings:
    """Return output formatting configuration as :class:`FormatSettings`."""
    defaults = config.get("format", {}).copy()
    for field_name in FormatSettings.__dataclass_fields__:
        defaults.setdefault(field_name, getattr(FormatSettings(), field_name))
    return FormatSettings.from_dict(defaults)


def get_format_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get format configuration"""
    return config.get(
        "format", {"default": "jsonl", "include_metadata": True, "pretty_json": True}
    )


def get_prompt(config: Dict[str, Any], prompt_name: str) -> str:
    """Get prompt by name"""
    prompts = config.get("prompts", {})
    if prompt_name not in prompts:
        raise ValueError(f"Prompt '{prompt_name}' not found in configuration")
    return prompts[prompt_name]


def merge_configs(
    base_config: Dict[str, Any], override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two configuration dictionaries"""
    result = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_config_with_overrides(
    config_path: str | None = None, overrides: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Convenience wrapper around :func:`load_config` applying ``overrides``."""

    cfg = load_config(config_path)
    if overrides:
        cfg = merge_configs(cfg, overrides)
    return cfg


def get_model_profile(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Retrieve a model profile by name."""
    profiles = config.get("models", {})
    if name not in profiles:
        raise KeyError(f"Model profile '{name}' not found")
    return profiles[name]


def get_redis_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("databases", {}).get("redis", {"host": "localhost", "port": 6379})


def get_neo4j_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("databases", {}).get(
        "neo4j",
        {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "neo4j"},
    )


# ---------------------------------------------------------------------------
# Global configuration with hot-reload support
# ---------------------------------------------------------------------------

_config_data: Dict[str, Any] = load_config()  # pragma: no cover - load once at import
_config_lock = threading.RLock()
_config_observer: Observer | None = None


class Config:
    """Thread-safe access to the live configuration."""

    @classmethod
    def get(cls) -> Dict[str, Any]:
        """Return a copy of the current configuration."""

        with _config_lock:
            return dict(_config_data)

    @classmethod
    def reload(cls) -> None:
        """Reload YAML configuration into memory."""  # pragma: no cover

        global _config_data
        with _config_lock:
            _config_data = load_config()


class _ConfigHandler(FileSystemEventHandler):
    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()

    def on_modified(self, event) -> None:  # type: ignore[override] pragma: no cover
        if Path(event.src_path).resolve() == self.path:
            try:
                Config.reload()
            except Exception:
                logger.exception("config reload failed")


def start_config_watcher(
    cfg_path: str | os.PathLike | None = None,
) -> None:  # pragma: no cover
    """Start watchdog observer reloading the global configuration."""

    global _config_observer
    if _config_observer is not None:
        return

    path = Path(cfg_path or os.getenv(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH))
    handler = _ConfigHandler(path)
    observer = Observer()
    observer.schedule(handler, str(path.parent), recursive=False)
    observer.daemon = True
    observer.start()
    _config_observer = observer


def stop_config_watcher() -> None:  # pragma: no cover
    """Stop the configuration watcher if running."""

    global _config_observer
    if _config_observer is None:
        return
    _config_observer.stop()
    _config_observer.join(timeout=0.5)
    _config_observer = None
