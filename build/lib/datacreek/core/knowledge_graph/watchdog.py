"""Watchdog for cleanup configuration hot-reloading."""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:  # optional dependency
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except Exception:  # pragma: no cover - fallback when watchdog missing
    FileSystemEventHandler = object  # type: ignore[misc]

    class _DummyObserver:  # pragma: no cover - lightweight stub
        def schedule(self, *a, **k):
            logging.getLogger(__name__).debug("Watchdog unavailable; schedule noop")

        def start(self):
            logging.getLogger(__name__).debug("Watchdog unavailable; start noop")

        def stop(self):
            logging.getLogger(__name__).debug("Watchdog unavailable; stop noop")

        def join(self, timeout=None):
            logging.getLogger(__name__).debug(
                "Watchdog unavailable; join noop, timeout=%s", timeout
            )

    Observer = _DummyObserver  # type: ignore[assignment]

WATCHDOG_AVAILABLE = FileSystemEventHandler is not object
_watcher_lock = threading.Lock()

try:
    from ...utils.config import load_config
except ImportError:
    # Fallback for when running tests or in different context
    from datacreek.utils.config import load_config

# ---------------------------------------------------------------------------
# Cleanup config hot-reload support
# ---------------------------------------------------------------------------
_cleanup_cfg: Dict[str, float | int] = {}
_cleanup_lock = threading.Lock()
_observer: Observer | None = None
_cfg_path: Path | None = None


def _load_cleanup() -> None:  # pragma: no cover - heavy
    """Load cleanup parameters from YAML configuration."""
    cfg = load_config()
    cleanup = cfg.get("cleanup", {})
    data = {
        "tau": int(cleanup.get("tau", 5)),
        "sigma": float(cleanup.get("sigma", 0.95)),
        "k_min": int(cleanup.get("k_min", 5)),
        "lp_sigma": float(cleanup.get("lp_sigma", 0.3)),
        "lp_topk": int(cleanup.get("lp_topk", 50)),
        "hub_deg": int(cleanup.get("hub_deg", 1000)),
    }
    with _cleanup_lock:
        _cleanup_cfg.update(data)


def get_cleanup_cfg() -> Dict[str, float | int]:  # pragma: no cover - heavy
    """Return the currently loaded cleanup configuration."""
    with _cleanup_lock:
        return dict(_cleanup_cfg)


@dataclass
class CleanupConfig:
    """Current cleanup thresholds used across the application."""

    tau: int = 5
    sigma: float = 0.95
    k_min: int = 5
    lp_sigma: float = 0.3
    lp_topk: int = 50
    hub_deg: int = 1000


def apply_cleanup_config() -> None:  # pragma: no cover - heavy
    """Update :class:`CleanupConfig` with freshly loaded values."""

    vals = get_cleanup_cfg()
    CleanupConfig.tau = int(vals.get("tau", CleanupConfig.tau))
    CleanupConfig.sigma = float(vals.get("sigma", CleanupConfig.sigma))
    CleanupConfig.k_min = int(vals.get("k_min", CleanupConfig.k_min))
    CleanupConfig.lp_sigma = float(vals.get("lp_sigma", CleanupConfig.lp_sigma))
    CleanupConfig.lp_topk = int(vals.get("lp_topk", CleanupConfig.lp_topk))
    CleanupConfig.hub_deg = int(vals.get("hub_deg", CleanupConfig.hub_deg))
    logging.getLogger(__name__).info(
        "[CFG-HOT] cleanup thresholds updated at %s",
        datetime.now(timezone.utc).isoformat(),
    )


class ConfigReloader(FileSystemEventHandler):
    """Watch a YAML file and reload cleanup thresholds on changes."""

    def __init__(self, cfg_path: Path) -> None:  # pragma: no cover - heavy
        self.cfg_path = Path(cfg_path).resolve()

    def on_modified(
        self, event
    ) -> None:  # pragma: no cover - heavy type: ignore[override]
        if Path(event.src_path).resolve() == self.cfg_path:
            try:
                _load_cleanup()
                apply_cleanup_config()
            except Exception:
                logging.getLogger(__name__).exception("cleanup reload failed")


def verify_thresholds() -> None:  # pragma: no cover - heavy
    """Assert live cleanup thresholds match the YAML configuration.

    This loads the cleanup section from the current YAML config file and
    compares each value against :class:`CleanupConfig`. A mismatch indicates
    the hot-reload watcher failed to update live parameters.
    """

    cfg = load_config()
    expected = cfg.get("cleanup", {})
    for key in ["tau", "sigma", "k_min", "lp_sigma", "lp_topk", "hub_deg"]:
        yaml_val = expected.get(key)
        if yaml_val is None:
            continue
        curr_val = getattr(CleanupConfig, key)
        if curr_val != yaml_val:
            raise RuntimeError(f"CFG-HOT mismatch: {key}\u2260{yaml_val}")


def start_cleanup_watcher(  # pragma: no cover - heavy
    cfg_path: str | os.PathLike | None = None, interval: float = 300.0
) -> None:  # pragma: no cover - heavy
    """Start filesystem watcher reloading cleanup thresholds.

    Parameters
    ----------
    cfg_path:
        Optional path to the YAML configuration. When ``None`` the value of
        ``DATACREEK_CONFIG`` is used and defaults to ``configs/default.yaml``.
    interval:
        Polling interval for the :class:`watchdog.observers.Observer`.
    """

    global _observer, _cfg_path
    if not WATCHDOG_AVAILABLE:
        logger.info("Watchdog not installed; cleanup watcher disabled")
        return
    with _watcher_lock:
        if _observer is not None:
            logger.debug("cleanup watcher already running")
            return

    env_path = os.environ.get("DATACREEK_CONFIG", "configs/default.yaml")
    cfg_file = Path(cfg_path or env_path)
    _cfg_path = cfg_file.resolve()
    _load_cleanup()
    apply_cleanup_config()
    handler = ConfigReloader(_cfg_path)
    try:
        observer = Observer()
    except Exception as exc:
        logger.exception("Failed to instantiate watchdog observer: %s", exc)
        return
    observer.schedule(handler, str(_cfg_path.parent), recursive=False)
    observer.daemon = True
    observer.start()
    logger.info("CFG-HOT watcher started")
    _observer = observer


def stop_cleanup_watcher() -> None:  # pragma: no cover - heavy
    """Stop the cleanup configuration watcher."""

    global _observer
    with _watcher_lock:
        if _observer is None:
            return
        _observer.stop()
        _observer.join(timeout=0.5)
        _observer = None
