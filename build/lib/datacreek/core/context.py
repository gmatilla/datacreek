"""Application context utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from datacreek.utils.config import CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH, load_config


class AppContext:
    """Context manager for global application state."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize application context.

        Parameters
        ----------
        config_path : Optional[Path]
            Explicit path to a configuration file. If not provided, the
            ``DATACREEK_CONFIG`` environment variable is checked before
            falling back to :data:`DEFAULT_CONFIG_PATH`.
        """
        env_path = os.environ.get(CONFIG_PATH_ENV)
        path = Path(config_path or env_path or DEFAULT_CONFIG_PATH)
        self.config_path = path
        self.config: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        """Load configuration from :attr:`config_path`."""
        self.config = load_config(str(self.config_path))
        return self.config
