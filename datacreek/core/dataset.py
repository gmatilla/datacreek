"""DatasetBuilder wrapper selecting light or full implementation."""

import os

if os.getenv("DATACREEK_LIGHT_DATASET") == "1":  # pragma: no cover
    from .dataset_light import *  # pragma: no cover
else:  # pragma: no cover
    from .dataset_full import *  # pragma: no cover
