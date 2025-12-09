"""Tests for the Feast feature repository and cache utilities."""

import importlib.machinery
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import sklearn  # Ensure sklearn is loaded before Feast imports

# Some minimal environments provide a placeholder ``sklearn`` without a module
# spec, which breaks libraries checking ``importlib.util.find_spec``.
if getattr(sklearn, "__spec__", None) is None:
    sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)

# Dynamically load the module to avoid name clashes with the external ``feast`` package.
_spec = importlib.util.spec_from_file_location(
    "feature_repo", Path(__file__).resolve().parents[2] / "feast" / "feature_repo.py"
)
feature_repo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feature_repo)  # type: ignore[arg-type]

# Re-export symbols for readability
cache_hit_rate = feature_repo.cache_hit_rate
embedding = feature_repo.embedding
get_vector_fp8 = feature_repo.get_vector_fp8
reset_cache = feature_repo.reset_cache
vector_fp8_view = feature_repo.vector_fp8_view


def test_entity_and_feature_view_definitions():
    """Ensure entity and feature view carry the expected names."""
    assert embedding.name == "embedding_hash"
    assert vector_fp8_view.name == "vector_fp8"


def test_get_vector_fp8_uses_cache():
    """A repeated lookup should be served from cache and increase hit rate."""
    store = MagicMock()
    # Simulate Feast online store returning a vector
    response = MagicMock()
    response.to_dict.return_value = {"vector_fp8": [[0.1, 0.2, 0.3]]}
    store.get_online_features.return_value = response

    reset_cache()
    first = get_vector_fp8(store, "hash")
    assert first == [0.1, 0.2, 0.3]
    assert cache_hit_rate() == 0.0
    store.get_online_features.assert_called_once()

    second = get_vector_fp8(store, "hash")
    assert second == first
    # One cache hit out of two lookups
    assert cache_hit_rate() == 0.5
    store.get_online_features.assert_called_once()
