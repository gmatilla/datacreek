import os
import sys
import types

# Ensure the repository root is importable so test modules can resolve local
# packages such as ``metrics_prometheus`` and ``scripts`` without relying on
# ``PYTHONPATH`` environment tweaks.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("DATACREEK_REQUIRE_PERSISTENCE", "0")
os.environ.setdefault("DATACREEK_LIGHT_DATASET", "1")


import pytest

# Lightweight stubs for scikit-learn to avoid heavy dependency.
if "sklearn" not in sys.modules:
    sklearn = types.ModuleType("sklearn")
    cd = types.ModuleType("sklearn.cross_decomposition")

    class DummyCCA:
        def fit(self, X, Y):
            return self

        def transform(self, X, Y=None):
            return X

    cd.CCA = DummyCCA
    decomp = types.ModuleType("sklearn.decomposition")

    class DummyPCA:
        def __init__(self, n_components=None):
            self.n_components = n_components

        def fit_transform(self, X):
            return X

        def inverse_transform(self, X):
            return X

    decomp.PCA = DummyPCA
    sklearn.cross_decomposition = cd
    sklearn.decomposition = decomp
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.cross_decomposition"] = cd
    sys.modules["sklearn.decomposition"] = decomp


@pytest.fixture(autouse=True)
def _no_neo4j(monkeypatch):
    """Disable Neo4j access during tests."""
    try:
        monkeypatch.setattr("datacreek.api.get_neo4j_driver", lambda: None)
    except Exception:
        # API may require optional deps like fastapi
        pass
    try:
        monkeypatch.setattr("datacreek.core.dataset.InvariantPolicy.loops", 0)
    except Exception:
        # dataset module may be missing heavy deps
        pass
    yield


import importlib.util

HEAVY_DEPS = [
    "numpy",
    "networkx",
    "torch",
    "fakeredis",
    "fastapi",
    "sqlalchemy",
    "prometheus_client",
    "yaml",
    "pydantic",
    "hypothesis",
    "opentelemetry",
]


def pytest_configure(config):
    markexpr = getattr(config.option, "markexpr", "")
    if "heavy" in markexpr:
        missing = [d for d in HEAVY_DEPS if importlib.util.find_spec(d) is None]
        if missing:
            pytest.exit(
                "Skipping heavy tests due to missing dependencies: "
                + ", ".join(missing),
                returncode=0,
            )
