"""Lightweight stub for scikit-learn to satisfy optional dependencies in tests.

This package exposes a minimal ``__spec__`` so that libraries performing
``importlib.util.find_spec('sklearn')`` succeed even when the full scikit-learn
stack is not available.
"""
