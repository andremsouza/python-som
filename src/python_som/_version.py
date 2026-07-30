"""The package version, in one place.

Separate from ``__init__`` so that :mod:`python_som._som` can record it in a training report without
importing the package it is part of. ``tests/test_packaging.py`` asserts it matches
``pyproject.toml``, which is the other half of the single source of truth.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.5.0"
