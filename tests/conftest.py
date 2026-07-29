"""Shared fixtures for the python_som test suite."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from hypothesis import settings

import python_som

# Deterministic replay in CI, so a failure reported by a run can be reproduced from the log.
settings.register_profile("ci", derandomize=True, max_examples=50, deadline=None)
settings.register_profile("dev", max_examples=25, deadline=None)
settings.load_profile("ci")

SEED = 20260728


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded generator, so every test that draws data is reproducible."""
    return np.random.default_rng(SEED)


@pytest.fixture
def blobs(rng: np.random.Generator) -> npt.NDArray[np.floating]:
    """Three well-separated Gaussian clusters in 3-D, 60 samples total."""
    centres = np.array([[0.0, 0.0, 0.0], [8.0, 8.0, 8.0], [-8.0, 8.0, -8.0]])
    return np.concatenate([c + rng.normal(scale=0.4, size=(20, 3)) for c in centres])


@pytest.fixture
def blob_labels() -> npt.NDArray[np.int_]:
    """Cluster identity for each sample of :func:`blobs`."""
    return np.repeat([0, 1, 2], 20)


@pytest.fixture
def frame(blobs: npt.NDArray[np.floating]) -> pd.DataFrame:
    """Return the blobs fixture as a DataFrame, for the input-type parity tests."""
    return pd.DataFrame(blobs, columns=["a", "b", "c"])


@pytest.fixture
def som() -> python_som.SOM:
    """Return a small seeded map with the default gaussian neighborhood."""
    return python_som.SOM(x=8, y=6, input_len=3, random_seed=SEED)


def make_som(**kwargs: Any) -> python_som.SOM:  # noqa: ANN401
    """Build a seeded map, overriding any constructor argument.

    :param kwargs: Constructor arguments to override.
    :return: The constructed map.
    """
    params: dict[str, Any] = {
        "x": 21,
        "y": 21,
        "input_len": 3,
        "random_seed": SEED,
    }
    params.update(kwargs)
    return python_som.SOM(**params)


def grid_radius(shape: tuple[int, int], c: tuple[int, int]) -> npt.NDArray[np.floating]:
    """Euclidean distance from ``c`` to every node of a non-cyclic grid.

    :param shape: Shape of the grid.
    :param c: Coordinates of the centre.
    :return: Distances, with the shape of the grid.
    """
    dx = np.arange(shape[0]) - c[0]
    dy = np.arange(shape[1]) - c[1]
    return np.sqrt(np.add.outer(dx**2, dy**2))
