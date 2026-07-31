"""Shared fixtures and parameter sets for the asv suite.

Kept in one place so that every benchmark measures the same map on the same data, and so a change
to the case sizes moves all of them together rather than some of them.
"""

from __future__ import annotations

import numpy as np

import python_som

#: Fixed, so a difference between two commits is a difference in the code. asv reruns the same
#: benchmark against many revisions, and a fresh draw each time would drown the signal it looks for.
SEED = 20260730

#: Grid shapes covered by the parameterised benchmarks. Small enough that a full asv sweep over a
#: commit range finishes, large enough that per-node costs are visible.
SHAPES = [(10, 10), (30, 30), (60, 60)]

#: Feature counts. Varying this shifts how much of the work is the contraction rather than the
#: neighborhood, which is where this package's kernel does or does not pay off.
FEATURES = [4, 16]

#: Samples in the benchmark dataset.
SAMPLES = 300

#: Neighborhood radius. Above the 0.5 default floor, so the floor never engages.
RADIUS = 3.0


def data(n_features: int, n_samples: int = SAMPLES) -> np.ndarray:
    """Build the benchmark dataset.

    :param n_features: Number of features.
    :param n_samples: Number of samples.
    :return: The dataset.
    """
    return np.random.default_rng(SEED).normal(size=(n_samples, n_features))


def som(shape: tuple[int, int], n_features: int, neighborhood: str = "gaussian") -> python_som.SOM:
    """Build a map with random models, ready to train.

    Random rather than linear initialization: linear would fit a PCA on every setup call, charging
    every benchmark for work that ``initialization.py`` measures on its own.

    :param shape: Grid shape.
    :param n_features: Number of features.
    :param neighborhood: Neighborhood function name.
    :return: The map.
    """
    built = python_som.SOM(
        x=shape[0],
        y=shape[1],
        input_len=n_features,
        neighborhood_radius=RADIUS,
        neighborhood_function=neighborhood,
        random_seed=SEED,
    )
    built.weight_initialization(mode="random")
    return built
