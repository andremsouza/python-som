"""Weight initialization, above all the SVD that replaced scikit-learn's PCA in 0.4.0.

Dropping scikit-learn saved 264 MB of required install, and the twenty lines of ``np.linalg.svd``
that replaced it are also more accurate: the solver scikit-learn picked by default was wrong by 5.8%
on data far from the origin. Both claims are checked for correctness by
``tests/test_linalg_matches_sklearn.py``; this is where the cost side stays visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import python_som

from .common import FEATURES, SEED, SHAPES, data

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


class Initialize:
    """The three ways a map's models can be seeded."""

    params = (SHAPES, FEATURES)
    param_names = ("shape", "n_features")

    som: python_som.SOM
    data: np.ndarray

    def setup(self, shape: tuple[int, int], n_features: int) -> None:
        """Build an uninitialized map and the dataset outside the timed region.

        :param shape: Grid shape.
        :param n_features: Number of features.
        """
        self.som = python_som.SOM(x=shape[0], y=shape[1], input_len=n_features, random_seed=SEED)
        self.data = data(n_features)

    def time_linear(self, shape: tuple[int, int], n_features: int) -> None:
        """Time the PCA-based initialization, which is the SVD.

        :param shape: Unused.
        :param n_features: Unused.
        """
        del shape, n_features
        self.som.weight_initialization(mode="linear", data=self.data)

    def time_sample(self, shape: tuple[int, int], n_features: int) -> None:
        """Time seeding from drawn samples.

        :param shape: Unused.
        :param n_features: Unused.
        """
        del shape, n_features
        self.som.weight_initialization(mode="sample", data=self.data)

    def time_random(self, shape: tuple[int, int], n_features: int) -> None:
        """Time the cheapest initializer, as a floor to read the other two against.

        :param shape: Unused.
        :param n_features: Unused.
        """
        del shape, n_features
        self.som.weight_initialization(mode="random")
