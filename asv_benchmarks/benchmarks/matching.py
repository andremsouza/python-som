"""Matching inputs to models: the other half of batch training, and all of scoring.

``accumulate`` is roughly 40% of a batch iteration and loops over samples in Python, so it is the
half of the cost that scales with the dataset rather than with the grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from python_som._core._distance import euclidean_distance
from python_som._core._match import accumulate, quantization, winner

from .common import FEATURES, SHAPES, data, som

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


class Match:
    """Winner search, accumulation and quantization on the same map and dataset."""

    params = (SHAPES, FEATURES)
    param_names = ("shape", "n_features")

    weights: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]

    def setup(self, shape: tuple[int, int], n_features: int) -> None:
        """Build the models and dataset outside the timed region.

        :param shape: Grid shape.
        :param n_features: Number of features.
        """
        self.shape = shape
        self.weights = som(shape, n_features).get_weights()
        self.data = data(n_features)

    def time_accumulate(self, shape: tuple[int, int], n_features: int) -> None:
        """Time one batch iteration's worth of per-node sums and counts.

        :param shape: Unused.
        :param n_features: Unused.
        """
        del shape, n_features
        accumulate(self.data, self.weights, self.shape, euclidean_distance)

    def time_winner_for_every_sample(self, shape: tuple[int, int], n_features: int) -> None:
        """Time the best-matching-unit search alone, without the accumulation around it.

        :param shape: Unused.
        :param n_features: Unused.
        """
        del shape, n_features
        for sample in self.data:
            winner(sample, self.weights, euclidean_distance)

    def time_quantization(self, shape: tuple[int, int], n_features: int) -> None:
        """Time what ``quantization_error`` costs, which every ``train`` call pays once.

        :param shape: Unused.
        :param n_features: Unused.
        """
        del shape, n_features
        quantization(self.data, self.weights, euclidean_distance)
