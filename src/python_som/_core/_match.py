"""Matching inputs to models: which node wins, and how far away it is.

Everything here is used by both training and analysis, which is why it is its own module rather than
living beside either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from ._protocols import DistanceFunction

__all__ = ["accumulate", "activate", "quantization", "winner"]


def activate(
    x: npt.ArrayLike, weights: npt.NDArray[Any], distance: DistanceFunction
) -> npt.NDArray[np.floating]:
    """Return the distance from ``x`` to every model of the network.

    :param x: Input vector.
    :param weights: Models, of shape ``(x, y, n_features)``.
    :param distance: Dissimilarity measure.
    :return: Distances, with the shape of the grid.
    """
    return distance(x, weights)


def winner(
    x: npt.ArrayLike, weights: npt.NDArray[Any], distance: DistanceFunction
) -> tuple[int, int]:
    """Return the coordinates of the best-matching unit for ``x``.

    This is ``c = argmin_i ||x - m_i||`` of Kohonen (2013), Eq. (4). Ties go to the first index in
    C order, which is ``argmin``'s behaviour and is arbitrary but deterministic.

    :param x: Input vector.
    :param weights: Models, of shape ``(x, y, n_features)``.
    :param distance: Dissimilarity measure.
    :return: Coordinates of the winner.
    """
    activation = activate(x, weights, distance)
    index = np.unravel_index(activation.argmin(), activation.shape)
    return int(index[0]), int(index[1])


def quantization(
    data: npt.NDArray[Any], weights: npt.NDArray[Any], distance: DistanceFunction
) -> npt.NDArray[np.floating]:
    """Return the distance from each sample to its best-matching model.

    :param data: Dataset of shape ``(n_samples, n_features)``.
    :param weights: Models, of shape ``(x, y, n_features)``.
    :param distance: Dissimilarity measure.
    :return: One distance per sample.
    """
    return np.array([distance(i, weights[winner(i, weights, distance)]) for i in data])


def accumulate(
    data: npt.NDArray[Any],
    weights: npt.NDArray[Any],
    shape: tuple[int, int],
    distance: DistanceFunction,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Sum the samples mapped to each node, and count them.

    These are the ``n_j`` and ``n_j * xbar_j`` of Kohonen (2013), Eq. (8): the count of samples
    whose best match is node ``j``, and their sum.

    :param data: Dataset of shape ``(n_samples, n_features)``.
    :param weights: Models, of shape ``(x, y, n_features)``.
    :param shape: Shape of the grid.
    :param distance: Dissimilarity measure.
    :return: Per-node sums of shape ``(x, y, n_features)`` and counts of shape ``(x, y)``.
    """
    sums = np.zeros((*shape, weights.shape[-1]))
    counts = np.zeros(shape)
    for sample in data:
        node = winner(sample, weights, distance)
        sums[node] += sample
        counts[node] += 1
    return sums, counts
