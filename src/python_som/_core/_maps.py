"""Summaries of a trained map: the U-matrix, and the three per-node mappings."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np

from ._match import winner
from ._neighborhood import bubble

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from ._match import DistanceFunction

__all__ = ["activation_matrix", "label_map", "u_matrix", "winner_map"]

#: The U-matrix compares each node with the ring immediately around it.
_ADJACENT = 1.0


def u_matrix(
    weights: npt.NDArray[Any],
    shape: tuple[int, int],
    cyclic: tuple[bool, bool],
    distance: DistanceFunction,
    *,
    normalize: bool = False,
) -> npt.NDArray[np.floating]:
    """Return the U-matrix: the summed distance from each model to its immediate neighbours.

    Ultsch's display (1993), cited by Kohonen (2013) Section 3.6 as the way cluster structure is
    made visible on the grid: a large value means neighbouring models are far apart, so it reads
    as a boundary.

    The adjacency is deliberately a flat ring of radius 1 rather than the configured neighborhood
    function. The U-matrix describes the grid, not the training schedule. The centre is included and
    contributes a distance of zero, so it does not affect the sum.

    Distances are computed and consumed one node at a time rather than accumulated into a full
    ``(x, y, x, y)`` tensor, which would cost ``(x*y)**2`` floats: about 800 MB on a 100x100 map, to
    produce ``x*y`` numbers.

    :param weights: Models, of shape ``(x, y, n_features)``.
    :param shape: Shape of the grid.
    :param cyclic: Whether each axis wraps around.
    :param distance: Dissimilarity measure.
    :param normalize: Whether to rescale the result to ``[0, 1]``.
    :return: U-matrix, with the shape of the grid.
    """
    um = np.zeros(shape)
    for index in np.ndindex(shape):
        node = (int(index[0]), int(index[1]))
        adjacency = bubble(shape, node, _ADJACENT, cyclic)
        um[node] = np.sum(adjacency * distance(weights[node], weights))
    if normalize:
        spread = np.max(um) - np.min(um)
        if spread > 0:
            um = (um - np.min(um)) / spread
    return um


def activation_matrix(
    data: npt.NDArray[Any],
    weights: npt.NDArray[Any],
    shape: tuple[int, int],
    distance: DistanceFunction,
) -> npt.NDArray[np.floating]:
    """Return how many samples map to each node.

    Kohonen (2013) Section 3.6 notes the SOM "is often used as a kind of histogram on which one
    displays the number of input data items that is mapped into each of the nodes".

    :param data: Dataset of shape ``(n_samples, n_features)``.
    :param weights: Models, of shape ``(x, y, n_features)``.
    :param shape: Shape of the grid.
    :param distance: Dissimilarity measure.
    :return: Counts, with the shape of the grid.
    """
    counts = np.zeros(shape)
    for sample in data:
        counts[winner(sample, weights, distance)] += 1
    return counts


def winner_map(
    data: npt.NDArray[Any],
    weights: npt.NDArray[Any],
    shape: tuple[int, int],
    distance: DistanceFunction,
) -> dict[tuple[int, int], list[npt.NDArray[Any]]]:
    """Return, for each node, the samples that map to it.

    Every node appears as a key, including the empty ones, so a caller can iterate the grid without
    checking for absence.

    :param data: Dataset of shape ``(n_samples, n_features)``.
    :param weights: Models, of shape ``(x, y, n_features)``.
    :param shape: Shape of the grid.
    :param distance: Dissimilarity measure.
    :return: Mapping from node coordinates to the samples assigned to that node.
    """
    result: dict[tuple[int, int], list[npt.NDArray[Any]]] = {
        (int(i), int(j)): [] for i, j in np.ndindex(shape)
    }
    for sample in data:
        result[winner(sample, weights, distance)].append(sample)
    return result


def label_map(
    data: npt.NDArray[Any],
    labels: npt.NDArray[Any],
    weights: npt.NDArray[Any],
    shape: tuple[int, int],
    distance: DistanceFunction,
) -> dict[tuple[int, int], Counter[Any]]:
    """Return, for each node, the frequency of each label mapped to it.

    This is the calibration step of Kohonen (2013) Section 3.2: "A particular model is labeled
    according to the majority of input samples that match with this model."

    :param data: Dataset of shape ``(n_samples, n_features)``.
    :param labels: One label per sample, in the same order as ``data``.
    :param weights: Models, of shape ``(x, y, n_features)``.
    :param shape: Shape of the grid.
    :param distance: Dissimilarity measure.
    :return: Mapping from node coordinates to a label counter.
    :raises ValueError: If ``data`` and ``labels`` have different lengths.
    """
    if len(data) != len(labels):
        msg = f"'data' and 'labels' must have the same length, got {len(data)} and {len(labels)}"
        raise ValueError(msg)
    counts: dict[tuple[int, int], Counter[Any]] = {
        (int(i), int(j)): Counter() for i, j in np.ndindex(shape)
    }
    for sample, label in zip(data, labels, strict=True):
        counts[winner(sample, weights, distance)].update([label])
    return counts
