"""Matching inputs to models: which node wins, and how far away it is.

Everything here is used by both training and analysis, which is why it is its own module rather than
living beside either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._distance import euclidean_distance

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from ._protocols import BmuKernel, DistanceFunction

__all__ = ["accumulate", "activate", "bmu_indices", "quantization", "winner"]

#: Bytes the best-matching-unit search may hold in its score block at once. It sets the chunk size:
#: ``chunk = budget / (n_nodes * 8)``. Tuned rather than guessed, on a 60x60 map with 2000 samples:
#:
#: ======  ========  ========
#: budget  time      peak
#: ======  ========  ========
#: 512 KB  7.62 ms   1.07 MB
#: 2 MB    7.50 ms   2.57 MB
#: 8 MB    11.14 ms  8.56 MB
#: ======  ========  ========
#:
#: Larger is both slower and heavier, because a block that fits in cache is read back by ``argmin``
#: for free and one that does not is read back from memory.
_SCORE_BUDGET_BYTES = 512_000


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
    flat = weights.reshape(-1, weights.shape[-1])
    nodes = bmu_indices(data, weights, distance)
    # The distance is recomputed against the chosen model rather than read out of the search, which
    # keeps this exact for the Euclidean case: `bmu_indices` drops ||x||^2, so its scores order the
    # models correctly but are not distances.
    return np.array([distance(x, flat[node]) for x, node in zip(data, nodes, strict=True)])


def bmu_indices(
    data: npt.NDArray[Any],
    weights: npt.NDArray[Any],
    distance: DistanceFunction,
    kernel: BmuKernel | None = None,
) -> npt.NDArray[np.intp]:
    """Return the flat index of the best-matching model for every sample.

    This is Eq. (4) of Kohonen (2013), ``c = argmin_i ||x - m_i||``, for a whole dataset. Ties go to
    the first index in C order, which is ``argmin``'s behaviour and matches :func:`winner`.

    For the Euclidean distance this expands the norm, ``||x - w||^2 = ||x||^2 - 2 x.w + ||w||^2``,
    and drops ``||x||^2`` because it is constant across models and so cannot move the ``argmin``.
    What remains is a matrix product, which is 1.8x to 6.3x faster than one full-grid norm per
    sample. Any other distance takes the loop, since only the Euclidean one has this identity.

    **This is not the dot-product map of Kohonen Section 4.5.** That is a different algorithm,
    ``c = argmax_i dot(x, m_i)`` (Eq. 9), which requires the models to be renormalized to constant
    length after every cycle and selects a different node when they are not. This is an exact
    re-expansion of the Euclidean distance and needs no normalization.

    **The models are centred before the product, and that is not an optimization.** Without it the
    expansion is catastrophically cancelling: with models offset by 1e9, ``||w||^2`` is about 1e18
    while the differences between models are of order 1, and the subtraction loses every significant
    digit. Measured, 499 of 500 samples then get a different node. Subtracting a common shift is
    exact in ``||x - w||``, costs 1%, and removes it at every offset up to 1e12.

    :param data: Dataset of shape ``(n_samples, n_features)``.
    :param weights: Models, of shape ``(x, y, n_features)``.
    :param distance: Dissimilarity measure.
    :param kernel: Optional accelerated search, from ``python_som._accelerate``. Passed in rather
        than imported, so this module stays numpy-only.
    :return: One flat node index per sample.
    """
    flat = weights.reshape(-1, weights.shape[-1])
    if distance is not euclidean_distance:
        return np.array([np.asarray(distance(x, flat)).argmin() for x in data], dtype=np.intp)

    shift = flat.mean(axis=0)
    centred = flat - shift
    squared = np.einsum("nf,nf->n", centred, centred)

    if kernel is not None:  # pragma: no cover  reached only with the `fast` extra
        return kernel(data - shift, centred, squared)

    n_nodes = len(flat)
    chunk = max(1, _SCORE_BUDGET_BYTES // (n_nodes * 8))
    scores = np.empty((chunk, n_nodes))
    out = np.empty(len(data), dtype=np.intp)
    for start in range(0, len(data), chunk):
        block = data[start : start + chunk]
        # Into a preallocated buffer: allocating one per chunk was 1.5x slower and 8x heavier.
        np.matmul(block - shift, centred.T, out=scores[: len(block)])
        block_scores = scores[: len(block)]
        block_scores *= -2.0
        block_scores += squared
        out[start : start + len(block)] = block_scores.argmin(axis=1)
    return out


def accumulate(
    data: npt.NDArray[Any],
    weights: npt.NDArray[Any],
    shape: tuple[int, int],
    distance: DistanceFunction,
    kernel: BmuKernel | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Sum the samples mapped to each node, and count them.

    These are the ``n_j`` and ``n_j * xbar_j`` of Kohonen (2013), Eq. (8): the count of samples
    whose best match is node ``j``, and their sum.

    :param data: Dataset of shape ``(n_samples, n_features)``.
    :param weights: Models, of shape ``(x, y, n_features)``.
    :param shape: Shape of the grid.
    :param distance: Dissimilarity measure.
    :param kernel: Optional accelerated search; see :func:`bmu_indices`.
    :return: Per-node sums of shape ``(x, y, n_features)`` and counts of shape ``(x, y)``.
    """
    nodes = bmu_indices(data, weights, distance, kernel)
    n_nodes = shape[0] * shape[1]
    sums = np.zeros((n_nodes, weights.shape[-1]))
    np.add.at(sums, nodes, data)
    counts = np.bincount(nodes, minlength=n_nodes).astype(float)
    return sums.reshape(*shape, weights.shape[-1]), counts.reshape(shape)
