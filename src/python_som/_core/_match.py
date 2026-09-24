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

#: Bytes the winner search may hold in its score block at once, setting the chunk size. Tuned: at
#: 60x60 with 2000 samples an 8 MB budget is 2.6x slower and 8x heavier, because a block that fits
#: in cache is read back by ``argmin`` for free. See /explanation/how-batch-training-is-computed.
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

    For the Euclidean distance this expands the norm and drops the ``||x||^2`` term, which is
    constant across models, leaving a matrix product. Any other distance takes the loop, since only
    the Euclidean one has that identity.

    **Not the dot-product map of Kohonen Section 4.5**, which is a different algorithm requiring
    renormalized models. This is an exact re-expansion of the Euclidean distance.

    **The centring is not an optimization.** Without it the expansion cancels catastrophically:
    with models offset by 1e9, 499 of 500 samples get a different node. See
    :doc:`/explanation/how-batch-training-is-computed`.

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

    if not np.isfinite(flat).all():
        raise ValueError("weights must contain only finite values")
    shift = flat.mean(axis=0)
    centred = flat - shift
    squared = np.einsum("nf,nf->n", centred, centred)

    if kernel is not None and not np.isnan(data).any():  # pragma: no cover
        return kernel(data - shift, centred, squared)

    n_nodes = len(flat)
    chunk = max(1, _SCORE_BUDGET_BYTES // (n_nodes * 8))
    scores = np.empty((chunk, n_nodes))
    out = np.empty(len(data), dtype=np.intp)
    for start in range(0, len(data), chunk):
        block = data[start : start + chunk]
        if np.isinf(block).any():
            raise ValueError("data must contain only finite values or NaN")
        mask = ~np.isnan(block)
        if not mask.any(axis=1).all():
            raise ValueError("each sample must contain at least one finite value")
        centred_block = np.where(mask, block - shift, 0.0)
        block_scores = scores[: len(block)]
        np.matmul(centred_block, centred.T, out=block_scores)
        block_scores *= -2.0
        block_scores += np.matmul(mask, (centred**2).T)
        block_scores += np.einsum("nf,nf->n", centred_block, centred_block)[:, None]
        out[start : start + len(block)] = block_scores.argmin(axis=1)
    return out


def accumulate(
    data: npt.NDArray[Any],
    weights: npt.NDArray[Any],
    shape: tuple[int, int],
    distance: DistanceFunction,
    kernel: BmuKernel | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Sum observed sample features mapped to each node, and count them.

    These are the per-feature versions of ``n_j`` and ``n_j * xbar_j`` of Kohonen (2013), Eq. (8).
    Missing features contribute to neither the sum nor the count.

    :param data: Dataset of shape ``(n_samples, n_features)``.
    :param weights: Models, of shape ``(x, y, n_features)``.
    :param shape: Shape of the grid.
    :param distance: Dissimilarity measure.
    :param kernel: Optional accelerated search; see :func:`bmu_indices`.
    :return: Per-node sums and feature counts, both of shape ``(x, y, n_features)``.
    """
    nodes = bmu_indices(data, weights, distance, kernel)
    n_nodes = shape[0] * shape[1]
    n_features = weights.shape[-1]
    valid = np.isfinite(data)
    sums = np.zeros((n_nodes, n_features))
    counts = np.zeros((n_nodes, n_features))
    np.add.at(sums, nodes, np.where(valid, data, 0.0))
    np.add.at(counts, nodes, valid)
    return sums.reshape(*shape, n_features), counts.reshape(*shape, n_features)
