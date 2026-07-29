"""Distance measures between input vectors and the models of the network."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["euclidean_distance"]


def euclidean_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """Euclidean distance between the elements of the last axis of ``a`` and ``b``.

    Both arguments may be n-dimensional as long as their shapes broadcast. The result takes the
    shape of the broadcast minus its last axis, so passing one input vector and the whole
    ``(x, y, input_len)`` weight array yields an ``(x, y)`` map of distances.

    Kohonen (2013) Section 3.3 notes that the Euclidean distance, applied to normalized data, "is
    already applicable to most practical studies".

    :param a: Array-like of values. Must not be a scalar.
    :param b: Array-like of values. Must not be a scalar.
    :return: Distances between ``a`` and ``b``.
    """
    result: npt.NDArray[np.floating] = np.linalg.norm(np.subtract(a, b), ord=2, axis=-1)
    return result
