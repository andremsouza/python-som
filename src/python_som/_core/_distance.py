"""Distance measures between input vectors and the models of the network."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover
    from ._protocols import DistanceFunction

__all__ = ["DISTANCE_FUNCTIONS", "euclidean_distance", "resolve_distance"]


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


DISTANCE_FUNCTIONS: Final[dict[str, DistanceFunction]] = {
    "euclidean_distance": euclidean_distance,
}
"""Distance functions by name, so a saved map can name the one it used.

One entry, because one is what the package ships. The registry exists for the same reason
:data:`DECAY_FUNCTIONS` does -- a name is what makes a map reloadable -- and a dictionary with one
key is the honest shape for that rather than a special case in the loader.
"""


def resolve_distance(name: str) -> DistanceFunction:
    """Look up a distance function by name.

    :param name: Name of the distance function.
    :return: The corresponding function.
    :raises ValueError: If the name is not recognised.
    """
    try:
        return DISTANCE_FUNCTIONS[name]
    except KeyError as exc:
        valid = sorted(DISTANCE_FUNCTIONS)
        msg = f"Unknown distance function {name!r}. Value should be one of {valid}"
        raise ValueError(msg) from exc
