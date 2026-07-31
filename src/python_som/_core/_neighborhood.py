"""Neighborhood functions: how the winner's correction spreads over the grid.

Kohonen (2013) Eq. (5) defines a neighborhood as a function of ``sqdist(c, i)``, the squared grid
distance between two nodes, so it must depend on that distance and not on the two axis offsets
separately. See :doc:`/explanation/why-isotropy-matters` for why an outer product of two 1-D
profiles is wrong for anything but the gaussian.

References:
Teuvo Kohonen, Essentials of the self-organizing map, Neural Networks 37 (2013) 52-65,
https://doi.org/10.1016/j.neunet.2012.09.018

O. J. Vrieze, Kohonen network, in: Artificial Neural Networks, Lecture Notes in Computer Science,
vol. 931, Springer, 1995, pp. 83-100, https://doi.org/10.1007/BFb0027024
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from ._protocols import AxisProfile, KernelFunction, NeighborhoodFunction

__all__ = [
    "AXIS_PROFILES",
    "NEIGHBORHOOD_FUNCTIONS",
    "SIGNED_NEIGHBORHOODS",
    "AxisProfile",
    "KernelFunction",
    "NeighborhoodFunction",
    "axis_matrix",
    "axis_offsets",
    "bubble",
    "bubble_axis_profile",
    "gaussian",
    "gaussian_axis_profile",
    "mexican_hat",
    "resolve",
    "resolve_axis_profile",
    "squared_grid_distance",
]

Grid = tuple[int, int]
Coordinates = tuple[int, int]


def _validate_radius(sigma: float, *, allow_zero: bool = False) -> None:
    """Reject a radius that would divide by zero or propagate NaN.

    ``not np.isfinite(sigma)`` is checked explicitly rather than relying on ``sigma <= 0``, which is
    ``False`` for NaN and would let it through.

    :param sigma: Radius to validate.
    :param allow_zero: Whether a radius of exactly zero is admissible.
    :raises ValueError: If the radius is not finite, is negative, or is zero when not allowed.
    """
    if not np.isfinite(sigma) or sigma < 0 or (sigma == 0 and not allow_zero):
        bound = "non-negative" if allow_zero else "positive"
        msg = f"The neighborhood radius 'sigma' must be a finite {bound} number, got {sigma!r}"
        raise ValueError(msg)


def axis_offsets(length: int, center: int, *, cyclic: bool) -> npt.NDArray[np.floating]:
    """Signed offsets from ``center`` to every coordinate along one axis.

    On a cyclic axis the minimum-image convention folds each offset into ``[-length/2, length/2)``.
    Both tails must be folded: an offset of -9 on an axis of length 10 is a distance of 1, not 9.

    :param length: Number of nodes along the axis.
    :param center: Coordinate of the winner along the axis.
    :param cyclic: Whether the axis wraps around.
    :return: Signed offsets, one per coordinate.
    """
    d = np.arange(length, dtype=float) - center
    if cyclic:
        d = (d + length / 2) % length - length / 2
    return d


def squared_grid_distance(
    shape: Grid, c: Coordinates, cyclic: tuple[bool, bool]
) -> npt.NDArray[np.floating]:
    """Squared geometric distance from ``c`` to every node, i.e. ``sqdist(c, i)`` of Eq. (5).

    :param shape: Shape of the network.
    :param c: Coordinates of the winner.
    :param cyclic: Whether each axis wraps around.
    :return: Squared distances, with the shape of the network.
    """
    dx = axis_offsets(shape[0], c[0], cyclic=cyclic[0])
    dy = axis_offsets(shape[1], c[1], cyclic=cyclic[1])
    return np.add.outer(np.square(dx), np.square(dy))


# One implementation of each formula, taking axis offsets rather than a grid and a centre.
# Validation lives here, so no caller can skip it.


def _gaussian_profile(
    dx: npt.NDArray[np.floating], dy: npt.NDArray[np.floating], sigma: float
) -> npt.NDArray[np.floating]:
    """Evaluate ``exp(-sqdist / (2 sigma^2))`` over an outer product of axis offsets.

    :param dx: Offsets along the first axis.
    :param dy: Offsets along the second axis.
    :param sigma: Neighborhood radius.
    :return: Neighborhood weights over the two axes.
    :raises ValueError: If the radius is not a finite positive number.
    """
    _validate_radius(sigma)
    return np.exp(-np.add.outer(np.square(dx), np.square(dy)) / (2.0 * sigma * sigma))


def _mexican_hat_profile(
    dx: npt.NDArray[np.floating], dy: npt.NDArray[np.floating], sigma: float
) -> npt.NDArray[np.floating]:
    """Evaluate ``(1 - u) exp(-u)`` over ``u = sqdist / (2 sigma^2)``.

    :param dx: Offsets along the first axis.
    :param dy: Offsets along the second axis.
    :param sigma: Neighborhood radius.
    :return: Neighborhood weights over the two axes.
    :raises ValueError: If the radius is not a finite positive number.
    """
    _validate_radius(sigma)
    u = np.add.outer(np.square(dx), np.square(dy)) / (2.0 * sigma * sigma)
    return (1.0 - u) * np.exp(-u)


def _bubble_profile(
    dx: npt.NDArray[np.floating], dy: npt.NDArray[np.floating], sigma: float
) -> npt.NDArray[np.floating]:
    """Evaluate the Chebyshev indicator ``max(|dx|, |dy|) <= round(sigma)``.

    Not built on ``sqdist``, unlike the other two: the bubble's metric is Chebyshev. See
    :func:`bubble`.

    :param dx: Offsets along the first axis.
    :param dy: Offsets along the second axis.
    :param sigma: Neighborhood radius, rounded to the nearest integer.
    :return: Neighborhood weights over the two axes.
    :raises ValueError: If the radius is not a finite non-negative number.
    """
    _validate_radius(sigma, allow_zero=True)
    radius = int(np.around(sigma))
    return np.multiply.outer(np.abs(dx) <= radius, np.abs(dy) <= radius).astype(float)


def gaussian(
    shape: Grid, c: Coordinates, sigma: float, cyclic: tuple[bool, bool]
) -> npt.NDArray[np.floating]:
    """Gaussian neighborhood, ``exp(-sqdist(c, i) / (2 * sigma**2))``.

    Eq. (5) of Kohonen (2013) with the learning rate factored out, so ``h(c, c) == 1``. Strictly
    positive and monotonically decreasing with distance.

    :param shape: Shape of the network.
    :param c: Coordinates of the winner.
    :param sigma: Neighborhood radius. Must be finite and positive.
    :param cyclic: Whether each axis wraps around.
    :return: Neighborhood weights, with the shape of the network.
    :raises ValueError: If the radius is not a finite positive number.
    """
    return _gaussian_profile(
        axis_offsets(shape[0], c[0], cyclic=cyclic[0]),
        axis_offsets(shape[1], c[1], cyclic=cyclic[1]),
        sigma,
    )


def mexican_hat(
    shape: Grid, c: Coordinates, sigma: float, cyclic: tuple[bool, bool]
) -> npt.NDArray[np.floating]:
    """Mexican hat neighborhood, ``(1 - u) * exp(-u)`` over ``u = sqdist(c, i) / (2 * sigma**2)``.

    The Ricker wavelet, or Laplacian of Gaussian: excitatory near the winner, inhibitory beyond it,
    vanishing with distance (Vrieze 1995, Fig. 3). Normalized so ``h(c, c) == 1``, zero at
    ``sqrt(2) * sigma``, minimum ``-exp(-2)`` at ``2 * sigma``.

    Not an outer product of two 1-D Ricker wavelets, which is a different and wrong function; see
    :doc:`/explanation/why-isotropy-matters`. Signed, so batch training rejects it.

    :param shape: Shape of the network.
    :param c: Coordinates of the winner.
    :param sigma: Neighborhood radius. Must be finite and positive.
    :param cyclic: Whether each axis wraps around.
    :return: Neighborhood weights, with the shape of the network.
    :raises ValueError: If the radius is not a finite positive number.
    """
    return _mexican_hat_profile(
        axis_offsets(shape[0], c[0], cyclic=cyclic[0]),
        axis_offsets(shape[1], c[1], cyclic=cyclic[1]),
        sigma,
    )


def bubble(
    shape: Grid, c: Coordinates, sigma: float, cyclic: tuple[bool, bool]
) -> npt.NDArray[np.floating]:
    """Flat neighborhood: 1 for nodes within ``sigma`` of the winner, 0 elsewhere.

    The truncated inner lobe of the mexican hat, which Vrieze (1995) p. 85 calls "just as effective
    and sometimes even better" than a distance-dependent one.

    **The metric is Chebyshev, not Euclidean**, so the region is a square: a node is included when
    ``max(|dx|, |dy|) <= round(sigma)``. This follows Vrieze's appendix, where Kohonen's "up to a
    certain radius" (Section 4.1) reads as Euclidean; the two sources differ, and
    :doc:`/explanation/why-isotropy-matters` covers the consequence, that a Chebyshev ball is not
    isotropic under the Euclidean metric.

    A radius of zero is admissible here and not for the other two: it selects the winner alone,
    where they would divide by zero.

    :param shape: Shape of the network.
    :param c: Coordinates of the winner.
    :param sigma: Neighborhood radius, rounded to the nearest integer. Must be finite and
        non-negative.
    :param cyclic: Whether each axis wraps around.
    :return: Neighborhood weights, with the shape of the network.
    :raises ValueError: If the radius is not a finite non-negative number.
    """
    return _bubble_profile(
        axis_offsets(shape[0], c[0], cyclic=cyclic[0]),
        axis_offsets(shape[1], c[1], cyclic=cyclic[1]),
        sigma,
    )


NEIGHBORHOOD_FUNCTIONS: Final[dict[str, NeighborhoodFunction]] = {
    "gaussian": gaussian,
    "bubble": bubble,
    "mexicanhat": mexican_hat,
    "mexican_hat": mexican_hat,
}
"""Neighborhood functions by name. ``mexican_hat`` is an alias of ``mexicanhat``."""


# Axis profiles: the per-axis factor of a separable neighborhood, used to contract Eq. (8) into two
# matrix products. A contraction strategy for a function of sqdist, never a redefinition of one.
# The mexican hat has no entry and must not acquire one: (1 - u) exp(-u) does not factor.
# See /explanation/how-batch-training-is-computed.


def gaussian_axis_profile(d: npt.NDArray[np.floating], sigma: float) -> npt.NDArray[np.floating]:
    """Per-axis factor of the gaussian, ``exp(-d^2 / (2 sigma^2))``.

    Its product over the two axes is :func:`gaussian`, because the exponential factors.

    :param d: Offsets along one axis.
    :param sigma: Neighborhood radius. Must be finite and positive.
    :return: Weights for those offsets.
    :raises ValueError: If the radius is not a finite positive number.
    """
    _validate_radius(sigma)
    return np.exp(-np.square(d) / (2.0 * sigma * sigma))


def bubble_axis_profile(d: npt.NDArray[np.floating], sigma: float) -> npt.NDArray[np.floating]:
    """Per-axis factor of the bubble, the indicator ``|d| <= round(sigma)``.

    Its product over the two axes is :func:`bubble`. It factors because the metric is Chebyshev; a
    Euclidean disc would not.

    :param d: Offsets along one axis.
    :param sigma: Neighborhood radius, rounded to the nearest integer. Must be finite and
        non-negative.
    :return: Weights for those offsets.
    :raises ValueError: If the radius is not a finite non-negative number.
    """
    _validate_radius(sigma, allow_zero=True)
    return (np.abs(d) <= int(np.around(sigma))).astype(float)


AXIS_PROFILES: Final[dict[str, AxisProfile]] = {
    "gaussian": gaussian_axis_profile,
    "bubble": bubble_axis_profile,
}
"""Per-axis factor of each separable neighborhood, keyed as :data:`NEIGHBORHOOD_FUNCTIONS`.

Membership decides what batch training accepts: a neighborhood absent from it is rejected by name
rather than approximated.
"""


def resolve_axis_profile(name: str) -> AxisProfile:
    """Look up the per-axis factor of a neighborhood function by name.

    :param name: Name of the neighborhood function.
    :return: The corresponding axis profile.
    :raises ValueError: If the function has no axis profile, and so is not separable.
    """
    try:
        return AXIS_PROFILES[name]
    except KeyError as exc:
        valid = sorted(AXIS_PROFILES)
        msg = (
            f"The {name!r} neighborhood function is not separable, so it has no axis profile. "
            f"Value should be one of {valid}"
        )
        raise ValueError(msg) from exc


def axis_matrix(
    length: int, sigma: float, *, cyclic: bool, profile: AxisProfile
) -> npt.NDArray[np.floating]:
    """Build ``H[a, c] = profile(a - c)`` for every pair of coordinates on one axis.

    Contracting ``sums`` against one of these per axis evaluates Eq. (8) for every node at once. The
    cyclic fold is the minimum-image convention of :func:`axis_offsets`, on pairwise offsets.

    :param length: Number of nodes along the axis.
    :param sigma: Neighborhood radius.
    :param cyclic: Whether the axis wraps around.
    :param profile: Per-axis factor to evaluate, from :data:`AXIS_PROFILES`.
    :return: Weights of shape ``(length, length)``.
    """
    d = np.subtract.outer(np.arange(length), np.arange(length)).astype(float)
    if cyclic:
        d = (d + length / 2) % length - length / 2
    return profile(d, sigma)


SIGNED_NEIGHBORHOODS: Final[frozenset[str]] = frozenset({"mexicanhat", "mexican_hat"})
"""Names of neighborhood functions that take negative values, which batch training cannot use."""


def resolve(name: str) -> NeighborhoodFunction:
    """Look up a neighborhood function by name.

    :param name: Name of the neighborhood function.
    :return: The corresponding function.
    :raises ValueError: If the name is not recognised.
    """
    try:
        return NEIGHBORHOOD_FUNCTIONS[name]
    except KeyError as exc:
        valid = sorted(NEIGHBORHOOD_FUNCTIONS)
        msg = (
            f"Invalid value for 'neighborhood_function' parameter: {name!r}. "
            f"Value should be one of {valid}"
        )
        raise ValueError(msg) from exc
