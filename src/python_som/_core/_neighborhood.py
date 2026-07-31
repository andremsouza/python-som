"""Neighborhood functions: how the winner's correction spreads over the grid.

Kohonen (2013) Eq. (5) defines the neighborhood as a function of ``sqdist(c, i)``, "the square of
the geometric distance between the nodes c and i in the grid". Vrieze (1995) Fig. 3 plots the
"Mexican-hat" lateral interaction against a single axis labelled "Lateral distance", writes the
coefficient as ``h_{i i_c} = 1 / ||i_c - i||``, and states that the grid is assumed to be a metric
space.

The consequence is that a neighborhood function must depend on the distance between two nodes and
not on the two axis offsets separately. The gaussian happens to factor into a product of per-axis
terms, but that is a property of the exponential, not of neighborhood functions in general: an
outer product of two 1-D Ricker wavelets is positive in the diagonal quadrants where both factors
are negative, placing an excitatory lobe exactly where the mexican hat must inhibit.

References:
Teuvo Kohonen, Essentials of the self-organizing map, Neural Networks 37 (2013) 52-65,
https://doi.org/10.1016/j.neunet.2012.09.018

O. J. Vrieze, Kohonen network, in: Artificial Neural Networks: An Introduction to ANN Theory and
Practice, Lecture Notes in Computer Science, vol. 931, Springer, 1995, pp. 83-100,
https://doi.org/10.1007/BFb0027024
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

    On a cyclic axis the minimum-image convention folds each offset into ``[-length/2, length/2)``,
    so the shortest way around the torus is used. Both tails must be folded: an offset of -9 on an
    axis of length 10 represents a distance of 1, not 9.

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


# ---------------------------------------------------------------------------------------------
# The profiles: one implementation of each formula.
#
# Each takes the two axes' offsets rather than a grid and a centre, so the public function above can
# supply the offsets from one winner. Validation lives here, so no caller can skip it.
# ---------------------------------------------------------------------------------------------


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

    Deliberately not built on ``sqdist``, unlike the other two: the bubble's metric is Chebyshev, so
    it is a product of two per-axis indicators rather than a function of a Euclidean distance. That
    asymmetry is the implementation following Vrieze's appendix, and is preserved rather than
    quietly unified. See :func:`bubble`.

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

    This is Eq. (5) of Kohonen (2013) with the learning rate factored out, so ``h(c, c) == 1``.
    Strictly positive everywhere and monotonically decreasing with distance.

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

    Also known as the Ricker wavelet or the Laplacian of Gaussian. This is the biologically
    motivated lateral-interaction function: nodes near the winner are excited, nodes past a certain
    distance are inhibited, and the inhibition vanishes as distance grows further (Vrieze 1995,
    Fig. 3).

    Normalized so ``h(c, c) == 1``. Crosses zero at a radius of ``sqrt(2) * sigma`` and reaches its
    minimum of ``-exp(-2)``, about -0.135, at a radius of ``2 * sigma``.

    This is deliberately not the outer product of two 1-D Ricker wavelets. See the module docstring
    for why that construction is wrong.

    Takes negative values, so it cannot be used with batch training; see
    :data:`SIGNED_NEIGHBORHOODS`.

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

    This is the truncated inner, excitatory lobe of the mexican hat. Vrieze (1995) p. 85: "In
    Kohonen networks usually only the inner stimulation area is used, i.e., when a neuron i fires, a
    positive feedback takes place for all neurons i', whose distance to i is smaller than some given
    number rho", and notes that this flat choice is "just as effective and sometimes even better"
    than a distance-dependent one.

    **The metric here is Chebyshev, not Euclidean**, so the region is a square rather than a disc:
    a node is included when ``max(|dx|, |dy|) <= round(sigma)``. That matches the pseudo-code in
    Vrieze's appendix, which computes ``b = MAX(ABS(i - w_i), ABS(j - w_j))``, though Kohonen's
    phrase "up to a certain radius from the winner" (Section 4.1) reads as Euclidean. The two
    sources genuinely differ; this implementation follows Vrieze, and the choice is preserved rather
    than changed so that existing results stay reproducible.

    One consequence worth stating, because it is easy to assume otherwise: a Chebyshev ball is not
    isotropic under the Euclidean metric. On a large enough grid, nodes at equal Euclidean distance
    from the winner can fall on opposite sides of the boundary. The smallest case is a radius of
    ``sqrt(50)``, where ``(5, 5)`` lies inside a ``sigma = 5`` square while ``(7, 1)`` lies outside.

    Unlike the other neighborhood functions, a radius of zero is admissible: it selects the winner
    alone, which is well defined, whereas for the gaussian and the mexican hat it is a division by
    zero.

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


# ---------------------------------------------------------------------------------------------
# Axis profiles: the per-axis factor of a separable neighborhood.
#
# Eq. (8) sums h over every pair of nodes. Because h depends only on the offset between two nodes,
# that sum is a convolution, and a separable h turns it into two small matrix contractions instead
# of one pass per node. `AXIS_PROFILES` holds the factor for each neighborhood that has one.
#
# The isotropic definitions above remain the only definitions. A profile here is a contraction
# strategy for a function of sqdist, never a redefinition of it: the gaussian factors because the
# exponential does, and the bubble factors because its metric is Chebyshev. The mexican hat has no
# entry, and must not acquire one. (1 - u) exp(-u) does not factor, and an outer product of two 1-D
# Ricker wavelets is a different function, positive in the diagonal quadrants where the mexican hat
# must inhibit. See the module docstring.
# ---------------------------------------------------------------------------------------------


def gaussian_axis_profile(d: npt.NDArray[np.floating], sigma: float) -> npt.NDArray[np.floating]:
    """Per-axis factor of the gaussian, ``exp(-d^2 / (2 sigma^2))``.

    The product of this over the two axes is :func:`gaussian`, because
    ``exp(-(dx^2 + dy^2) / 2s^2) == exp(-dx^2 / 2s^2) * exp(-dy^2 / 2s^2)``.

    :param d: Offsets along one axis.
    :param sigma: Neighborhood radius. Must be finite and positive.
    :return: Weights for those offsets.
    :raises ValueError: If the radius is not a finite positive number.
    """
    _validate_radius(sigma)
    return np.exp(-np.square(d) / (2.0 * sigma * sigma))


def bubble_axis_profile(d: npt.NDArray[np.floating], sigma: float) -> npt.NDArray[np.floating]:
    """Per-axis factor of the bubble, the indicator ``|d| <= round(sigma)``.

    The product of this over the two axes is :func:`bubble`. It factors because the bubble's metric
    is Chebyshev, ``max(|dx|, |dy|) <= r``, which is the conjunction of two per-axis tests. A
    Euclidean disc, which Kohonen's "up to a certain radius" (Section 4.2) reads as, would not.

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
"""Per-axis factor of each separable neighborhood function, keyed as :data:`NEIGHBORHOOD_FUNCTIONS`.

Batch training resolves a neighborhood here, so this registry is what decides which functions batch
training can run. A neighborhood absent from it is rejected by name rather than approximated.
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
    matrix is ``length x length`` rather than the ``2 * length - 1`` a full-offset kernel needs.

    The cyclic fold is the same minimum-image convention as :func:`axis_offsets`, applied to the
    pairwise offsets.

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
