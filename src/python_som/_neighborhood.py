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

from collections.abc import Callable
from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = [
    "NEIGHBORHOOD_FUNCTIONS",
    "SIGNED_NEIGHBORHOODS",
    "axis_offsets",
    "bubble",
    "gaussian",
    "mexican_hat",
    "resolve",
    "squared_grid_distance",
]

Grid = tuple[int, int]
Coordinates = tuple[int, int]
NeighborhoodFunction = Callable[
    [Grid, Coordinates, float, tuple[bool, bool]], npt.NDArray[np.floating]
]


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
    _validate_radius(sigma)
    return np.exp(-squared_grid_distance(shape, c, cyclic) / (2.0 * sigma * sigma))


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
    _validate_radius(sigma)
    u = squared_grid_distance(shape, c, cyclic) / (2.0 * sigma * sigma)
    return (1.0 - u) * np.exp(-u)


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
    _validate_radius(sigma, allow_zero=True)
    radius = int(np.around(sigma))
    dx = np.abs(axis_offsets(shape[0], c[0], cyclic=cyclic[0]))
    dy = np.abs(axis_offsets(shape[1], c[1], cyclic=cyclic[1]))
    return np.multiply.outer(dx <= radius, dy <= radius).astype(float)


NEIGHBORHOOD_FUNCTIONS: Final[dict[str, NeighborhoodFunction]] = {
    "gaussian": gaussian,
    "bubble": bubble,
    "mexicanhat": mexican_hat,
    "mexican_hat": mexican_hat,
}
"""Neighborhood functions by name. ``mexican_hat`` is an alias of ``mexicanhat``."""

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
