"""Analytic properties of the neighborhood functions.

These assert closed forms rather than golden values, so they stay meaningful if the implementation
is rewritten. The central property for the gaussian and the mexican hat is *isotropy*: they model
lateral interaction over a metric space, so they must depend on the distance between two nodes and
nothing else.

The bubble is deliberately excluded from the isotropy tests. It is a Chebyshev ball, not a
Euclidean one; see :func:`test_bubble_is_not_euclidean_isotropic`.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from python_som._core._neighborhood import (
    NEIGHBORHOOD_FUNCTIONS,
    NeighborhoodFunction,
    axis_offsets,
    bubble,
    gaussian,
    mexican_hat,
    resolve,
    squared_grid_distance,
)
from tests.conftest import grid_radius

ALL_FUNCTIONS = [gaussian, bubble, mexican_hat]
SMOOTH_FUNCTIONS = [gaussian, mexican_hat]
NO_WRAP = (False, False)
WRAP = (True, True)
GRID = (21, 21)
CENTRE = (10, 10)


# ---------------------------------------------------------------------------------------------
# Shared by every neighborhood function
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("func", ALL_FUNCTIONS)
def test_shape_matches_the_grid(func: NeighborhoodFunction) -> None:
    assert func(GRID, CENTRE, 3.0, NO_WRAP).shape == GRID


@pytest.mark.parametrize("func", ALL_FUNCTIONS)
def test_winner_has_maximum_response(func: NeighborhoodFunction) -> None:
    """The winner is maximally excited."""
    h = func(GRID, CENTRE, 3.0, NO_WRAP)
    assert h[CENTRE] == pytest.approx(h.max())


@pytest.mark.parametrize("func", SMOOTH_FUNCTIONS)
def test_normalized_to_unity_at_the_winner(func: NeighborhoodFunction) -> None:
    assert func(GRID, CENTRE, 3.0, NO_WRAP)[CENTRE] == pytest.approx(1.0)


@pytest.mark.parametrize("func", ALL_FUNCTIONS)
@pytest.mark.parametrize("sigma", [-1.0, -0.5, float("nan"), float("inf")])
def test_negative_or_non_finite_radius_is_rejected(
    func: NeighborhoodFunction, sigma: float
) -> None:
    """A negative or non-finite radius is meaningless for any neighborhood function."""
    with pytest.raises(ValueError, match="sigma"):
        func(GRID, CENTRE, sigma, NO_WRAP)


@pytest.mark.parametrize("func", SMOOTH_FUNCTIONS)
def test_zero_radius_is_rejected_where_it_divides(func: NeighborhoodFunction) -> None:
    """The radius appears in a denominator, so zero must raise rather than yield inf or nan."""
    with pytest.raises(ValueError, match="sigma"):
        func(GRID, CENTRE, 0.0, NO_WRAP)


@pytest.mark.parametrize("func", ALL_FUNCTIONS)
def test_four_fold_symmetry_about_a_centred_winner(func: NeighborhoodFunction) -> None:
    h = func(GRID, CENTRE, 3.0, NO_WRAP)
    np.testing.assert_allclose(h, h[::-1, :])
    np.testing.assert_allclose(h, h[:, ::-1])


@pytest.mark.parametrize("func", ALL_FUNCTIONS)
def test_output_is_finite(func: NeighborhoodFunction) -> None:
    assert np.isfinite(func(GRID, CENTRE, 3.0, NO_WRAP)).all()


# ---------------------------------------------------------------------------------------------
# Isotropy: the property a separable product violates
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("func", SMOOTH_FUNCTIONS)
def test_is_isotropic(func: NeighborhoodFunction) -> None:
    """The response depends only on the grid distance.

    Kohonen (2013) Eq. (5) defines the neighborhood over ``sqdist(c, i)`` and Vrieze (1995) Fig. 3
    plots it against a single "lateral distance" axis, so nodes equidistant from the winner must
    receive equal responses. An outer product of two 1-D profiles fails this for every profile
    except the gaussian, which separates only by a property of the exponential.
    """
    h = func(GRID, CENTRE, 3.0, NO_WRAP)
    r = np.round(grid_radius(GRID, CENTRE), 9)
    for radius in np.unique(r):
        shell = h[r == radius]
        np.testing.assert_allclose(shell, shell[0], atol=1e-12)


@given(
    size=st.integers(min_value=5, max_value=25),
    cx=st.integers(min_value=0, max_value=24),
    sigma=st.floats(min_value=0.3, max_value=8.0),
)
def test_isotropy_holds_for_generated_grids(size: int, cx: int, sigma: float) -> None:
    """Isotropy under adversarial grid sizes, centres and radii, not just hand-picked ones."""
    centre = (min(cx, size - 1), min(cx, size - 1))
    h = mexican_hat((size, size), centre, sigma, NO_WRAP)
    r = np.round(grid_radius((size, size), centre), 9)
    for radius in np.unique(r):
        shell = h[r == radius]
        np.testing.assert_allclose(shell, shell[0], atol=1e-12)


def test_mexican_hat_is_not_the_separable_product() -> None:
    """Guards against a regression to ``np.outer`` of two 1-D Ricker wavelets.

    The decisive difference is on the diagonal, where both 1-D factors are negative and their
    product turns positive: +0.165 where the isotropic form gives -0.055.
    """
    sigma = 3.0
    dx = np.arange(GRID[0]) - CENTRE[0]
    ax = (1 - dx**2 / sigma**2) * np.exp(-(dx**2) / (2 * sigma**2))
    separable = np.outer(ax, ax)
    h = mexican_hat(GRID, CENTRE, sigma, NO_WRAP)

    assert separable[16, 16] > 0 > h[16, 16]
    assert separable[16, 16] == pytest.approx(0.164840749998, rel=1e-9)
    assert h[16, 16] == pytest.approx(-0.054946916666, rel=1e-9)

    # and the separable product is measurably anisotropic
    r = np.round(grid_radius(GRID, CENTRE), 9)
    spreads = [np.ptp(separable[r == v]) for v in np.unique(r)]
    assert max(spreads) > 0.4


# ---------------------------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------------------------


def test_gaussian_matches_its_closed_form() -> None:
    sigma = 3.0
    expected = np.exp(-(grid_radius(GRID, CENTRE) ** 2) / (2 * sigma**2))
    np.testing.assert_allclose(gaussian(GRID, CENTRE, sigma, NO_WRAP), expected)


def test_gaussian_is_strictly_positive_and_decreasing() -> None:
    h = gaussian(GRID, CENTRE, 3.0, NO_WRAP)
    assert (h > 0).all()
    assert (np.diff(h[CENTRE[0], CENTRE[1] :]) < 0).all()


# ---------------------------------------------------------------------------------------------
# Mexican hat
# ---------------------------------------------------------------------------------------------


def test_mexican_hat_crosses_zero_at_sqrt2_sigma() -> None:
    """``(1 - u) exp(-u)`` with ``u = r^2 / 2 sigma^2`` vanishes at ``u = 1``."""
    shape, centre, sigma = (41, 41), (20, 20), 4.0
    h = mexican_hat(shape, centre, sigma, NO_WRAP)
    r = grid_radius(shape, centre)
    assert (h[r < np.sqrt(2) * sigma - 1e-9] > 0).all()
    beyond = (r > np.sqrt(2) * sigma + 1e-9) & (r < 4 * sigma)
    assert (h[beyond] < 0).all()


def test_mexican_hat_minimum_is_minus_exp_minus_two_at_two_sigma() -> None:
    """``d/du [(1 - u) e^-u] = 0`` at ``u = 2``, giving ``h = -e^-2`` at ``r = 2 sigma``."""
    shape, centre, sigma = (41, 41), (20, 20), 4.0
    h = mexican_hat(shape, centre, sigma, NO_WRAP)
    r = grid_radius(shape, centre)
    assert h.min() == pytest.approx(-np.exp(-2.0), rel=1e-9)
    assert r.flat[h.argmin()] == pytest.approx(2 * sigma, rel=1e-9)


def test_mexican_hat_has_no_positive_lobe_beyond_the_zero_crossing() -> None:
    """Excitation is confined to the inner disc."""
    shape, centre, sigma = (41, 41), (20, 20), 4.0
    h = mexican_hat(shape, centre, sigma, NO_WRAP)
    r = grid_radius(shape, centre)
    assert h[r > np.sqrt(2) * sigma + 1e-9].max() <= 0.0


# ---------------------------------------------------------------------------------------------
# Bubble
# ---------------------------------------------------------------------------------------------


def test_bubble_is_a_chebyshev_ball() -> None:
    """Vrieze (1995) p. 85, and the ``MAX(ABS(...), ABS(...))`` of his appendix pseudo-code."""
    h = bubble(GRID, CENTRE, 2.0, NO_WRAP)
    assert set(np.unique(h)) <= {0.0, 1.0}
    expected = np.zeros(GRID)
    expected[8:13, 8:13] = 1.0
    np.testing.assert_array_equal(h, expected)


def test_bubble_admits_a_zero_radius() -> None:
    """A zero radius selects the winner alone, which is well defined for an indicator."""
    h = bubble(GRID, CENTRE, 0.0, NO_WRAP)
    assert h[CENTRE] == 1.0
    assert h.sum() == 1.0


def test_bubble_is_not_euclidean_isotropic() -> None:
    """Pinning down a property that is easy to assume and is false.

    The bubble is a Chebyshev ball, so two nodes at equal *Euclidean* distance from the winner can
    fall on opposite sides of its boundary. The smallest such case is a radius of ``sqrt(50)``,
    where ``(5, 5)`` is inside a ``sigma = 5`` square while ``(7, 1)`` is outside.

    An earlier version of this suite asserted the bubble *was* isotropic and passed, because the
    grid it used was too small for any collision to exist: you need ``a^2 + b^2 = c^2 + d^2`` with
    ``max(a, b) <= sigma < max(c, d)``, and the smallest is ``50 = 25 + 25 = 49 + 1``.
    """
    shape, centre, sigma = (31, 31), (15, 15), 5.0
    h = bubble(shape, centre, sigma, NO_WRAP)
    assert h[15 + 5, 15 + 5] == 1.0  # Chebyshev distance 5, inside
    assert h[15 + 7, 15 + 1] == 0.0  # Chebyshev distance 7, outside
    # both are at Euclidean radius sqrt(50)
    assert np.hypot(5, 5) == pytest.approx(np.hypot(7, 1))


def test_bubble_rounds_its_radius() -> None:
    np.testing.assert_array_equal(
        bubble(GRID, CENTRE, 1.4, NO_WRAP), bubble(GRID, CENTRE, 1.0, NO_WRAP)
    )
    np.testing.assert_array_equal(
        bubble(GRID, CENTRE, 1.6, NO_WRAP), bubble(GRID, CENTRE, 2.0, NO_WRAP)
    )


# ---------------------------------------------------------------------------------------------
# Cyclic (toroidal) grids
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("func", ALL_FUNCTIONS)
def test_cyclic_wraps_from_both_edges(func: NeighborhoodFunction) -> None:
    """Regression for the one-sided fold-back.

    The original fold handled ``dx > shape/2`` and left offsets below ``-shape/2`` alone, so a
    winner at row 0 wrapped (``h[9] = 0.6065``) while a winner at row 9 did not (``h[0] = 0.0``).
    """
    shape = (10, 10)
    low = func(shape, (0, 5), 1.0, WRAP)[:, 5]
    high = func(shape, (9, 5), 1.0, WRAP)[:, 5]
    assert low[9] == pytest.approx(low[1])
    assert high[0] == pytest.approx(high[8])
    np.testing.assert_allclose(high, np.roll(low, 9))


@pytest.mark.parametrize("func", ALL_FUNCTIONS)
def test_cyclic_response_is_translation_invariant(func: NeighborhoodFunction) -> None:
    """Every node of a torus is equivalent, so the response depends only on the offset."""
    shape = (12, 12)
    reference = func(shape, (0, 0), 2.0, WRAP)
    for cx in range(shape[0]):
        for cy in range(shape[1]):
            np.testing.assert_allclose(
                func(shape, (cx, cy), 2.0, WRAP),
                np.roll(reference, (cx, cy), axis=(0, 1)),
            )


def test_non_cyclic_grid_does_not_wrap() -> None:
    assert gaussian((10, 10), (0, 5), 1.0, NO_WRAP)[9, 5] < 1e-12


@given(
    length=st.integers(min_value=2, max_value=40),
    center=st.integers(min_value=0, max_value=39),
)
def test_cyclic_offsets_never_exceed_half_the_axis(length: int, center: int) -> None:
    """The minimum-image convention bounds every offset by ``length / 2``."""
    d = axis_offsets(length, center % length, cyclic=True)
    assert np.abs(d).max() <= length / 2 + 1e-12


@given(
    length=st.integers(min_value=2, max_value=40),
    center=st.integers(min_value=0, max_value=39),
)
def test_cyclic_offset_matches_the_wrap_around_distance(length: int, center: int) -> None:
    c = center % length
    d = np.abs(axis_offsets(length, c, cyclic=True))
    direct = np.abs(np.arange(length) - c)
    expected = np.minimum(direct, length - direct)
    np.testing.assert_allclose(d, expected)


# ---------------------------------------------------------------------------------------------
# sqdist and lookup
# ---------------------------------------------------------------------------------------------


def test_squared_grid_distance_matches_euclidean() -> None:
    np.testing.assert_allclose(
        squared_grid_distance(GRID, CENTRE, NO_WRAP), grid_radius(GRID, CENTRE) ** 2
    )


def test_squared_grid_distance_is_zero_at_the_centre() -> None:
    assert squared_grid_distance(GRID, CENTRE, NO_WRAP)[CENTRE] == 0.0


@pytest.mark.parametrize("name", sorted(NEIGHBORHOOD_FUNCTIONS))
def test_every_registered_name_resolves(name: str) -> None:
    assert callable(resolve(name))


def test_mexican_hat_alias_is_the_same_function() -> None:
    assert resolve("mexican_hat") is resolve("mexicanhat")


def test_unknown_name_raises_value_error_listing_the_options() -> None:
    with pytest.raises(ValueError, match="neighborhood_function") as excinfo:
        resolve("sombrero")
    assert "gaussian" in str(excinfo.value)
