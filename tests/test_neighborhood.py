"""Analytic property tests for the neighborhood functions of the self-organizing map.

These assert closed-form properties rather than golden values, so they remain meaningful if the
implementation is rewritten. The central property is *isotropy*: a neighborhood function models
lateral interaction over the grid, so it must be a function of the distance between two nodes and
nothing else.

References:
Teuvo Kohonen, Essentials of the self-organizing map, Neural Networks 37 (2013) 52-65,
https://doi.org/10.1016/j.neunet.2012.09.018 -- Eq. (5), the neighborhood as a function of
sqdist(c, i).

O. J. Vrieze, Kohonen network, in: Artificial Neural Networks: An Introduction to ANN Theory and
Practice, LNCS 931, Springer, 1995, pp. 83-100, https://doi.org/10.1007/BFb0027024 -- Fig. 3, the
"Mexican-hat" function of lateral interaction plotted against a single "lateral distance" axis.
"""

import numpy as np
import pytest

import python_som


def make_som(
    x=21, y=21, cyclic_x=False, cyclic_y=False, neighborhood_function="gaussian"
):
    """Builds a SOM with a fixed seed; only the grid geometry matters for these tests."""
    return python_som.SOM(
        x=x,
        y=y,
        input_len=3,
        neighborhood_function=neighborhood_function,
        cyclic_x=cyclic_x,
        cyclic_y=cyclic_y,
        random_seed=0,
    )


def grid_radius(shape, c):
    """Euclidean distance from c to every node of a non-cyclic grid."""
    dx = np.arange(shape[0]) - c[0]
    dy = np.arange(shape[1]) - c[1]
    return np.sqrt(np.add.outer(dx**2, dy**2))


# --------------------------------------------------------------------------------------------
# Properties shared by every neighborhood function
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["gaussian", "bubble", "mexicanhat"])
def test_shape_matches_the_grid(name):
    som = make_som(neighborhood_function=name)
    assert som._neighborhood_function((10, 10), 3.0).shape == (21, 21)


@pytest.mark.parametrize("name", ["gaussian", "bubble", "mexicanhat"])
def test_winner_has_maximum_response(name):
    """The winner is maximally excited: h(c, c) is the global maximum."""
    som = make_som(neighborhood_function=name)
    h = som._neighborhood_function((10, 10), 3.0)
    assert h[10, 10] == pytest.approx(h.max())


@pytest.mark.parametrize("name", ["gaussian", "mexicanhat"])
def test_normalized_to_unity_at_the_winner(name):
    som = make_som(neighborhood_function=name)
    assert som._neighborhood_function((10, 10), 3.0)[10, 10] == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["gaussian", "bubble", "mexicanhat"])
@pytest.mark.parametrize("sigma", [-1.0, -0.5, float("nan"), float("inf")])
def test_negative_or_non_finite_sigma_is_rejected(name, sigma):
    """A negative or non-finite radius is meaningless for any neighborhood function."""
    som = make_som(neighborhood_function=name)
    with pytest.raises(ValueError, match="sigma"):
        som._neighborhood_function((10, 10), sigma)


@pytest.mark.parametrize("name", ["gaussian", "mexicanhat"])
def test_zero_sigma_is_rejected_where_it_divides(name):
    """sigma appears in a denominator, so zero must raise rather than yield inf/nan."""
    som = make_som(neighborhood_function=name)
    with pytest.raises(ValueError, match="sigma"):
        som._neighborhood_function((10, 10), 0.0)


def test_bubble_admits_a_zero_radius():
    """For the bubble a zero radius is well defined: it selects the winner alone."""
    som = make_som(neighborhood_function="bubble")
    h = som._bubble((10, 10), 0.0)
    assert h[10, 10] == 1
    assert h.sum() == 1


@pytest.mark.parametrize("name", ["gaussian", "bubble", "mexicanhat"])
def test_four_fold_symmetry_about_the_winner(name):
    """h is symmetric under reflection in both axes when the winner is centred."""
    som = make_som(neighborhood_function=name)
    h = som._neighborhood_function((10, 10), 3.0)
    np.testing.assert_allclose(h, h[::-1, :])
    np.testing.assert_allclose(h, h[:, ::-1])


@pytest.mark.parametrize("name", ["gaussian", "bubble", "mexicanhat"])
def test_is_isotropic(name):
    """h depends only on the grid distance -- the property a separable product violates.

    Kohonen (2013) Eq. (5) defines the neighborhood over sqdist(c, i); Vrieze (1995) Fig. 3 plots
    it against a single "lateral distance" axis. Nodes equidistant from the winner must therefore
    receive equal responses. An outer product of two 1-D profiles fails this for every profile
    except the gaussian, which is separable only by accident of the exponential.
    """
    som = make_som(neighborhood_function=name)
    h = som._neighborhood_function((10, 10), 3.0)
    r = np.round(grid_radius((21, 21), (10, 10)), 9)
    for radius in np.unique(r):
        responses = h[r == radius]
        np.testing.assert_allclose(responses, responses[0], atol=1e-12)


# --------------------------------------------------------------------------------------------
# Gaussian
# --------------------------------------------------------------------------------------------


def test_gaussian_matches_its_closed_form():
    som = make_som()
    sigma = 3.0
    h = som._gaussian((10, 10), sigma)
    expected = np.exp(-(grid_radius((21, 21), (10, 10)) ** 2) / (2 * sigma**2))
    np.testing.assert_allclose(h, expected)


def test_gaussian_is_strictly_positive_and_decreasing():
    som = make_som()
    h = som._gaussian((10, 10), 3.0)
    assert (h > 0).all()
    centre_row = h[10, 10:]
    assert (np.diff(centre_row) < 0).all()


# --------------------------------------------------------------------------------------------
# Mexican hat -- the properties that define the shape
# --------------------------------------------------------------------------------------------


def test_mexican_hat_crosses_zero_at_sqrt2_sigma():
    """(1 - u) exp(-u) with u = r^2/(2 sigma^2) vanishes at u = 1, i.e. r = sqrt(2) sigma."""
    som = make_som(x=41, y=41)
    sigma = 4.0
    h = som._mexicanhat((20, 20), sigma)
    r = grid_radius((41, 41), (20, 20))
    inside = r < np.sqrt(2) * sigma - 1e-9
    outside_but_near = (r > np.sqrt(2) * sigma + 1e-9) & (r < 4 * sigma)
    assert (h[inside] > 0).all()
    assert (h[outside_but_near] < 0).all()


def test_mexican_hat_minimum_is_minus_exp_minus_two_at_two_sigma():
    """d/du [(1-u) e^-u] = 0 at u = 2, giving h = -e^-2 at r = 2 sigma."""
    som = make_som(x=41, y=41)
    sigma = 4.0
    h = som._mexicanhat((20, 20), sigma)
    r = grid_radius((41, 41), (20, 20))
    assert h.min() == pytest.approx(-np.exp(-2.0), rel=1e-9)
    assert r.flat[h.argmin()] == pytest.approx(2 * sigma, rel=1e-9)


def test_mexican_hat_is_inhibitory_on_the_diagonal():
    """Regression for the separable outer product.

    An outer product of two 1-D Ricker wavelets is positive wherever both factors are negative,
    putting a spurious excitatory lobe on the diagonals. Measured on this exact grid, that
    construction gives +0.165 at (c+2s, c+2s) where the isotropic form gives -0.055.
    """
    som = make_som()
    sigma = 3.0
    h = som._mexicanhat((10, 10), sigma)
    assert h[16, 16] < 0
    assert h[16, 16] == pytest.approx(-0.054946916666, rel=1e-9)
    assert h[19, 19] < 0


def test_mexican_hat_has_no_positive_lobe_beyond_the_zero_crossing():
    """Excitation is confined to the inner disc; everything beyond it inhibits or vanishes."""
    som = make_som(x=41, y=41)
    sigma = 4.0
    h = som._mexicanhat((20, 20), sigma)
    r = grid_radius((41, 41), (20, 20))
    assert h[r > np.sqrt(2) * sigma + 1e-9].max() <= 0.0


def test_mexican_hat_differs_from_the_separable_product():
    """Guards against a regression to np.outer of two 1-D wavelets."""
    som = make_som()
    sigma = 3.0
    dx = np.arange(21) - 10
    ax = (1 - dx**2 / sigma**2) * np.exp(-(dx**2) / (2 * sigma**2))
    separable = np.outer(ax, ax)
    h = som._mexicanhat((10, 10), sigma)
    assert not np.allclose(h, separable)
    # The decisive difference: on the diagonal the separable product is excitatory where the
    # isotropic form inhibits, because there both of its 1-D factors are negative.
    assert separable[16, 16] > 0 > h[16, 16]
    assert separable[16, 16] == pytest.approx(0.164840749998, rel=1e-9)
    # The separable product is also anisotropic: (10, 14) and (13, 13) are not equidistant
    # in its value despite both lying at grid radius 4 and 3*sqrt(2) respectively.
    assert separable[10, 6] != pytest.approx(separable[13, 13])


# --------------------------------------------------------------------------------------------
# Bubble
# --------------------------------------------------------------------------------------------


def test_bubble_is_the_indicator_of_a_neighbourhood_set():
    """Vrieze (1995) p. 85: N_i = {i' : d(i, i') <= rho}, the truncated inner lobe."""
    som = make_som()
    h = som._bubble((10, 10), 2.0)
    assert set(np.unique(h)) <= {0, 1}
    expected = np.zeros((21, 21), dtype=int)
    expected[8:13, 8:13] = 1
    np.testing.assert_array_equal(h, expected)


# --------------------------------------------------------------------------------------------
# Cyclic (toroidal) grids
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["gaussian", "mexicanhat"])
def test_cyclic_wraps_from_both_edges(name):
    """Regression for the one-sided fold-back.

    The original fold only handled dx > shape/2, leaving offsets below -shape/2 unfolded. A winner
    at row 0 wrapped correctly (h[9] = 0.6065) while a winner at row 9 did not (h[0] = 0.0).
    """
    som = make_som(x=10, y=10, cyclic_x=True, cyclic_y=True, neighborhood_function=name)
    low = som._neighborhood_function((0, 5), 1.0)[:, 5]
    high = som._neighborhood_function((9, 5), 1.0)[:, 5]
    assert low[9] == pytest.approx(low[1])
    assert high[0] == pytest.approx(high[8])
    np.testing.assert_allclose(high, np.roll(low, 9))


@pytest.mark.parametrize("name", ["gaussian", "mexicanhat"])
def test_cyclic_response_is_translation_invariant(name):
    """On a torus every node is equivalent, so h depends only on the offset from the winner."""
    som = make_som(x=12, y=12, cyclic_x=True, cyclic_y=True, neighborhood_function=name)
    reference = som._neighborhood_function((0, 0), 2.0)
    for cx in range(12):
        for cy in range(12):
            shifted = som._neighborhood_function((cx, cy), 2.0)
            np.testing.assert_allclose(
                shifted, np.roll(reference, (cx, cy), axis=(0, 1))
            )


def test_non_cyclic_grid_does_not_wrap():
    som = make_som(x=10, y=10)
    h = som._gaussian((0, 5), 1.0)[:, 5]
    assert h[9] < 1e-12


# --------------------------------------------------------------------------------------------
# Selection and guards at the SOM level
# --------------------------------------------------------------------------------------------


def test_unknown_neighborhood_function_raises_value_error():
    with pytest.raises(ValueError, match="neighborhood_function"):
        python_som.SOM(x=8, y=8, input_len=3, neighborhood_function="mexican_hat")


def test_batch_training_rejects_the_mexican_hat():
    """Kohonen Eq. (8) divides by sum_j n_j h_ji, which is not sign-definite for a signed h."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(30, 3))
    som = make_som(x=12, y=12, neighborhood_function="mexicanhat")
    with pytest.raises(ValueError, match="batch"):
        som.train(data, n_iteration=1, mode="batch")


@pytest.mark.parametrize("mode", ["random", "sequential"])
def test_stepwise_training_accepts_the_mexican_hat(mode):
    rng = np.random.default_rng(0)
    som = make_som(x=12, y=12, neighborhood_function="mexicanhat")
    som.train(rng.normal(size=(30, 3)), n_iteration=30, mode=mode)
    assert np.isfinite(som.get_weights()).all()
