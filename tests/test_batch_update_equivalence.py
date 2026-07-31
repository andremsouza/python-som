"""Batch training contracts the neighborhood by axis. It must equal evaluating it per node.

Eq. (8) sums ``h`` over every pair of nodes. Because ``h`` depends only on the offset between two
nodes, that sum is a convolution, and a separable ``h`` turns it into two matrix contractions with
no loop over nodes. The saving is large, so the equality has to be held down hard rather than
assumed.

Separability is an identity for exactly the two neighborhoods batch training admits, and for
neither of the reasons is it general:

- the gaussian, because ``exp(-(dx^2 + dy^2) / 2s^2) == exp(-dx^2 / 2s^2) * exp(-dy^2 / 2s^2)``;
- the bubble, because ``max(|dx|, |dy|) <= r`` is the conjunction of two per-axis tests.

The mexican hat factors under neither, which is why it has no axis profile and why batch training
rejects it. An outer product of two 1-D Ricker wavelets is a different function, positive in the
diagonal quadrants where the mexican hat must inhibit; that was a real defect in this package once,
and these tests are what stop the contraction quietly reintroducing it.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import python_som
from python_som import Neighborhood
from python_som._core._match import accumulate
from python_som._core._neighborhood import (
    AXIS_PROFILES,
    NEIGHBORHOOD_FUNCTIONS,
    SIGNED_NEIGHBORHOODS,
    axis_matrix,
    bubble,
    gaussian,
    resolve_axis_profile,
)
from python_som._core._update import batch_update

#: Round-off scale for the contraction against the per-node reference. Measured at 3.1e-15 relative
#: on a 60x60 map; the two sum the same terms in a different order, so exact equality is not
#: available and asserting it would be asserting the wrong thing.
TOLERANCE = 1e-12

#: Fixed so a failure is reproducible.
SEED = 20260730

#: Shapes, including the degenerate single-row and single-column maps where an axis has length 1.
SHAPES = [(1, 6), (6, 1), (5, 5), (7, 4), (12, 9), (20, 16)]

#: Radii, including one below 1 and one larger than the grid.
RADII = [0.5, 1.0, 2.5, 4.0, 30.0]

#: Neighborhoods batch training admits. Derived rather than listed, so a new one joins the sweep.
SEPARABLE = sorted(AXIS_PROFILES)


def _per_node_reference(
    weights: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
    shape: tuple[int, int],
    name: str,
    sigma: float,
    cyclic: tuple[bool, bool],
) -> np.ndarray:
    """Evaluate Eq. (8) node by node from the isotropic definition.

    This is the definition the contraction has to match: it calls the public neighborhood function,
    which is a function of ``sqdist``, once per node.

    :param weights: Current models.
    :param sums: Per-node sums.
    :param counts: Per-node counts.
    :param shape: Grid shape.
    :param name: Neighborhood function name.
    :param sigma: Neighborhood radius.
    :param cyclic: Whether each axis wraps.
    :return: The updated models.
    """
    evaluate = NEIGHBORHOOD_FUNCTIONS[name]
    updated = weights.copy()
    for node in np.ndindex(shape):
        node_2d = (int(node[0]), int(node[1]))
        h = evaluate(shape, node_2d, sigma, cyclic)
        denominator = float(np.sum(h * counts))
        if denominator > 0:
            updated[node_2d] = np.einsum("xy,xyf->f", h, sums) / denominator
    return updated


def _case(shape: tuple[int, int], n_features: int = 3) -> tuple[np.ndarray, ...]:
    """Build models, per-node sums and per-node counts for one grid.

    Counts are drawn with zeros in them on purpose: a node with no data in reach is the case that
    must keep its previous value, and it is the one a naive implementation destroys.

    :param shape: Grid shape.
    :param n_features: Number of features.
    :return: Weights, sums and counts.
    """
    rng = np.random.default_rng(SEED + shape[0] * 100 + shape[1])
    return (
        rng.normal(size=(*shape, n_features)),
        rng.normal(size=(*shape, n_features)),
        rng.integers(0, 3, size=shape).astype(float),
    )


# ---------------------------------------------------------------------------------------------
# The contraction equals the definition
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("name", SEPARABLE)
@pytest.mark.parametrize("cyclic", list(itertools.product([False, True], repeat=2)))
@pytest.mark.parametrize("sigma", RADII)
def test_the_contraction_equals_the_per_node_definition(
    shape: tuple[int, int], name: str, cyclic: tuple[bool, bool], sigma: float
) -> None:
    """Every shape, every neighborhood, every cyclic combination, every radius."""
    weights, sums, counts = _case(shape)
    profile = resolve_axis_profile(name)
    hx = axis_matrix(shape[0], sigma, cyclic=cyclic[0], profile=profile)
    hy = axis_matrix(shape[1], sigma, cyclic=cyclic[1], profile=profile)

    contracted = batch_update(weights, sums, counts, hx, hy)
    reference = _per_node_reference(weights, sums, counts, shape, name, sigma, cyclic)

    scale = max(float(np.abs(reference).max()), 1.0)
    assert float(np.abs(contracted - reference).max()) / scale < TOLERANCE


@pytest.mark.parametrize("name", SEPARABLE)
@pytest.mark.parametrize("cyclic", list(itertools.product([False, True], repeat=2)))
def test_the_axis_factors_multiply_to_the_isotropic_neighborhood(
    name: str, cyclic: tuple[bool, bool]
) -> None:
    """The claim separability rests on, asserted directly rather than only through Eq. (8).

    For each node, the outer product of the two axis factors must be that node's neighborhood as the
    public function computes it from ``sqdist``.
    """
    shape, sigma = (9, 7), 2.0
    profile = resolve_axis_profile(name)
    hx = axis_matrix(shape[0], sigma, cyclic=cyclic[0], profile=profile)
    hy = axis_matrix(shape[1], sigma, cyclic=cyclic[1], profile=profile)
    evaluate = NEIGHBORHOOD_FUNCTIONS[name]

    for node in np.ndindex(shape):
        node_2d = (int(node[0]), int(node[1]))
        factored = np.multiply.outer(hx[:, node_2d[0]], hy[:, node_2d[1]])
        np.testing.assert_allclose(factored, evaluate(shape, node_2d, sigma, cyclic), atol=1e-15)


def test_a_node_with_no_data_in_reach_keeps_its_previous_value() -> None:
    """Kohonen Eq. (8) is undefined where the denominator is zero, so the old model stands.

    Regression for a defect that wiped 282 of 900 models in a single step on a 30x30 map by building
    the result from a zeroed array. The bubble makes it reachable: it is exactly zero outside its
    radius, where the gaussian is merely small.
    """
    shape, sigma = (30, 30), 1.0
    weights, sums, counts = _case(shape)
    counts[:] = 0.0
    counts[0, 0] = 5.0

    profile = resolve_axis_profile("bubble")
    hx = axis_matrix(shape[0], sigma, cyclic=False, profile=profile)
    hy = axis_matrix(shape[1], sigma, cyclic=False, profile=profile)
    updated = batch_update(weights, sums, counts, hx, hy)

    reached = np.zeros(shape, dtype=bool)
    reached[:2, :2] = True
    np.testing.assert_array_equal(updated[~reached], weights[~reached])
    assert not np.array_equal(updated[0, 0], weights[0, 0]), "the node with data must have moved"


def test_the_update_is_concurrent_over_every_node() -> None:
    """Kohonen Section 4.4: models are replaced "in one concurrent computing operation".

    Every node must be computed from the models as they stood at the start of the iteration. A loop
    writing into the array it reads would satisfy the other tests here and fail this one.
    """
    shape, sigma = (8, 6), 2.0
    weights, sums, counts = _case(shape)
    profile = resolve_axis_profile("gaussian")
    hx = axis_matrix(shape[0], sigma, cyclic=False, profile=profile)
    hy = axis_matrix(shape[1], sigma, cyclic=False, profile=profile)

    updated = batch_update(weights, sums, counts, hx, hy)

    # Recompute one late node from the *original* models. If anything had leaked from an earlier
    # node's new value, this would disagree.
    late = (shape[0] - 1, shape[1] - 1)
    h = gaussian(shape, late, sigma, (False, False))
    expected = np.einsum("xy,xyf->f", h, sums) / float(np.sum(h * counts))
    np.testing.assert_allclose(updated[late], expected, rtol=1e-12)

    assert not np.shares_memory(updated, weights), "the update must not alias its input"


# ---------------------------------------------------------------------------------------------
# The registry is the batch-legality rule
# ---------------------------------------------------------------------------------------------


def test_only_separable_neighborhoods_have_an_axis_profile() -> None:
    """The mexican hat must never acquire one.

    ``(1 - u) exp(-u)`` does not factor. An outer product of two 1-D Ricker wavelets is a different
    function: it is positive in the diagonal quadrants, +0.165 at 2 sigma where the correct value is
    -0.055, placing an excitatory lobe where the mexican hat must inhibit.
    """
    assert set(AXIS_PROFILES) == {"gaussian", "bubble"}
    assert SIGNED_NEIGHBORHOODS.isdisjoint(AXIS_PROFILES)


def test_every_unsigned_neighborhood_is_separable() -> None:
    """What batch training relies on: anything it accepts, the contraction can express.

    If a future neighborhood is unsigned but not separable, this fails and the choice becomes
    explicit rather than silently approximated.
    """
    unsigned = set(NEIGHBORHOOD_FUNCTIONS) - set(SIGNED_NEIGHBORHOODS)
    assert unsigned == set(AXIS_PROFILES)


def test_resolve_axis_profile_rejects_a_non_separable_neighborhood() -> None:
    """The mexican hat reaches this only if the signed check goes; the message still names why."""
    with pytest.raises(ValueError, match="not separable"):
        resolve_axis_profile("mexican_hat")


def test_resolve_axis_profile_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="not separable"):
        resolve_axis_profile("spectral")


@pytest.mark.parametrize("name", SEPARABLE)
def test_the_axis_profiles_validate_the_radius(name: str) -> None:
    """Validation lives in the profile, so the contraction path cannot skip it."""
    profile = resolve_axis_profile(name)
    with pytest.raises(ValueError, match="must be a finite"):
        profile(np.array([0.0, 1.0]), float("nan"))
    with pytest.raises(ValueError, match="must be a finite"):
        profile(np.array([0.0, 1.0]), -1.0)


def test_the_bubble_accepts_a_zero_radius_and_the_gaussian_does_not() -> None:
    """Unchanged from the per-node forms: zero selects the winner alone, or divides by zero."""
    np.testing.assert_array_equal(
        AXIS_PROFILES["bubble"](np.array([-1.0, 0.0, 1.0]), 0.0), np.array([0.0, 1.0, 0.0])
    )
    with pytest.raises(ValueError, match="must be a finite positive"):
        AXIS_PROFILES["gaussian"](np.array([0.0]), 0.0)


# ---------------------------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("neighborhood", [Neighborhood.GAUSSIAN, Neighborhood.BUBBLE])
@pytest.mark.parametrize("cyclic", list(itertools.product([False, True], repeat=2)))
def test_batch_training_matches_the_per_node_definition_end_to_end(
    neighborhood: Neighborhood, cyclic: tuple[bool, bool]
) -> None:
    """A whole training run, not one update, so any per-iteration drift accumulates into view."""
    shape, n_iteration = (12, 9), 20
    rng = np.random.default_rng(SEED)
    data = rng.normal(size=(90, 4))
    initial = rng.normal(size=(*shape, 4))

    som = python_som.SOM(
        x=shape[0],
        y=shape[1],
        input_len=4,
        neighborhood_function=neighborhood,
        neighborhood_radius=3.0,
        cyclic_x=cyclic[0],
        cyclic_y=cyclic[1],
        random_seed=SEED,
    )
    som._weights = initial.copy()
    som.train(data, n_iteration=n_iteration, mode="batch")

    reference = initial.copy()
    for step in range(n_iteration):
        sigma = som._sigma(step, n_iteration)
        sums, counts = accumulate(data, reference, shape, som._distance_function)
        reference = _per_node_reference(
            reference, sums, counts, shape, neighborhood.value, sigma, cyclic
        )

    scale = float(np.abs(reference).max())
    assert float(np.abs(som.get_weights() - reference).max()) / scale < TOLERANCE


def test_batch_training_still_rejects_the_mexican_hat() -> None:
    """Unchanged, and the message is still about the sign rather than about separability."""
    som = python_som.SOM(x=6, y=6, input_len=3, neighborhood_function="mexican_hat", random_seed=1)
    with pytest.raises(ValueError, match="cannot be used with the 'batch' training mode"):
        som.train(np.random.default_rng(0).normal(size=(20, 3)), n_iteration=5, mode="batch")


def test_the_bubble_is_not_isotropic_under_the_euclidean_metric() -> None:
    """Its metric is Chebyshev, which is why it factors. Pinned because the sources disagree.

    Kohonen Section 4.2 describes the flat neighborhood as "1 up to a certain radius from the
    winner", which reads Euclidean; Vrieze's appendix computes ``MAX(ABS(i - w_i), ABS(j - w_j))``,
    which is Chebyshev, and that is what this package implements. A Euclidean disc would not be
    separable and so could not use this path at all.

    The smallest counterexample is a radius of ``sqrt(50)``: ``(5, 5)`` lies inside a ``sigma = 5``
    square while ``(7, 1)`` lies outside, at equal Euclidean distance from the winner.
    """
    h = bubble((15, 15), (0, 0), 5.0, (False, False))
    assert h[5, 5] == 1.0
    assert h[7, 1] == 0.0
