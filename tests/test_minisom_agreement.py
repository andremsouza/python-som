"""Check this package against MiniSom, the closest comparable implementation.

Two independent implementations of Kohonen's equations agreeing is stronger evidence than either
one's own tests, which is the same reasoning behind ``tests/test_linalg_matches_sklearn.py``. That
file earned its place: it failed on first run, the *reference* turned out to be the wrong one, and
pursuing it exposed a real 5.8% defect in this package's linear initialization.

The reason this file exists at all is that the two libraries look far more interchangeable than they
are. Method names line up (``activate``, ``winner``, ``quantization``, ``quantization_error``,
``get_weights``) while the mathematics behind some of them does not, so a comparison written from
the names alone would compare different algorithms and attribute the difference to something else.
What agrees and what does not, measured rather than assumed:

===========================  =============================================================
agrees exactly               ``asymptotic_decay``, at 0.0
agrees to round-off          the gaussian (1.1e-16), sequential training (2.8e-16),
                             batch training (3.1e-15)
**does not agree**           the mexican hat: ``(1-u)e^-u`` here, ``(1-2u)e^-u`` there, so the
                             zero crossing sits at sqrt(2)*sigma rather than sigma
**does not agree**           the bubble: ``max(|dx|,|dy|) <= round(sigma)`` here, strict
                             ``c-sigma < i < c+sigma`` there. At sigma=1, 9 nodes against 1
===========================  =============================================================

Both disagreements are convention choices rather than defects on either side, and MiniSom's mexican
hat is isotropic, so it does **not** have the separability bug this package fixed in 0.2.0. They are
pinned here so that nobody later "fixes" one into the other, and so the comparison published in
``docs/explanation/`` cannot quietly stop being true.

The end-to-end tests matter most. They are what makes it legitimate to time the two libraries
against each other: if the trained models agree to round-off, a difference in wall time is
implementation and nothing else. ``benchmarks/bench_vs_minisom.py`` relies on exactly that, and this
is where the claim is rechecked on every run.

Nothing here is timed. Timing belongs in ``benchmarks/``, where a noisy shared runner cannot turn it
into a spurious failure.

Skipped wholesale when MiniSom is absent, so the suite still runs in an environment that installed
only the runtime dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import python_som
from python_som._core._decay import asymptotic_decay
from python_som._core._neighborhood import bubble, gaussian, mexican_hat

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

minisom = pytest.importorskip("minisom")

#: Round-off scale for one evaluation of a neighborhood function. Measured at 1.1e-16, which is a
#: single unit in the last place; this leaves room for a different BLAS without admitting a
#: different formula.
NEIGHBORHOOD_TOLERANCE = 1e-14

#: Round-off scale for a whole training run, relative to the largest model component. Measured at
#: 1.5e-16 for sequential and 1.3e-15 for batch over 100 to 200 iterations, the latter larger
#: because the two accumulate Eq. (8) in a different order. Three orders of headroom: loose enough
#: to survive another machine, tight enough that a real divergence in the update rule fails it.
TRAINING_TOLERANCE = 1e-12

#: Independent of the fixture seeds: this file compares two implementations, not two runs.
SEED = 20260730

#: Grids to sweep, including a degenerate single-row map.
SHAPES = [(5, 5), (7, 4), (12, 9), (1, 6)]

#: Radii to sweep. Below 1 and above the grid are both legal and both worth covering.
RADII = [0.5, 1.0, 2.5, 3.0, 7.0]


def _data(n_samples: int, n_features: int) -> npt.NDArray[np.floating]:
    """Build a reproducible dataset.

    :param n_samples: Number of samples.
    :param n_features: Number of features.
    :return: The dataset.
    """
    rng = np.random.default_rng(SEED)
    return rng.normal(size=(n_samples, n_features))


def _peer(shape: tuple[int, int], n_features: int, **kwargs: object) -> Any:  # noqa: ANN401
    """Build a MiniSom of the given shape.

    ``Any`` because MiniSom ships no type information, so there is nothing more precise to say
    about what comes back.

    :param shape: Grid shape.
    :param n_features: Number of input features.
    :param kwargs: Passed through to the MiniSom constructor.
    :return: The MiniSom.
    """
    return minisom.MiniSom(shape[0], shape[1], n_features, random_seed=SEED, **kwargs)


def _relative_difference(ours: npt.NDArray[np.floating], theirs: npt.NDArray[np.floating]) -> float:
    """Return the largest absolute difference, scaled by the largest value being compared.

    Preferred over ``assert_allclose``'s ``rtol`` because a model component that happens to sit near
    zero would fail a relative comparison for no reason worth failing over.

    :param ours: This package's result.
    :param theirs: MiniSom's result.
    :return: The scaled difference.
    """
    scale = float(np.abs(theirs).max())
    difference = float(np.abs(ours - theirs).max())
    return difference / scale if scale else difference


# ---------------------------------------------------------------------------------------------
# The neighborhood functions
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("sigma", RADII)
def test_the_gaussians_are_the_same_function(shape: tuple[int, int], sigma: float) -> None:
    """Every node of every grid, not one spot check.

    The two are written differently and it is not obvious they agree. MiniSom takes the outer
    product of two one-dimensional gaussians; this package exponentiates a single squared grid
    distance. They coincide because the exponential is separable,
    ``exp(-(dx^2+dy^2)/2s^2) == exp(-dx^2/2s^2) * exp(-dy^2/2s^2)``, which is a property of the
    gaussian alone and does not carry to the other two neighborhoods. That is the whole reason the
    head-to-head benchmark uses the gaussian.
    """
    peer = _peer(shape, 3)
    for cx in range(shape[0]):
        for cy in range(shape[1]):
            ours = gaussian(shape, (cx, cy), sigma, (False, False))
            theirs = peer._gaussian((cx, cy), sigma)
            assert ours.shape == theirs.shape
            np.testing.assert_allclose(ours, theirs, atol=NEIGHBORHOOD_TOLERANCE)


def test_the_mexican_hats_differ_by_a_factor_of_two_in_the_linear_term() -> None:
    """A deliberate difference, pinned so it is not mistaken for a bug in either package.

    This package uses ``(1 - u) * exp(-u)`` with ``u = r^2 / 2s^2``, so the profile crosses zero at
    ``r = sqrt(2) * s``. MiniSom uses ``(1 - 2u) * exp(-u)``, crossing at ``r = s``. Both are
    standard ways to write a Ricker wavelet and neither is wrong; they are simply not the same
    function, so the two cannot be compared at equal sigma.

    Worth stating plainly because this package's own mexican hat *was* wrong once, in the form
    contributed in PR #2: an outer product of two one-dimensional Ricker wavelets, which is not
    radial and produced a positive side lobe of +0.165 on the diagonal at 2 sigma where the correct
    value is -0.055. MiniSom's applies its profile to a single squared distance and so never had
    that defect. This test asserts the remaining difference is only the factor of two.
    """
    shape, centre, sigma = (41, 41), (20, 20), 3.0
    ours = mexican_hat(shape, centre, sigma, (False, False))
    theirs = _peer(shape, 3)._mexican_hat(centre, sigma)

    radius = np.sqrt(
        np.add.outer(
            (np.arange(shape[0]) - centre[0]) ** 2.0, (np.arange(shape[1]) - centre[1]) ** 2.0
        )
    )
    # The crossing is bracketed rather than solved for, since the grid only samples the profile.
    assert radius[ours > 0].max() < np.sqrt(2) * sigma < radius[ours < 0].min()
    assert radius[theirs > 0].max() < sigma < radius[theirs < 0].min()

    # Both are radial, which is the property PR #2 got wrong, and both peak at 1 on the winner.
    assert ours[centre] == pytest.approx(1.0)
    assert theirs[centre] == pytest.approx(1.0)
    for profile in (ours, theirs):
        for distance in np.unique(radius):
            shell = profile[radius == distance]
            assert shell.max() - shell.min() == pytest.approx(0.0, abs=1e-15)


def test_the_bubbles_select_different_neighbourhoods() -> None:
    """Also deliberate, and a much larger difference than the mexican hat's.

    This package rounds sigma and includes the boundary, ``max(|dx|,|dy|) <= round(sigma)``, which
    at sigma=1 is the 3x3 block around the winner. MiniSom uses strict inequalities against an
    unrounded sigma, ``c - sigma < i < c + sigma``, selecting only the winner at sigma=1. Nine
    nodes against one is not a rounding difference, and a benchmark that used the bubble at equal
    sigma would be comparing a neighbourhood against a point update.
    """
    shape, centre = (9, 9), (4, 4)
    # sigma=1 is an integer >= 1, so MiniSom's "sigma should be an integer" warning does not fire
    # and the suite's filterwarnings=error setting is not tripped. The next test covers the case
    # where it does.
    peer = _peer(shape, 3, neighborhood_function="bubble", sigma=1)

    assert bubble(shape, centre, 1.0, (False, False)).sum() == 9
    assert peer._bubble(centre, 1.0).sum() == 1


def test_minisom_warns_about_a_non_integer_bubble_radius() -> None:
    """Recorded because this suite runs ``filterwarnings = ["error"]``.

    A future test that constructs a MiniSom with the bubble and a decayed, non-integer sigma would
    fail with a ``UserWarning`` rather than an assertion, which is a confusing way to find out. This
    package places no such restriction on its own bubble.
    """
    with pytest.warns(UserWarning, match="sigma should be an integer"):
        _peer((9, 9), 3, neighborhood_function="bubble", sigma=1.5)


# ---------------------------------------------------------------------------------------------
# The pieces the training loops are built from
# ---------------------------------------------------------------------------------------------


def test_the_asymptotic_decays_are_identical() -> None:
    """Exactly ``0.0``, not a tolerance: both evaluate ``x / (1 + t / (max_t / 2))``.

    This is what lets the radius and learning-rate schedules be matched between the two libraries by
    naming the same decay on each side, rather than by reimplementing one of them.
    """
    peer = _peer((5, 5), 3)
    for x in (0.5, 1.0, 3.0, 10.0):
        for max_t in (10, 100, 1000):
            for t in range(0, max_t, max(1, max_t // 7)):
                assert asymptotic_decay(x, t, max_t) - peer._asymptotic_decay(x, t, max_t) == 0.0


def test_the_winners_agree_for_the_same_models() -> None:
    """Same best-matching unit for every sample, including the coordinate convention."""
    shape, n_features = (7, 5), 4
    data = _data(40, n_features)
    weights = np.random.default_rng(SEED).normal(size=(*shape, n_features))

    som = python_som.SOM(x=shape[0], y=shape[1], input_len=n_features, random_seed=SEED)
    som._weights = weights.copy()
    peer = _peer(shape, n_features)
    peer._weights = weights.copy()

    for sample in data:
        assert som.winner(sample) == tuple(peer.winner(sample))


def test_the_winners_agree_on_a_deliberate_tie() -> None:
    """Ties go to the first index in C order in both, which is arbitrary but must be the same one.

    Constructed rather than hoped for: two nodes are placed at exactly equal distance from the
    sample. If the two packages broke ties differently, every later comparison would drift apart on
    data with duplicate models, which is common right after random initialization from a small
    dataset.
    """
    shape, n_features = (3, 3), 2
    weights = np.full((*shape, n_features), 10.0)
    weights[0, 2] = [1.0, 0.0]
    weights[2, 0] = [1.0, 0.0]
    sample = np.array([1.0, 0.0])

    som = python_som.SOM(x=shape[0], y=shape[1], input_len=n_features, random_seed=SEED)
    som._weights = weights.copy()
    peer = _peer(shape, n_features)
    peer._weights = weights.copy()

    assert som.winner(sample) == (0, 2)
    assert tuple(peer.winner(sample)) == (0, 2)


def test_the_quantization_errors_agree() -> None:
    """The metric the benchmark reports, so the two must define it the same way. They do.

    Both are the mean Euclidean distance from each sample to its best-matching model. Note that the
    similarly named ``quantization`` does **not** agree and is not interchangeable: this package
    returns one distance per sample, MiniSom returns the winning model vector per sample. Only
    ``quantization_error`` is the comparable pair, which is asserted below.
    """
    shape, n_features = (6, 6), 3
    data = _data(50, n_features)
    weights = np.random.default_rng(SEED).normal(size=(*shape, n_features))

    som = python_som.SOM(x=shape[0], y=shape[1], input_len=n_features, random_seed=SEED)
    som._weights = weights.copy()
    peer = _peer(shape, n_features)
    peer._weights = weights.copy()

    assert som.quantization_error(data) == pytest.approx(peer.quantization_error(data), abs=1e-14)
    assert som.quantization(data).shape == (len(data),)
    assert peer.quantization(data).shape == (len(data), n_features)


# ---------------------------------------------------------------------------------------------
# End to end: the claim the benchmark rests on
# ---------------------------------------------------------------------------------------------

#: Grid, sample count, feature count and iteration count per end-to-end case.
TRAINING_CASES = [((8, 8), 60, 3, 100), ((12, 9), 120, 5, 200), ((20, 20), 200, 4, 150)]

#: Initial radius and learning rate, shared by both libraries. Chosen with the floor below in mind.
SIGMA_0 = 3.0
LEARNING_RATE_0 = 0.5


def _matched_som(shape: tuple[int, int], n_features: int) -> python_som.SOM:
    """Build a map whose schedules match what MiniSom will be given.

    Everything that has to agree between the two libraries is fixed here rather than at the call
    sites, so a case cannot silently opt out of one of the controls.

    :param shape: Grid shape.
    :param n_features: Number of input features.
    :return: The map.
    """
    return python_som.SOM(
        x=shape[0],
        y=shape[1],
        input_len=n_features,
        learning_rate=LEARNING_RATE_0,
        learning_rate_decay=asymptotic_decay,
        neighborhood_radius=SIGMA_0,
        neighborhood_radius_decay=asymptotic_decay,
        neighborhood_function="gaussian",
        random_seed=SEED,
    )


@pytest.mark.parametrize(("shape", "n_samples", "n_features", "n_iteration"), TRAINING_CASES)
def test_the_radius_floor_never_binds_in_these_cases(
    shape: tuple[int, int], n_samples: int, n_features: int, n_iteration: int
) -> None:
    """A precondition of the two tests below, checked rather than assumed.

    This package floors the decayed radius at ``min_neighborhood_radius`` (Kohonen Section 4.2:
    the radius "should always remain, say, above half of the grid spacing"). MiniSom has no such
    floor, so if it ever engaged the two would be running different schedules and the agreement
    below would be measuring the wrong thing. At sigma=3 over 100 iterations the final radius is
    1.007, comfortably above the 0.5 default, but that is a property of the chosen cases rather than
    of the code, so it is asserted per case.
    """
    del n_samples, n_features  # part of the shared case tuple, not needed here
    som = _matched_som(shape, 3)
    for t in range(n_iteration):
        floored = som._sigma(t, n_iteration)
        assert floored == asymptotic_decay(SIGMA_0, t, n_iteration), (
            f"the radius floor engaged at t={t}; pick a larger sigma or fewer iterations"
        )


@pytest.mark.parametrize(("shape", "n_samples", "n_features", "n_iteration"), TRAINING_CASES)
def test_sequential_training_agrees_end_to_end(
    shape: tuple[int, int], n_samples: int, n_features: int, n_iteration: int
) -> None:
    """Kohonen Eq. (3), driven identically on both sides, must land on the same models.

    Note which MiniSom method this is. ``train_batch`` is *stepwise* training in sequential sample
    order, the counterpart of ``mode="sequential"`` here; MiniSom's Eq. (8) implementation is called
    ``train_batch_offline`` and is exercised by the next test. Reaching for the name that matches
    this package's would compare per-sample gradient steps against a weighted mean.

    Sequential rather than random because the index sequences then coincide exactly:
    ``np.resize(np.arange(n), T)`` here against ``arange(T) % n`` there. The random modes are
    genuinely different, i.i.d. draws here against a shuffle of a fixed multiset there, so no seed
    reconciles them.

    Measured agreement: 2.2e-16 to 2.8e-16 relative, after up to 200 iterations.
    """
    data = _data(n_samples, n_features)
    initial = np.random.default_rng(SEED + 1).normal(size=(*shape, n_features))

    som = _matched_som(shape, n_features)
    som._weights = initial.copy()
    som.train(data, n_iteration=n_iteration, mode="sequential")

    peer = _peer(
        shape,
        n_features,
        sigma=SIGMA_0,
        learning_rate=LEARNING_RATE_0,
        decay_function="asymptotic_decay",
        sigma_decay_function="asymptotic_decay",
        neighborhood_function="gaussian",
    )
    peer._weights = initial.copy()
    peer.train_batch(data, n_iteration)

    assert _relative_difference(som.get_weights(), peer.get_weights()) < TRAINING_TOLERANCE


@pytest.mark.parametrize(("shape", "n_samples", "n_features", "n_iteration"), TRAINING_CASES)
def test_batch_training_agrees_end_to_end(
    shape: tuple[int, int], n_samples: int, n_features: int, n_iteration: int
) -> None:
    """Kohonen Eq. (8) on both sides, once MiniSom's extra learning rate is removed.

    MiniSom's ``train_batch_offline`` cites the same paper this package works from, and computes the
    same neighborhood-weighted mean, but then relaxes toward it rather than assigning it:
    ``w <- (1 - eta) w + eta * mean``. Eq. (8) has no step size, so the two coincide only at
    ``eta == 1``. Passing ``learning_rate=1.0`` alone is not enough, because the rate is still fed
    through a decay; a constant callable is what pins it.

    That difference is the reason this test exists rather than being folded into the one above. It
    is exactly the kind of thing that would otherwise show up as a modest, believable quality gap in
    a published comparison.

    Measured agreement: 6.0e-16 to 1.3e-15 relative, looser than the sequential case because the two
    accumulate the same sum in a different order.
    """
    data = _data(n_samples, n_features)
    initial = np.random.default_rng(SEED + 1).normal(size=(*shape, n_features))

    som = _matched_som(shape, n_features)
    som._weights = initial.copy()
    som.train(data, n_iteration=n_iteration, mode="batch")

    peer = _peer(
        shape,
        n_features,
        sigma=SIGMA_0,
        learning_rate=1.0,
        decay_function=lambda rate, t, max_t: 1.0,  # noqa: ARG005
        sigma_decay_function="asymptotic_decay",
        neighborhood_function="gaussian",
    )
    peer._weights = initial.copy()
    peer.train_batch_offline(data, n_iteration)

    assert _relative_difference(som.get_weights(), peer.get_weights()) < TRAINING_TOLERANCE


def test_minisom_batch_without_the_pinned_rate_does_not_agree() -> None:
    """The control from the previous test is load-bearing, so its absence is asserted too.

    Without it the two still look similar, which is the danger: they diverge enough to move a
    published quantization error but not enough to look like a bug. Guards against someone
    simplifying the constant-rate callable away because "the default is close enough".
    """
    shape, n_features, n_iteration = (12, 9), 5, 100
    data = _data(120, n_features)
    initial = np.random.default_rng(SEED + 1).normal(size=(*shape, n_features))

    som = _matched_som(shape, n_features)
    som._weights = initial.copy()
    som.train(data, n_iteration=n_iteration, mode="batch")

    peer = _peer(
        shape,
        n_features,
        sigma=SIGMA_0,
        learning_rate=LEARNING_RATE_0,
        decay_function="asymptotic_decay",
        sigma_decay_function="asymptotic_decay",
        neighborhood_function="gaussian",
    )
    peer._weights = initial.copy()
    peer.train_batch_offline(data, n_iteration)

    assert _relative_difference(som.get_weights(), peer.get_weights()) > TRAINING_TOLERANCE
