"""Check the NumPy PCA and z-score against the scikit-learn implementations they replaced.

0.4.0 dropped scikit-learn as a runtime dependency and reimplemented the two things this package
used it for on ``np.linalg.svd``. That trades 264 MB of required install for about twenty lines of
linear algebra, and it is the one change in the release a reviewer should be least willing to take
on trust.

So it is not taken on trust. scikit-learn stays in the ``dev`` extra purely so this file can run,
and every CI run re-derives the same fits both ways and compares them. The claim under test is not
"close enough" but "the same numbers": tolerances here are at the scale of double-precision
round-off, not at the scale of a plausible-looking answer.

The module is skipped rather than failed where scikit-learn is absent, so that the package's own
test suite still runs in an environment that installed only the runtime dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from python_som._core._linalg import pca, standardize

sklearn_decomposition = pytest.importorskip("sklearn.decomposition")
sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")

#: Round-off scale for a double-precision SVD. Agreement is expected at about 1e-15; this leaves
#: room for the conditioning of the harder shapes without admitting a genuinely different answer.
TOLERANCE = 1e-10

#: Shapes worth covering, including the awkward ones. ``(6, 12)`` has fewer samples than features,
#: which is where a naive covariance-matrix implementation would disagree; ``(200, 2)`` is exactly
#: as many components as are asked for.
SHAPES = [(50, 5), (200, 2), (6, 12), (30, 8), (1000, 4)]

#: Independent of the fixture seeds: this file compares two implementations, not two runs.
SEED = 20260730


def _matrix(shape: tuple[int, int], *, scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Build a reproducible random matrix.

    :param shape: Shape of the matrix.
    :param scale: Multiplier applied to every entry, to vary conditioning.
    :param offset: Constant added to every entry, to move it away from the origin.
    :return: The matrix.
    """
    rng = np.random.default_rng(SEED + shape[0] * 100 + shape[1])
    return rng.normal(size=shape) * scale + offset


@pytest.mark.parametrize("shape", SHAPES)
def test_pca_matches_sklearn(shape: tuple[int, int]) -> None:
    """Mean, components and explained variance all agree, signs included."""
    data = _matrix(shape)
    n_components = min(2, *shape)

    theirs = sklearn_decomposition.PCA(n_components=n_components, random_state=0).fit(data)
    ours = pca(data, n_components=n_components)

    np.testing.assert_allclose(ours.mean, theirs.mean_, atol=TOLERANCE)
    np.testing.assert_allclose(ours.explained_variance, theirs.explained_variance_, atol=TOLERANCE)
    # Not abs(): the sign convention is the part most likely to be wrong, so it is what is asserted.
    np.testing.assert_allclose(ours.components, theirs.components_, atol=TOLERANCE)


@pytest.mark.parametrize(("scale", "offset"), [(1e-6, 0.0), (1e6, 0.0), (1.0, 1e5), (1e-3, -50.0)])
def test_pca_matches_sklearn_away_from_unit_scale(scale: float, offset: float) -> None:
    """Badly scaled and far-from-origin data, against scikit-learn's *exact* solver.

    Deliberately ``svd_solver="full"`` rather than the default ``"auto"``, and the reason is worth
    recording because the first version of this test failed and the reference was what was wrong.

    Since 1.5, ``"auto"`` selects ``covariance_eigh`` when samples comfortably outnumber features,
    which forms the covariance matrix and eigendecomposes it. Squaring the data squares its
    condition number, so on data centred far from the origin that path loses precision. Measured on
    ``(80, 6)`` offset by 1e5, against a reference centred in ``longdouble`` before decomposing:

    ==========================  ====================
    solver                      relative error
    ==========================  ====================
    sklearn ``auto``            1.4e-06 to 5.5e-06
    sklearn ``full``            ~1e-15
    this implementation         ~1e-15
    ==========================  ====================

    So the SVD of the centred matrix is the more accurate of the two, and comparing against
    ``"auto"`` would fail a correct implementation for agreeing with the mathematics rather than
    with scikit-learn's default. ``"full"`` is the same algorithm this package uses and is what the
    comparison is worth making against.
    """
    data = _matrix((80, 6), scale=scale, offset=offset)

    theirs = sklearn_decomposition.PCA(n_components=2, random_state=0, svd_solver="full").fit(data)
    ours = pca(data)

    np.testing.assert_allclose(ours.mean, theirs.mean_, rtol=1e-9)
    np.testing.assert_allclose(ours.explained_variance, theirs.explained_variance_, rtol=1e-9)
    np.testing.assert_allclose(ours.components, theirs.components_, atol=1e-9)


@pytest.mark.parametrize("offset", [0.0, 1e5, -1e6])
def test_pca_is_accurate_against_a_higher_precision_reference(offset: float) -> None:
    """Agreement with scikit-learn is evidence; agreement with the mathematics is the claim.

    Centring in ``longdouble`` before decomposing removes the cancellation that far-from-origin
    data causes, giving a reference that does not depend on any library's solver choice. This is
    the assertion that would survive scikit-learn changing its defaults again.
    """
    data = _matrix((80, 6), offset=offset)

    exact = data.astype(np.longdouble)
    centred = (exact - exact.mean(axis=0)).astype(np.float64)
    truth = np.linalg.svd(centred, compute_uv=False)[:2] ** 2 / (data.shape[0] - 1)

    np.testing.assert_allclose(pca(data).explained_variance, truth, rtol=1e-12)


def test_pca_sign_convention_is_v_based_not_u_based() -> None:
    """Pin the convention itself, not just agreement on one dataset.

    scikit-learn's ``svd_flip`` defaults to ``u_based_decision=True``, but its PCA passes ``False``.
    Taking the default would still produce a valid PCA, so no test of orthonormality or explained
    variance would notice; only comparing the signs does. This asserts the resulting property
    directly: each component's largest-magnitude entry is positive.
    """
    for shape in [(50, 5), (30, 8), (6, 12)]:
        components = pca(_matrix(shape)).components
        dominant = np.abs(components).argmax(axis=1)
        leading = components[np.arange(components.shape[0]), dominant]
        assert (leading > 0).all(), f"{shape}: expected positive dominant loadings, got {leading}"


@pytest.mark.parametrize("shape", SHAPES)
def test_standardize_matches_standard_scaler(shape: tuple[int, int]) -> None:
    """The z-score agrees column for column."""
    data = _matrix(shape)
    theirs = sklearn_preprocessing.StandardScaler().fit_transform(data)
    np.testing.assert_allclose(standardize(data), theirs, atol=TOLERANCE)


def test_standardize_matches_standard_scaler_on_a_constant_column() -> None:
    """A column with no variance must not become ``inf`` or ``nan``.

    Dividing by its standard deviation of zero is the obvious implementation and the wrong one.
    scikit-learn scales such a column by 1, leaving it centred at zero, and so does this.
    """
    data = _matrix((40, 4))
    data[:, 2] = 7.0

    ours = standardize(data)
    theirs = sklearn_preprocessing.StandardScaler().fit_transform(data)

    assert np.isfinite(ours).all(), "a constant column produced a non-finite value"
    np.testing.assert_allclose(ours[:, 2], 0.0, atol=TOLERANCE)
    np.testing.assert_allclose(ours, theirs, atol=TOLERANCE)


def test_standardize_matches_standard_scaler_on_a_near_constant_column() -> None:
    """Constant in exact arithmetic, merely tiny in floating point.

    This is why the guard is scikit-learn's Chan-Golub-LeVeque bound rather than ``variance == 0``:
    a column built by arithmetic that should cancel exactly can retain a variance of about 1e-30,
    which passes an equality test and then divides the column by roughly 1e-15.
    """
    data = _matrix((40, 3))
    data[:, 1] = np.full(40, 1e8) + np.linspace(0, 1e-9, 40)

    ours = standardize(data)
    theirs = sklearn_preprocessing.StandardScaler().fit_transform(data)

    assert np.isfinite(ours).all()
    np.testing.assert_allclose(ours, theirs, atol=TOLERANCE)


def test_linear_initialization_is_accurate_far_from_the_origin() -> None:
    """Regression for a real defect in 0.3.0, not merely a precision preference.

    Linear initialization fits PCA on **raw** data on purpose, so the models share the space of the
    inputs they are compared against. Through 0.3.0 that PCA went to scikit-learn's ``auto`` solver,
    which for these shapes is ``covariance_eigh``: it forms the covariance matrix, and when the mean
    is large relative to the spread, squaring the data destroys the information the spread lives in.

    On ``(150, 4)`` data offset by 1e7, 0.3.0's second explained variance was wrong by **5.8%**, and
    the resulting models differed from the correct ones by 2.43 against a total model spread of 2.0
    -- an error larger than the structure being initialized. Data offset far from the origin is not
    exotic: timestamps, easting/northing coordinates and absolute sensor readings all look like it.

    The reference is computed in ``longdouble`` so it depends on no library's solver choice.
    """
    rng = np.random.default_rng(3)
    data = rng.normal(size=(150, 4)) + 1e7

    exact = data.astype(np.longdouble)
    centred = (exact - exact.mean(axis=0)).astype(np.float64)
    truth = np.linalg.svd(centred, compute_uv=False)[:2] ** 2 / (data.shape[0] - 1)

    np.testing.assert_allclose(pca(data).explained_variance, truth, rtol=1e-12)

    # Show the defect this protects against, when the installed scikit-learn still exhibits it.
    # Conditioned on the solver rather than assumed: `covariance_eigh` arrived in 1.5, the CI matrix
    # runs an older pin on Python 3.10, and a future release could pick differently again. The
    # assertion above is the claim; this is the illustration, and it must not fail for being stale.
    legacy = sklearn_decomposition.PCA(n_components=2, random_state=0).fit(data)
    if getattr(legacy, "_fit_svd_solver", None) == "covariance_eigh":
        error = (np.abs(legacy.explained_variance_ - truth) / truth).max()
        assert error > 1e-3, f"expected the covariance path to be inaccurate here, got {error:.2e}"


def test_the_numpy_implementation_is_the_one_the_package_uses() -> None:
    """Guard against the differential test quietly becoming the reason the dependency stays.

    The corresponding negative — that no module in the core imports scikit-learn at all — is
    asserted by ``tests/test_core_boundary.py``, which scans imports rather than behaviour.
    """
    som_pca = pca(_matrix((40, 4)))
    assert isinstance(som_pca.components, np.ndarray)
    assert som_pca.components.shape == (2, 4)
