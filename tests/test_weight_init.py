"""Weight initialization, including the PCA hyperplane fix."""

from __future__ import annotations

import numpy as np
import pytest
import sklearn.decomposition

from tests.conftest import make_som


def test_linear_init_spans_the_principal_plane(blobs: np.ndarray) -> None:
    """Regression: linear initialization used the eigen*values* instead of the eigen*vectors*.

    ``pca.explained_variance_[0]`` and ``[1]`` are two scalars, so every model came out as
    ``c1 * var0 + c2 * var1`` broadcast across all features: a constant vector on the all-ones
    diagonal, rank 1, not a plane.

    Kohonen Section 4.3 asks for "a regular, two-dimensional sequence of vectors taken along a
    hyperplane spanned by the two largest principal components".
    """
    som = make_som(x=6, y=5, input_len=3)
    som.weight_initialization(mode="linear", data=blobs)
    weights = som.get_weights().reshape(-1, 3)

    # not constant within a model
    assert not np.allclose(weights, weights[:, :1])

    # the models span a rank-2 affine subspace
    centred = weights - weights.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    assert singular[1] > 1e-8
    assert singular[2] < 1e-8 * max(1.0, singular[0])


def test_linear_init_is_centred_on_the_data_mean(blobs: np.ndarray) -> None:
    som = make_som(x=7, y=7, input_len=3)
    som.weight_initialization(mode="linear", data=blobs)
    np.testing.assert_allclose(
        som.get_weights().reshape(-1, 3).mean(axis=0), blobs.mean(axis=0), atol=1e-8
    )


def test_linear_init_aligns_with_the_principal_components(blobs: np.ndarray) -> None:
    """The plane the models span must be the plane of the first two components."""
    som = make_som(x=6, y=6, input_len=3)
    som.weight_initialization(mode="linear", data=blobs)
    weights = som.get_weights().reshape(-1, 3)

    pca = sklearn.decomposition.PCA(n_components=2, random_state=0).fit(blobs)
    centred = weights - weights.mean(axis=0)
    # every model offset lies in the span of the two components
    residual = centred - centred @ pca.components_.T @ pca.components_
    assert np.abs(residual).max() < 1e-8


def test_linear_init_lives_in_the_data_space(blobs: np.ndarray) -> None:
    """PCA is fitted on raw data, not standardized data.

    Fitting on standardized data while comparing against raw data during training put the models in
    a different space entirely.
    """
    som = make_som(x=6, y=6, input_len=3)
    som.weight_initialization(mode="linear", data=blobs)
    weights = som.get_weights().reshape(-1, 3)
    assert weights.min() >= blobs.min() - 1.0
    assert weights.max() <= blobs.max() + 1.0


def test_linear_init_is_deterministic(blobs: np.ndarray) -> None:
    a, b = make_som(x=5, y=5), make_som(x=5, y=5, random_seed=999)
    a.weight_initialization(mode="linear", data=blobs)
    b.weight_initialization(mode="linear", data=blobs)
    np.testing.assert_allclose(a.get_weights(), b.get_weights())


def test_linear_init_rejects_a_degenerate_dataset() -> None:
    som = make_som(x=4, y=4, input_len=1)
    with pytest.raises(ValueError, match="at least 2 samples"):
        som.weight_initialization(mode="linear", data=np.zeros((5, 1)))


def test_sample_init_draws_from_the_dataset(blobs: np.ndarray) -> None:
    som = make_som(x=4, y=4, input_len=3)
    som.weight_initialization(mode="sample", data=blobs)
    for model in som.get_weights().reshape(-1, 3):
        assert np.isclose(blobs, model).all(axis=1).any()


def test_sample_init_handles_more_nodes_than_samples() -> None:
    data = np.arange(12, dtype=float).reshape(4, 3)
    som = make_som(x=5, y=5, input_len=3)
    som.weight_initialization(mode="sample", data=data)
    assert som.get_weights().shape == (5, 5, 3)


@pytest.mark.parametrize("sample_mode", ["standard_normal", "uniform"])
def test_random_init_modes(sample_mode: str) -> None:
    som = make_som(x=5, y=5, input_len=3)
    som.weight_initialization(mode="random", sample_mode=sample_mode)
    weights = som.get_weights()
    assert weights.shape == (5, 5, 3)
    if sample_mode == "uniform":
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0


def test_random_init_rejects_an_unknown_sample_mode() -> None:
    som = make_som(x=4, y=4)
    with pytest.raises(ValueError, match="sample_mode"):
        som.weight_initialization(mode="random", sample_mode="cauchy")


def test_unknown_initialization_mode_raises() -> None:
    som = make_som(x=4, y=4)
    with pytest.raises(ValueError, match="mode"):
        # Deliberately invalid; see the note in test_construction.py.
        som.weight_initialization(mode="spectral")  # type: ignore[arg-type]


def test_initialization_is_reproducible_from_the_seed() -> None:
    a = make_som(x=5, y=5, random_seed=7)
    b = make_som(x=5, y=5, random_seed=7)
    a.weight_initialization(mode="random")
    b.weight_initialization(mode="random")
    np.testing.assert_array_equal(a.get_weights(), b.get_weights())
