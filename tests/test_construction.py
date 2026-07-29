"""Constructor validation, automatic sizing and accessors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import sklearn.decomposition
import sklearn.preprocessing

import python_som
from tests.conftest import SEED, make_som

if TYPE_CHECKING:
    import pandas as pd


def test_shape_is_plain_ints() -> None:
    """Regression: ``get_shape`` used to return ``np.uint``.

    That made ``plt.figure(figsize=som.get_shape())`` fail, which the shipped example worked around
    with an explicit ``float()`` and the README did not.
    """
    shape = make_som(x=7, y=4).get_shape()
    assert shape == (7, 4)
    assert all(type(v) is int for v in shape)


def test_weights_have_the_declared_shape() -> None:
    assert make_som(x=7, y=4, input_len=5).get_weights().shape == (7, 4, 5)


def test_seed_is_recorded_and_reused() -> None:
    som = make_som(x=4, y=4, random_seed=99)
    assert som.get_random_seed() == 99


def test_omitting_the_seed_still_records_one() -> None:
    som = make_som(x=4, y=4, random_seed=None)
    assert isinstance(som.get_random_seed(), int)


def test_both_dimensions_none_raises() -> None:
    with pytest.raises(ValueError, match="At least one of the dimensions"):
        python_som.SOM(x=None, y=None, input_len=3)


def test_missing_dimension_without_data_raises() -> None:
    with pytest.raises(ValueError, match="dataset must be provided"):
        python_som.SOM(x=10, y=None, input_len=3)


@pytest.mark.parametrize(("x", "y"), [(0, 5), (5, 0), (-1, 5)])
def test_non_positive_dimensions_raise(x: int, y: int) -> None:
    with pytest.raises(ValueError, match="dimensions must be positive"):
        python_som.SOM(x=x, y=y, input_len=3)


def test_non_positive_input_len_raises() -> None:
    with pytest.raises(ValueError, match="input_len"):
        python_som.SOM(x=5, y=5, input_len=0)


def test_unknown_neighborhood_function_raises_listing_the_options() -> None:
    with pytest.raises(ValueError, match="neighborhood_function") as excinfo:
        python_som.SOM(x=5, y=5, input_len=3, neighborhood_function="sombrero")
    assert "gaussian" in str(excinfo.value)


def test_auto_dimension_uses_the_square_root_of_the_eigenvalue_ratio(
    blobs: np.ndarray,
) -> None:
    """Use the square root of the eigenvalue ratio for the automatic side length.

    Kohonen Section 3.5 asks for side lengths matching the *lengths* of the two largest principal
    components.

    A component is a unit direction, so its length in the data is the standard deviation
    ``sqrt(lambda)``; the ratio is therefore ``sqrt(lambda_1 / lambda_2)``. Using the raw eigenvalue
    ratio, as the code did before, over-elongates the map.
    """
    scaled = sklearn.preprocessing.StandardScaler().fit_transform(blobs)
    pca = sklearn.decomposition.PCA(n_components=2, random_state=0)
    pca.fit(scaled)
    ratio = float(np.sqrt(pca.explained_variance_[0] / pca.explained_variance_[1]))

    som = python_som.SOM(x=20, y=None, input_len=3, data=blobs, random_seed=SEED)
    assert som.get_shape() == (20, max(1, round(20 / ratio)))


def test_auto_dimension_derives_x_when_y_is_given(blobs: np.ndarray) -> None:
    """The mirror of the case above; both branches of the ratio must behave the same way.

    Only ``y=None`` was covered before, which left half of the changed auto-sizing code untested.
    """
    scaled = sklearn.preprocessing.StandardScaler().fit_transform(blobs)
    pca = sklearn.decomposition.PCA(n_components=2, random_state=0)
    pca.fit(scaled)
    ratio = float(np.sqrt(pca.explained_variance_[0] / pca.explained_variance_[1]))

    som = python_som.SOM(x=None, y=20, input_len=3, data=blobs, random_seed=SEED)
    assert som.get_shape() == (max(1, round(20 / ratio)), 20)


def test_auto_dimension_is_symmetric_in_its_two_branches(blobs: np.ndarray) -> None:
    """Deriving x from y, or y from x, must apply the same ratio in the same direction."""
    from_y = python_som.SOM(x=None, y=20, input_len=3, data=blobs, random_seed=SEED).get_shape()
    from_x = python_som.SOM(x=20, y=None, input_len=3, data=blobs, random_seed=SEED).get_shape()
    assert from_y[0] == from_x[1]
    assert from_y[1] == from_x[0] == 20


def test_auto_dimension_never_returns_zero() -> None:
    """Floor division could previously produce a zero-length side for a strongly 1-D dataset."""
    rng = np.random.default_rng(SEED)
    almost_1d = np.column_stack(
        [rng.normal(scale=50.0, size=200), rng.normal(scale=0.01, size=200)]
    )
    som = python_som.SOM(x=3, y=None, input_len=2, data=almost_1d, random_seed=SEED)
    assert som.get_shape()[1] >= 1


def test_auto_dimension_accepts_a_dataframe(frame: pd.DataFrame) -> None:
    som = python_som.SOM(x=12, y=None, input_len=3, data=frame, random_seed=SEED)
    assert som.get_shape()[0] == 12


def test_setters_update_the_hyperparameters() -> None:
    som = make_som(x=4, y=4)
    som.set_learning_rate(0.25)
    som.set_neighborhood_radius(3.5)
    assert som._learning_rate == pytest.approx(0.25)
    assert som._neighborhood_radius == pytest.approx(3.5)


def test_neighborhood_is_reachable_from_the_public_api() -> None:
    som = make_som(x=9, y=9, neighborhood_function="mexicanhat")
    h = som.neighborhood((4, 4), 2.0)
    assert h.shape == (9, 9)
    assert h[4, 4] == pytest.approx(1.0)


def test_cyclic_flags_reach_the_neighborhood() -> None:
    flat = make_som(x=10, y=10, cyclic_x=False).neighborhood((0, 5), 1.0)
    torus = make_som(x=10, y=10, cyclic_x=True, cyclic_y=True).neighborhood((0, 5), 1.0)
    assert flat[9, 5] < 1e-12
    assert torus[9, 5] > 0.5
