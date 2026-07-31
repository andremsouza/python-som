"""The vectorised best-matching-unit search must select the same nodes as the definition.

Eq. (4) of Kohonen (2013) is ``c = argmin_i ||x - m_i||``. For the Euclidean distance the search
expands that norm into a matrix product, which is much faster and is *not* obviously the same thing.
These tests hold the two properties that make it the same thing: it picks the node the definition
picks, and it stays the Euclidean map rather than becoming the dot-product map of Section 4.5.

The expansion also has a failure mode that only appears far from the origin, and it is severe enough
to have its own regression test below.
"""

from __future__ import annotations

import numpy as np
import pytest

import python_som
from python_som._core._distance import euclidean_distance
from python_som._core._match import accumulate, bmu_indices, quantization, winner

#: Fixed so a failure is reproducible.
SEED = 20260730


def _exact(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Select the best-matching node by the definition, one full norm per sample.

    :param data: Dataset.
    :param weights: Models.
    :return: One flat node index per sample.
    """
    flat = weights.reshape(-1, weights.shape[-1])
    return np.array([np.linalg.norm(x - flat, axis=-1).argmin() for x in data])


def _case(
    shape: tuple[int, int], n_samples: int, n_features: int, offset: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Build models and a dataset, optionally far from the origin.

    :param shape: Grid shape.
    :param n_samples: Number of samples.
    :param n_features: Number of features.
    :param offset: Constant added to both, to move them away from the origin.
    :return: Models and dataset.
    """
    rng = np.random.default_rng(SEED)
    return (
        rng.normal(size=(*shape, n_features)) + offset,
        rng.normal(size=(n_samples, n_features)) + offset,
    )


# ---------------------------------------------------------------------------------------------
# It selects what the definition selects
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "n_samples", "n_features"),
    [((5, 5), 50, 3), ((20, 20), 200, 4), ((40, 30), 500, 8), ((1, 9), 40, 2), ((60, 60), 300, 12)],
)
def test_the_fast_search_selects_the_same_nodes(
    shape: tuple[int, int], n_samples: int, n_features: int
) -> None:
    """Identical indices, not close ones: a different node is a different answer."""
    weights, data = _case(shape, n_samples, n_features)
    np.testing.assert_array_equal(
        bmu_indices(data, weights, euclidean_distance), _exact(data, weights)
    )


def test_it_agrees_with_winner_for_every_sample() -> None:
    """The single-sample path and the whole-dataset path must not drift apart."""
    weights, data = _case((7, 5), 60, 4)
    flat = bmu_indices(data, weights, euclidean_distance)
    rows, columns = np.unravel_index(flat, weights.shape[:2])
    for sample, row, column in zip(data, rows, columns, strict=True):
        assert (int(row), int(column)) == winner(sample, weights, euclidean_distance)


def test_ties_go_to_the_first_node_in_c_order() -> None:
    """Arbitrary but fixed, and it must match ``argmin``, which is what ``winner`` uses."""
    weights = np.full((3, 3, 2), 10.0)
    weights[0, 2] = [1.0, 0.0]
    weights[2, 0] = [1.0, 0.0]
    data = np.array([[1.0, 0.0]])

    assert int(bmu_indices(data, weights, euclidean_distance)[0]) == 2
    assert winner(data[0], weights, euclidean_distance) == (0, 2)


def test_a_chunk_boundary_does_not_change_the_result() -> None:
    """The search runs in blocks, so a dataset larger than one block exercises the seam.

    At 60x60 the block holds about 17 samples, so 500 samples cross it many times.
    """
    weights, data = _case((60, 60), 500, 6)
    np.testing.assert_array_equal(
        bmu_indices(data, weights, euclidean_distance), _exact(data, weights)
    )


# ---------------------------------------------------------------------------------------------
# The cancellation the expansion would otherwise cause
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0.0, 1e3, 1e6, 1e9, 1e12])
def test_the_search_is_exact_far_from_the_origin(offset: float) -> None:
    """Regression for catastrophic cancellation, with the measured numbers.

    ``||x - w||^2 = ||x||^2 - 2 x.w + ||w||^2`` is exact in real arithmetic and not in floating
    point. With models offset by 1e9, ``||w||^2`` is about 1e18 while the differences between models
    are of order 1, so the subtraction loses every significant digit. Measured without centring:

    ======  =========================
    offset  samples given a wrong node
    ======  =========================
    1e6     0 of 500
    1e9     **500 of 500**
    1e12    **500 of 500**
    ======  =========================

    Subtracting the models' mean from both sides is exact in ``||x - w||``, costs 1%, and removes it
    at every offset above. Data far from the origin is not exotic: timestamps, easting and northing
    coordinates and absolute sensor readings all look like this, and it is the same failure this
    package fixed in linear initialization in 0.4.0.
    """
    weights, data = _case((40, 40), 500, 6, offset=offset)
    np.testing.assert_array_equal(
        bmu_indices(data, weights, euclidean_distance), _exact(data, weights)
    )


def test_the_uncentred_expansion_really_does_fail_there() -> None:
    """Show the defect the centring prevents, so the fix is not mistaken for a redundant line.

    Without this, someone reading ``flat - shift`` sees an operation with no visible effect and
    deletes it, and every test above still passes at the offsets they happen to try.
    """
    weights, data = _case((40, 40), 500, 6, offset=1e9)
    flat = weights.reshape(-1, weights.shape[-1])

    uncentred = (np.einsum("nf,nf->n", flat, flat)[None, :] - 2.0 * (data @ flat.T)).argmin(axis=1)

    wrong = int((uncentred != _exact(data, weights)).sum())
    assert wrong > len(data) // 2, (
        f"expected the uncentred expansion to fail badly at 1e9, got {wrong} of {len(data)}"
    )


# ---------------------------------------------------------------------------------------------
# It is still the Euclidean map
# ---------------------------------------------------------------------------------------------


def test_it_is_not_the_dot_product_map_of_section_4_5() -> None:
    """Kohonen Section 4.5 defines a different algorithm, and this is not it.

    Eq. (9), ``c = argmax_i dot(x, m_i)``, requires the models to be "kept normalized to constant
    length all the time" and selects a different node when they are not. A matrix product in the
    winner search reads exactly like a silent switch to it, so the difference is asserted.

    The models here have deliberately unequal lengths, which is what makes the two criteria diverge.
    """
    weights = np.array([[[1.0, 0.0], [10.0, 10.0]]])
    data = np.array([[1.0, 0.5]])

    flat = weights.reshape(-1, 2)
    assert int(np.argmax(flat @ data[0])) == 1, "the dot-product criterion prefers the long model"
    assert int(bmu_indices(data, weights, euclidean_distance)[0]) == 0
    assert winner(data[0], weights, euclidean_distance) == (0, 0)


# ---------------------------------------------------------------------------------------------
# A custom distance keeps the exact path
# ---------------------------------------------------------------------------------------------


def _manhattan(x: object, weights: object) -> np.ndarray:
    """Sum of absolute differences along the last axis.

    :param x: Input vector.
    :param weights: One model or an array of them.
    :return: Distances.
    """
    result: np.ndarray = np.abs(np.asarray(x) - np.asarray(weights)).sum(axis=-1)
    return result


def test_a_custom_distance_is_used_rather_than_the_fast_path() -> None:
    """The expansion is an identity for the Euclidean norm alone, so anything else takes the loop.

    Constructed so the two metrics disagree: under Manhattan the first model wins, under Euclidean
    the second does. If the fast path were taken regardless, this would return the Euclidean answer.
    """
    weights = np.array([[[0.9, 0.9], [0.0, 1.4]]])
    data = np.array([[0.0, 0.0]])

    assert int(bmu_indices(data, weights, _manhattan)[0]) == 1
    assert int(bmu_indices(data, weights, euclidean_distance)[0]) == 0


def test_both_paths_agree_when_the_custom_distance_is_euclidean() -> None:
    """A user-supplied function that happens to be Euclidean must give the same nodes.

    ``python_som.euclidean_distance`` is selected by identity, so passing an equivalent but distinct
    callable takes the slow path. The two must still agree.
    """
    weights, data = _case((10, 8), 120, 5)

    def same_but_not_identical(x: object, w: object) -> np.ndarray:
        """Euclidean distance, written out so it is not the registered function object."""
        result: np.ndarray = np.linalg.norm(np.asarray(x) - np.asarray(w), axis=-1)
        return result

    np.testing.assert_array_equal(
        bmu_indices(data, weights, same_but_not_identical),
        bmu_indices(data, weights, euclidean_distance),
    )


def test_accumulate_and_quantization_go_through_the_same_search() -> None:
    """Eq. (8)'s inputs and the reported error must agree with the nodes the search chose."""
    shape = (9, 7)
    weights, data = _case(shape, 150, 4)

    _, counts = accumulate(data, weights, shape, euclidean_distance)
    nodes = bmu_indices(data, weights, euclidean_distance)

    expected_counts = np.bincount(nodes, minlength=shape[0] * shape[1]).reshape(shape)
    np.testing.assert_array_equal(counts, expected_counts.astype(float))
    assert counts.sum() == len(data)

    flat = weights.reshape(-1, weights.shape[-1])
    errors = quantization(data, weights, euclidean_distance)
    for error, sample, node in zip(errors, data, nodes, strict=True):
        assert error == pytest.approx(float(np.linalg.norm(sample - flat[node])))


def test_quantization_error_is_unchanged_by_the_faster_search() -> None:
    """The reported number is a distance, not the search's score, which drops a constant term."""
    som = python_som.SOM(x=8, y=6, input_len=4, random_seed=SEED)
    rng = np.random.default_rng(SEED)
    data = rng.normal(size=(80, 4))
    som.weight_initialization(mode="random")

    flat = som.get_weights().reshape(-1, 4)
    expected = float(
        np.mean([np.linalg.norm(x - flat, axis=-1).min() for x in data]),
    )
    assert som.quantization_error(data) == pytest.approx(expected)
