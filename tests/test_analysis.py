"""Analysis and visualization helpers: winner, quantization, U-matrix and the maps."""

from __future__ import annotations

import numpy as np
import pytest

from python_som import TrainingMode, WeightInit, euclidean_distance
from tests.conftest import make_som


def test_activate_returns_one_distance_per_node(blobs: np.ndarray) -> None:
    som = make_som(x=7, y=4)
    assert som.activate(blobs[0]).shape == (7, 4)


def test_winner_is_the_argmin_of_activate(blobs: np.ndarray) -> None:
    som = make_som(x=7, y=4)
    for sample in blobs[:10]:
        activation = som.activate(sample)
        expected = np.unravel_index(activation.argmin(), activation.shape)
        assert som.winner(sample) == (int(expected[0]), int(expected[1]))


def test_quantization_returns_one_value_per_sample(blobs: np.ndarray) -> None:
    som = make_som(x=6, y=6)
    assert som.quantization(blobs).shape == (len(blobs),)


def test_quantization_error_is_the_mean_distance(blobs: np.ndarray) -> None:
    som = make_som(x=6, y=6)
    assert som.quantization_error(blobs) == pytest.approx(som.quantization(blobs).mean())


def test_quantization_error_is_zero_for_an_exact_fit() -> None:
    """A map whose models are exactly the data has no quantization error."""
    data = np.arange(12, dtype=float).reshape(4, 3)
    som = make_som(x=2, y=2, input_len=3)
    som.weight_initialization(mode=WeightInit.SAMPLE, data=data)
    som._weights = data.reshape(2, 2, 3).copy()
    assert som.quantization_error(data) == pytest.approx(0.0)


def test_activation_matrix_counts_every_sample(blobs: np.ndarray) -> None:
    som = make_som(x=6, y=5)
    assert som.activation_matrix(blobs).sum() == len(blobs)


def test_winner_map_partitions_the_dataset(blobs: np.ndarray) -> None:
    som = make_som(x=6, y=5)
    winner_map = som.winner_map(blobs)
    assert len(winner_map) == 30
    assert sum(len(v) for v in winner_map.values()) == len(blobs)


def test_winner_map_agrees_with_the_activation_matrix(blobs: np.ndarray) -> None:
    som = make_som(x=6, y=5)
    counts = som.activation_matrix(blobs)
    for node, members in som.winner_map(blobs).items():
        assert counts[node] == len(members)


def test_label_map_counts_labels_per_node(blobs: np.ndarray, blob_labels: np.ndarray) -> None:
    som = make_som(x=6, y=5)
    label_map = som.label_map(blobs, blob_labels)
    assert sum(sum(c.values()) for c in label_map.values()) == len(blobs)


def test_label_map_separates_well_separated_clusters(
    blobs: np.ndarray, blob_labels: np.ndarray
) -> None:
    """After training on three distant blobs, most nodes should be label-pure."""
    som = make_som(x=8, y=8, neighborhood_radius=2.0)
    som.weight_initialization(mode=WeightInit.LINEAR, data=blobs)
    som.train(blobs, n_iteration=40, mode=TrainingMode.BATCH)
    label_map = som.label_map(blobs, blob_labels)
    occupied = [c for c in label_map.values() if sum(c.values()) > 0]
    pure = [c for c in occupied if len(c) == 1]
    assert len(pure) / len(occupied) > 0.8


def test_label_map_rejects_mismatched_lengths(blobs: np.ndarray) -> None:
    som = make_som(x=4, y=4)
    with pytest.raises(ValueError, match="same length"):
        som.label_map(blobs, [0, 1, 2])


def test_distance_matrix_has_the_grid_shape() -> None:
    som = make_som(x=6, y=4)
    assert som.distance_matrix().shape == (6, 4)


def test_distance_matrix_is_non_negative() -> None:
    assert (make_som(x=6, y=4).distance_matrix() >= 0).all()


def test_distance_matrix_normalizes_to_the_unit_interval() -> None:
    um = make_som(x=6, y=4).distance_matrix(normalize=True)
    assert um.min() == pytest.approx(0.0)
    assert um.max() == pytest.approx(1.0)


def test_distance_matrix_normalization_survives_a_flat_map() -> None:
    """A map whose models are all identical has zero spread; normalizing must not divide by zero."""
    som = make_som(x=4, y=4, input_len=3)
    som._weights = np.zeros((4, 4, 3))
    um = som.distance_matrix(normalize=True)
    assert np.isfinite(um).all()


def test_distance_matrix_is_larger_at_a_cluster_boundary() -> None:
    """The U-matrix should peak where neighbouring models are far apart."""
    som = make_som(x=4, y=1, input_len=1)
    som._weights = np.array([[0.0], [0.0], [10.0], [10.0]]).reshape(4, 1, 1)
    um = som.distance_matrix()
    assert um[1, 0] > um[0, 0]
    assert um[2, 0] > um[3, 0]


def test_euclidean_distance_broadcasts() -> None:
    a = np.zeros(3)
    b = np.ones((4, 5, 3))
    assert euclidean_distance(a, b).shape == (4, 5)
    assert euclidean_distance(a, b)[0, 0] == pytest.approx(np.sqrt(3))


def test_euclidean_distance_is_zero_for_identical_vectors() -> None:
    v = np.array([1.0, -2.0, 3.5])
    assert euclidean_distance(v, v) == pytest.approx(0.0)


def test_euclidean_distance_is_symmetric() -> None:
    a, b = np.array([1.0, 2.0]), np.array([-3.0, 0.5])
    assert euclidean_distance(a, b) == pytest.approx(euclidean_distance(b, a))
