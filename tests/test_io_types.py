"""Parity across the accepted input types: ndarray, DataFrame and plain list."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.conftest import SEED, make_som


@pytest.fixture
def variants(blobs: np.ndarray) -> dict[str, Any]:
    """Return the same data as an array, a DataFrame and a nested list."""
    return {
        "ndarray": blobs,
        "dataframe": pd.DataFrame(blobs, columns=["a", "b", "c"]),
        "list": blobs.tolist(),
    }


def test_quantization_error_agrees_across_input_types(variants: dict[str, Any]) -> None:
    errors = [make_som(x=6, y=6).quantization_error(v) for v in variants.values()]
    assert errors[0] == pytest.approx(errors[1])
    assert errors[0] == pytest.approx(errors[2])


def test_training_agrees_across_input_types(variants: dict[str, Any]) -> None:
    weights = []
    for v in variants.values():
        som = make_som(x=6, y=6, random_seed=SEED)
        som.train(v, n_iteration=20, mode="batch")
        weights.append(som.get_weights())
    np.testing.assert_allclose(weights[0], weights[1])
    np.testing.assert_allclose(weights[0], weights[2])


def test_activation_matrix_agrees_across_input_types(variants: dict[str, Any]) -> None:
    matrices = [make_som(x=5, y=5).activation_matrix(v) for v in variants.values()]
    np.testing.assert_array_equal(matrices[0], matrices[1])
    np.testing.assert_array_equal(matrices[0], matrices[2])


def test_linear_init_agrees_across_input_types(variants: dict[str, Any]) -> None:
    weights = []
    for v in variants.values():
        som = make_som(x=5, y=5)
        som.weight_initialization(mode="linear", data=v)
        weights.append(som.get_weights())
    np.testing.assert_allclose(weights[0], weights[1])
    np.testing.assert_allclose(weights[0], weights[2])


def test_labels_accept_a_series(blobs: np.ndarray, blob_labels: np.ndarray) -> None:
    som = make_som(x=5, y=5)
    from_array = som.label_map(blobs, blob_labels)
    from_series = som.label_map(blobs, pd.Series(blob_labels))
    assert from_array == from_series


def test_labels_accept_strings(blobs: np.ndarray) -> None:
    labels = ["setosa"] * 20 + ["versicolor"] * 20 + ["virginica"] * 20
    label_map = make_som(x=5, y=5).label_map(blobs, labels)
    seen = {k for counter in label_map.values() for k in counter}
    assert seen <= {"setosa", "versicolor", "virginica"}


def test_a_single_feature_dataset_works() -> None:
    data = np.linspace(0, 1, 20).reshape(-1, 1)
    som = make_som(x=4, y=3, input_len=1)
    som.train(data, n_iteration=10, mode="batch")
    assert np.isfinite(som.get_weights()).all()


def test_a_single_sample_dataset_works() -> None:
    som = make_som(x=3, y=3, input_len=3)
    som.train(np.array([[1.0, 2.0, 3.0]]), n_iteration=3, mode="batch")
    assert np.isfinite(som.get_weights()).all()
