"""End-to-end runs, marked ``slow``.

These exist because every other test in the suite checks one property in isolation. A map can pass
all of them and still fail to do the thing the library is for: take a dataset with structure in it
and lay that structure out on the grid. Deselect with ``-m 'not slow'``.

Nothing here touches the network. The example script uses seaborn's Iris loader, which downloads,
so the same code path is exercised over synthetic clusters instead.
"""

from __future__ import annotations

import numpy as np
import pytest

import python_som
from tests.conftest import SEED

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def clusters() -> tuple[np.ndarray, np.ndarray]:
    """Four well-separated clusters in 5-D, 200 samples, with their labels."""
    rng = np.random.default_rng(SEED)
    centres = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 10.0, 0.0],
            [-10.0, 0.0, 0.0, 10.0, 10.0],
        ]
    )
    data = np.concatenate([c + rng.normal(scale=0.5, size=(50, 5)) for c in centres])
    labels = np.repeat(np.arange(4), 50)
    return data, labels


def test_batch_training_separates_known_clusters(
    clusters: tuple[np.ndarray, np.ndarray],
) -> None:
    """The whole point of the library, asserted once: structure in, structure on the grid."""
    data, labels = clusters
    som = python_som.SOM(x=12, y=12, input_len=5, neighborhood_radius=4.0, random_seed=SEED)
    som.weight_initialization(mode="linear", data=data)
    error = som.train(data, n_iteration=40, mode="batch")

    # every cluster claims at least one node, and almost every occupied node is label-pure
    label_map = som.label_map(data, labels)
    occupied = [c for c in label_map.values() if sum(c.values()) > 0]
    pure = [c for c in occupied if len(c) == 1]
    claimed = {next(iter(c)) for c in pure}

    assert len(claimed) == 4, f"expected all four clusters represented, got {claimed}"
    assert len(pure) / len(occupied) > 0.9
    assert error < 1.0


def test_the_u_matrix_shows_the_cluster_boundaries(
    clusters: tuple[np.ndarray, np.ndarray],
) -> None:
    """A trained map's U-matrix must have more contrast than an untrained one.

    If training did nothing, the U-matrix of random models would be just as structured. This is the
    cheapest available check that ``distance_matrix`` reports something meaningful.
    """
    data, _ = clusters
    untrained = python_som.SOM(x=12, y=12, input_len=5, random_seed=SEED)
    trained = python_som.SOM(x=12, y=12, input_len=5, neighborhood_radius=4.0, random_seed=SEED)
    trained.weight_initialization(mode="linear", data=data)
    trained.train(data, n_iteration=40, mode="batch")

    assert np.ptp(trained.distance_matrix()) > np.ptp(untrained.distance_matrix())


@pytest.mark.parametrize("mode", ["random", "sequential", "batch"])
def test_every_mode_reaches_a_usable_map(
    mode: str, clusters: tuple[np.ndarray, np.ndarray]
) -> None:
    """All three training modes must produce finite models and reduce the error."""
    data, _ = clusters
    som = python_som.SOM(x=10, y=10, input_len=5, neighborhood_radius=3.0, random_seed=SEED)
    som.weight_initialization(mode="linear", data=data)
    before = som.quantization_error(data)
    after = som.train(data, n_iteration=60, mode=mode)

    assert np.isfinite(som.get_weights()).all()
    assert after < before


def test_a_toroidal_map_has_no_edge(clusters: tuple[np.ndarray, np.ndarray]) -> None:
    """On a torus every node is equivalent, so no row should be systematically starved of data.

    This is the end-to-end consequence of the cyclic fold-back fix: with the old one-sided fold,
    nodes in the upper half of an axis had a truncated neighbourhood and attracted fewer samples.
    """
    data, _ = clusters
    som = python_som.SOM(
        x=8,
        y=8,
        input_len=5,
        neighborhood_radius=3.0,
        cyclic_x=True,
        cyclic_y=True,
        random_seed=SEED,
    )
    som.weight_initialization(mode="linear", data=data)
    som.train(data, n_iteration=60, mode="batch")

    counts = som.activation_matrix(data)
    top_half = counts[:4].sum()
    bottom_half = counts[4:].sum()
    assert top_half > 0
    assert bottom_half > 0
