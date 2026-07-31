"""The optional numba kernel must select exactly the nodes the NumPy path selects.

``pip install numba`` swaps a compiled kernel into the best-matching-unit search. It is a second
implementation of the hottest code in the package, and the whole reason that is acceptable is that
the NumPy path stays the reference and this file asserts the two agree.

Agreement here is **identical indices**, not close ones. Both compute
``||w||^2 - 2 x.w`` over the same centred arrays, so there is no reason for them to differ, and a
tolerance would hide the case where one of them is wrong.

Skipped wholesale without numba. The CI job that installs it is what stops this file silently
skipping everywhere, which is the failure mode a guarded test file has.
"""

from __future__ import annotations

import numpy as np
import pytest

import python_som
import python_som._som
from python_som._accelerate import bmu_kernel
from python_som._core._distance import euclidean_distance
from python_som._core._match import accumulate, bmu_indices

BMU_KERNEL = bmu_kernel()

pytestmark = pytest.mark.skipif(BMU_KERNEL is None, reason="needs numba")

#: Fixed so a failure is reproducible.
SEED = 20260730


def _case(
    shape: tuple[int, int], n_samples: int, n_features: int, offset: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Build models and a dataset, optionally far from the origin.

    :param shape: Grid shape.
    :param n_samples: Number of samples.
    :param n_features: Number of features.
    :param offset: Constant added to both.
    :return: Models and dataset.
    """
    rng = np.random.default_rng(SEED)
    return (
        rng.normal(size=(*shape, n_features)) + offset,
        rng.normal(size=(n_samples, n_features)) + offset,
    )


@pytest.mark.parametrize(
    ("shape", "n_samples", "n_features"),
    [((5, 5), 40, 3), ((20, 20), 200, 4), ((40, 30), 500, 8), ((1, 9), 30, 2), ((60, 60), 300, 12)],
)
def test_the_kernel_selects_the_same_nodes_as_numpy(
    shape: tuple[int, int], n_samples: int, n_features: int
) -> None:
    """The claim the extra rests on."""
    weights, data = _case(shape, n_samples, n_features)
    np.testing.assert_array_equal(
        bmu_indices(data, weights, euclidean_distance, BMU_KERNEL),
        bmu_indices(data, weights, euclidean_distance),
    )


@pytest.mark.parametrize("offset", [0.0, 1e6, 1e9, 1e12])
def test_the_kernel_is_exact_far_from_the_origin(offset: float) -> None:
    """The centring happens before the kernel is called, so it must inherit the fix.

    The kernel receives arrays that are already shifted. If it were ever changed to take raw models
    and centre them itself, or not to centre at all, this is what would catch it.
    """
    weights, data = _case((40, 40), 300, 6, offset=offset)
    flat = weights.reshape(-1, 6)
    exact = np.array([np.linalg.norm(x - flat, axis=-1).argmin() for x in data])
    np.testing.assert_array_equal(bmu_indices(data, weights, euclidean_distance, BMU_KERNEL), exact)


def test_the_kernel_breaks_ties_to_the_lowest_index() -> None:
    """``argmin`` keeps the first minimum, and a ``<`` comparison in the kernel must match it.

    A ``<=`` in the inner loop would keep the *last* tied node instead, which no other test here
    would notice.
    """
    weights = np.full((3, 3, 2), 10.0)
    weights[0, 2] = [1.0, 0.0]
    weights[2, 0] = [1.0, 0.0]
    data = np.array([[1.0, 0.0]])
    assert int(bmu_indices(data, weights, euclidean_distance, BMU_KERNEL)[0]) == 2


def test_accumulate_agrees_through_the_kernel() -> None:
    """Eq. (8)'s inputs must not depend on which search produced the nodes."""
    shape = (12, 9)
    weights, data = _case(shape, 200, 5)

    fast_sums, fast_counts = accumulate(data, weights, shape, euclidean_distance, BMU_KERNEL)
    slow_sums, slow_counts = accumulate(data, weights, shape, euclidean_distance)

    np.testing.assert_array_equal(fast_counts, slow_counts)
    np.testing.assert_array_equal(fast_sums, slow_sums)


def test_training_agrees_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whole run, so any per-iteration divergence accumulates into view.

    Bit-identical: the two searches pick the same nodes, so Eq. (8) receives the same inputs and the
    arithmetic after that point is the same code.

    The NumPy arm is produced by patching the resolver rather than by uninstalling the extra.
    """
    shape, n_iteration = (20, 16), 25
    rng = np.random.default_rng(SEED)
    data = rng.normal(size=(300, 6))
    initial = rng.normal(size=(*shape, 6))

    def train() -> np.ndarray:
        """Train one map with whichever backend ``_som.BMU_KERNEL`` currently names.

        :return: The trained models.
        """
        som = python_som.SOM(
            x=shape[0], y=shape[1], input_len=6, neighborhood_radius=3.0, random_seed=SEED
        )
        som._weights = initial.copy()
        som.train(data, n_iteration=n_iteration, mode="batch")
        weights: np.ndarray = som.get_weights()
        return weights

    accelerated = train()
    monkeypatch.setattr(python_som._som, "bmu_kernel", lambda: None)
    np.testing.assert_array_equal(accelerated, train())


def test_a_custom_distance_still_bypasses_the_kernel() -> None:
    """The kernel computes a Euclidean criterion, so it must not be reached for anything else."""

    def manhattan(x: object, weights: object) -> np.ndarray:
        """Sum of absolute differences along the last axis.

        :param x: Input vector.
        :param weights: One model or an array of them.
        :return: Distances.
        """
        result: np.ndarray = np.abs(np.asarray(x) - np.asarray(weights)).sum(axis=-1)
        return result

    weights = np.array([[[0.9, 0.9], [0.0, 1.4]]])
    data = np.array([[0.0, 0.0]])
    assert int(bmu_indices(data, weights, manhattan, BMU_KERNEL)[0]) == 1


def test_importing_the_package_does_not_import_numba() -> None:
    """Installing numba must not add 104 ms to every ``import python_som``.

    numba is deferred to the first call that needs it, where the JIT compile is paid anyway. A
    module-level import in ``_accelerate`` would be invisible in every other test here and would
    quietly undo part of what numba buys.

    A subprocess, because numba is certainly already imported in this one.
    """
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    code = (
        "import sys, python_som; "
        "assert 'numba' not in sys.modules, sorted(m for m in sys.modules if 'numba' in m)[:3]; "
        "import numpy as np; "
        "som = python_som.SOM(x=4, y=4, input_len=2, random_seed=0); "
        "som.train(np.zeros((5, 2)), n_iteration=1, mode='batch'); "
        "assert 'numba' in sys.modules, 'the kernel should have loaded by now'; "
        "print('clean')"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout
