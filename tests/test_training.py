"""Training behaviour and the invariants that must hold for any input.

The invariants are the point of this module. The bugs found in this library were not wrong formulas
so much as properties nobody had asserted: that no model is silently destroyed, that a requested
iteration count is honoured, that a training step cannot divide by a vanishing denominator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:  # pragma: no cover
    from python_som._enums import TrainingModeStr

import python_som
from tests.conftest import SEED, make_som


def test_batch_never_zeroes_a_model(rng: np.random.Generator) -> None:
    """Regression for the batch update starting from a zeroed array.

    ``_train_batch`` used to build ``new_weights = np.zeros(...)``, fill only the nodes whose
    neighbourhood contained data, and assign the result. Every other model was destroyed: on this
    exact configuration, 282 of 900 models became the zero vector in a single step.
    """
    data = rng.normal(size=(20, 3))
    som = make_som(x=30, y=30, neighborhood_radius=0.5)
    som.weight_initialization(mode="sample", data=data)
    before = som.get_weights().copy()

    som.train(data, n_iteration=1, mode="batch")
    after = som.get_weights()

    zeroed = np.all(after == 0.0, axis=-1)
    assert not zeroed.any(), f"{zeroed.sum()} of {zeroed.size} models were wiped to zero"
    # untouched models keep their previous value exactly
    untouched = np.all(np.isclose(after, before), axis=-1)
    assert untouched.any(), "expected some models to be left alone at this radius"


def test_batch_preserves_finiteness(blobs: np.ndarray) -> None:
    som = make_som(x=10, y=10, neighborhood_radius=2.0)
    som.train(blobs, n_iteration=5, mode="batch")
    assert np.isfinite(som.get_weights()).all()


@pytest.mark.parametrize("mode", ["random", "sequential", "batch"])
def test_training_reduces_quantization_error(mode: TrainingModeStr, blobs: np.ndarray) -> None:
    """Training should fit the data better than the initial state."""
    som = make_som(x=8, y=8, neighborhood_radius=2.0, learning_rate=0.5)
    before = som.quantization_error(blobs)
    after = som.train(blobs, n_iteration=30, mode=mode)
    assert after < before


@pytest.mark.parametrize("mode", ["random", "sequential"])
def test_stepwise_runs_the_requested_number_of_iterations(
    mode: TrainingModeStr, blobs: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: sequential mode used to run ``len(data)`` steps regardless of ``n_iteration``.

    It iterated ``enumerate(data_array)`` directly, so asking for 500 iterations over 30 samples
    ran 30 steps while the decay functions still used 500 as their horizon.
    """
    som = make_som(x=6, y=6)
    steps = 0
    original = som._sigma

    def counting_sigma(t: int, n: int) -> float:
        nonlocal steps
        steps += 1
        return original(t, n)

    # _sigma is called exactly once per training step and nowhere else, so it counts iterations
    # without depending on how the winner lookup is routed internally.
    monkeypatch.setattr(som, "_sigma", counting_sigma)
    requested = 137
    som.train(blobs, n_iteration=requested, mode=mode)
    assert steps == requested


def test_sequential_cycles_through_the_dataset(blobs: np.ndarray) -> None:
    """More iterations than samples must wrap around rather than stop early."""
    som = make_som(x=5, y=5)
    som.train(blobs, n_iteration=len(blobs) * 3, mode="sequential")
    assert np.isfinite(som.get_weights()).all()


def test_random_samples_with_replacement(
    blobs: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the sampling semantics, which changed in 0.3.0 and are easy to alter by accident.

    ``random`` mode draws i.i.d. with replacement, the Robbins-Monro stochastic approximation
    Kohonen cites in Section 4.1. Before 0.3.0 the draw used
    ``replace=(n_iteration > len(data))``, so asking for fewer iterations than samples produced a
    permutation instead, making the character of the sampling depend on the iteration count.

    With replacement, drawing as many indices as there are samples repeats at least one of them
    with overwhelming probability: for n = 60 the chance of a permutation is 60!/60**60, far below
    any floating-point threshold.
    """
    seen: list[int] = []
    som = make_som(x=5, y=5)
    original = som.winner

    def recording_winner(x: np.ndarray) -> tuple[int, int]:
        seen.append(int(np.flatnonzero((blobs == x).all(axis=1))[0]))
        return original(x)

    monkeypatch.setattr(som, "winner", recording_winner)
    som.train(blobs, n_iteration=len(blobs), mode="random")
    drawn = seen[: len(blobs)]
    assert len(drawn) == len(blobs)
    assert len(set(drawn)) < len(blobs), "a with-replacement draw should repeat an index"


def test_sequential_visits_every_sample_exactly_once_per_epoch(
    blobs: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sequential is the counterpart: one full pass covers every sample once, in order."""
    seen: list[int] = []
    som = make_som(x=5, y=5)
    original = som.winner

    def recording_winner(x: np.ndarray) -> tuple[int, int]:
        seen.append(int(np.flatnonzero((blobs == x).all(axis=1))[0]))
        return original(x)

    monkeypatch.setattr(som, "winner", recording_winner)
    som.train(blobs, n_iteration=len(blobs), mode="sequential")
    assert seen[: len(blobs)] == list(range(len(blobs)))


def test_batch_rejects_a_signed_neighborhood(blobs: np.ndarray) -> None:
    """Kohonen Eq. (8) divides by ``sum_j n_j h_ji``, which is not sign-definite for a signed h.

    Measured on a 12x12 grid, 49 of 144 denominators come out negative.
    """
    som = make_som(x=12, y=12, neighborhood_function="mexicanhat")
    with pytest.raises(ValueError, match="batch"):
        som.train(blobs, n_iteration=1, mode="batch")


def test_batch_rejects_the_alias_too(blobs: np.ndarray) -> None:
    som = make_som(x=12, y=12, neighborhood_function="mexican_hat")
    with pytest.raises(ValueError, match="batch"):
        som.train(blobs, n_iteration=1, mode="batch")


@pytest.mark.parametrize("mode", ["random", "sequential"])
def test_stepwise_accepts_a_signed_neighborhood(mode: TrainingModeStr, blobs: np.ndarray) -> None:
    """The mexican hat is fine for stepwise training; only the batch mean is undefined."""
    som = make_som(x=12, y=12, neighborhood_function="mexicanhat")
    som.train(blobs, n_iteration=30, mode=mode)
    assert np.isfinite(som.get_weights()).all()


@pytest.mark.parametrize(("mode", "per_sample"), [("batch", 10), ("sequential", 1000)])
def test_omitting_n_iteration_uses_the_documented_default(
    mode: TrainingModeStr, per_sample: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default is 1000 iterations per sample for stepwise modes and 10 for batch.

    That default is documented on ``train`` but was never exercised, so the arithmetic behind it
    was untested. A two-sample dataset keeps even the 1000-per-sample case quick.
    """
    data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    som = make_som(x=3, y=3)
    steps = 0
    original = som._sigma

    def counting_sigma(t: int, n: int) -> float:
        nonlocal steps
        steps += 1
        return original(t, n)

    monkeypatch.setattr(som, "_sigma", counting_sigma)
    som.train(data, mode=mode)
    assert steps == per_sample * len(data)


def test_verbose_training_runs_with_a_progress_bar(blobs: np.ndarray) -> None:
    """The tqdm branch of the progress wrapper, which no other test reaches.

    tqdm ships in the dev extra, so this exercises the wrapped path rather than the fallback.
    """
    som = make_som(x=5, y=5)
    error = som.train(blobs, n_iteration=3, mode="batch", verbose=True)
    assert np.isfinite(error)


def test_unknown_mode_raises_value_error() -> None:
    som = make_som(x=5, y=5)
    with pytest.raises(ValueError, match="mode"):
        # Deliberately invalid: the Literal annotation is what stops this reaching a caller.
        som.train(np.zeros((4, 3)), n_iteration=1, mode="stochastic")  # type: ignore[arg-type]


def test_empty_dataset_raises_value_error() -> None:
    som = make_som(x=5, y=5)
    with pytest.raises(ValueError, match="empty"):
        som.train(np.zeros((0, 3)), n_iteration=1)


def test_non_positive_iteration_count_raises_value_error(blobs: np.ndarray) -> None:
    som = make_som(x=5, y=5)
    with pytest.raises(ValueError, match="n_iteration"):
        som.train(blobs, n_iteration=0)


def test_same_seed_reproduces_the_same_map(blobs: np.ndarray) -> None:
    a = make_som(x=8, y=8, random_seed=SEED)
    b = make_som(x=8, y=8, random_seed=SEED)
    a.train(blobs, n_iteration=25, mode="random")
    b.train(blobs, n_iteration=25, mode="random")
    np.testing.assert_array_equal(a.get_weights(), b.get_weights())


def test_different_seeds_give_different_maps(blobs: np.ndarray) -> None:
    a = make_som(x=8, y=8, random_seed=1)
    b = make_som(x=8, y=8, random_seed=2)
    a.train(blobs, n_iteration=25, mode="random")
    b.train(blobs, n_iteration=25, mode="random")
    assert not np.allclose(a.get_weights(), b.get_weights())


def test_construction_does_not_disturb_the_global_numpy_rng() -> None:
    """Regression: the constructor used to call ``np.random.seed`` on the global generator.

    Building a SOM would silently reseed NumPy for the whole host program.
    """
    np.random.seed(12345)  # noqa: NPY002
    expected = np.random.random(5)  # noqa: NPY002

    np.random.seed(12345)  # noqa: NPY002
    make_som(x=6, y=6, random_seed=None)
    actual = np.random.random(5)  # noqa: NPY002

    np.testing.assert_array_equal(expected, actual)


def test_radius_is_floored_during_training(blobs: np.ndarray) -> None:
    """Floor the radius so it never reaches zero during training.

    Kohonen Section 4.2: sigma "shall not go to zero, because otherwise the process loses its
    ordering power".

    ``linear_decay`` reaches zero at the horizon, which would be a division by zero in the
    gaussian. The floor keeps it defined.
    """
    som = make_som(
        x=6,
        y=6,
        neighborhood_radius_decay=python_som.linear_decay,
        min_neighborhood_radius=0.5,
    )
    som.train(blobs, n_iteration=10, mode="random")
    assert np.isfinite(som.get_weights()).all()


def test_invalid_minimum_radius_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_neighborhood_radius"):
        make_som(x=5, y=5, min_neighborhood_radius=0.0)


def test_winner_is_always_inside_the_grid(blobs: np.ndarray) -> None:
    som = make_som(x=7, y=4)
    for sample in blobs:
        i, j = som.winner(sample)
        assert 0 <= i < 7
        assert 0 <= j < 4


def test_batch_matches_a_reference_implementation(blobs: np.ndarray) -> None:
    """The contracted batch update must agree with the literal double loop of Eq. (8)."""
    som = make_som(x=6, y=5, neighborhood_radius=1.5)
    som.weight_initialization(mode="sample", data=blobs)
    reference = som.get_weights().copy()

    sigma = som._sigma(0, 1)
    winner_map = som.winner_map(blobs)
    expected = reference.copy()
    for i in np.ndindex(som.get_shape()):
        neig = som.neighborhood((int(i[0]), int(i[1])), sigma)
        upper = np.zeros(3)
        bottom = 0.0
        for j, members in winner_map.items():
            upper += neig[j] * np.sum(members, axis=0) if members else 0.0
            bottom += neig[j] * len(members)
        if abs(bottom) > 1e-12:
            expected[i] = upper / bottom

    som.train(blobs, n_iteration=1, mode="batch")
    np.testing.assert_allclose(som.get_weights(), expected, atol=1e-12)
