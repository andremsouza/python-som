"""The enums, the strategy protocols, and learning-rate validation.

Three additions in 0.4.0 that share one property: every call that worked before still works. The
enums are ``str`` subclasses, the protocols are structural, and the only behaviour that changes is
that a learning rate which could never have trained is now rejected instead of silently accepted.
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

import python_som
from python_som import (
    DecayFunction,
    DistanceFunction,
    Neighborhood,
    NeighborhoodFunction,
    SampleMode,
    TrainingMode,
    WeightInit,
)
from python_som._core._neighborhood import NEIGHBORHOOD_FUNCTIONS, bubble, gaussian
from python_som._som import INITIALIZATION_MODES, TRAINING_MODES
from tests.conftest import make_som

if TYPE_CHECKING:  # pragma: no cover
    from enum import Enum

ALL_ENUMS = [TrainingMode, Neighborhood, WeightInit, SampleMode]


# ---------------------------------------------------------------------------------------------
# The enums behave as strings, on every supported Python version
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("enum", ALL_ENUMS)
def test_every_member_is_a_string_equal_to_its_value(enum: type[Enum]) -> None:
    """The property that makes this additive rather than breaking.

    Existing code compares these against plain strings and uses them as dict keys; all of that has
    to keep working, which it does only because each member *is* a ``str``.
    """
    for member in enum:
        assert isinstance(member, str)
        assert member == member.value
        assert hash(member) == hash(member.value)
        assert {member.value: "found"}[member] == "found"


@pytest.mark.parametrize("enum", ALL_ENUMS)
def test_every_member_renders_as_its_value(enum: type[Enum]) -> None:
    """``enum.StrEnum`` is 3.11+, so the base class here is a shim; this is what it must reproduce.

    A bare ``class X(str, Enum)`` renders as ``'X.MEMBER'``, which would silently put the wrong text
    into an f-string, a filename or a log line. Checked for ``str``, f-strings, ``format`` and JSON,
    because they take different paths through the type.
    """
    for member in enum:
        assert str(member) == member.value
        assert f"{member}" == member.value
        assert format(member) == member.value
        assert json.loads(json.dumps({"m": member}))["m"] == member.value


def test_the_enums_cover_exactly_the_accepted_values() -> None:
    """The enums and the runtime validation must not drift apart.

    ``TRAINING_MODES`` and ``INITIALIZATION_MODES`` are derived from the enums, so this is really
    checking that the derivation is still wired up rather than hand-maintained again.
    """
    assert set(TRAINING_MODES) == {m.value for m in TrainingMode}
    assert set(INITIALIZATION_MODES) == {m.value for m in WeightInit}


def test_every_neighborhood_member_resolves() -> None:
    """The registry is keyed by string, so a member has to be a valid key.

    ``mexicanhat`` is deliberately absent from the enum -- one canonical spelling per option -- but
    must still resolve as a plain string, which the second assertion covers.
    """
    for member in Neighborhood:
        assert member.value in NEIGHBORHOOD_FUNCTIONS
    assert "mexicanhat" in NEIGHBORHOOD_FUNCTIONS, "the legacy spelling must keep working"


# ---------------------------------------------------------------------------------------------
# Enums and strings are interchangeable at every call site that takes one
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("as_enum", "as_string"),
    [
        (TrainingMode.RANDOM, "random"),
        (TrainingMode.SEQUENTIAL, "sequential"),
        (TrainingMode.BATCH, "batch"),
    ],
)
def test_training_accepts_either_spelling_with_identical_results(
    as_enum: TrainingMode, as_string: str, blobs: np.ndarray
) -> None:
    """Not merely accepted -- identical, since the enum member is the string it wraps."""
    first = make_som(x=6, y=5)
    second = make_som(x=6, y=5)
    error_enum = first.train(blobs, n_iteration=15, mode=as_enum)
    error_string = second.train(blobs, n_iteration=15, mode=as_string)  # type: ignore[arg-type]

    assert error_enum == error_string
    np.testing.assert_array_equal(first.get_weights(), second.get_weights())


@pytest.mark.parametrize("member", list(WeightInit))
def test_weight_initialization_accepts_either_spelling(
    member: WeightInit, blobs: np.ndarray
) -> None:
    som = make_som(x=5, y=4)
    kwargs = {} if member is WeightInit.RANDOM else {"data": blobs}
    som.weight_initialization(mode=member, **kwargs)
    from_enum = som.get_weights().copy()

    other = make_som(x=5, y=4)
    other.weight_initialization(mode=member.value, **kwargs)
    np.testing.assert_array_equal(from_enum, other.get_weights())


@pytest.mark.parametrize("member", list(Neighborhood))
def test_the_constructor_accepts_either_spelling(member: Neighborhood) -> None:
    from_enum = python_som.SOM(x=4, y=4, input_len=3, neighborhood_function=member, random_seed=1)
    from_string = python_som.SOM(
        x=4,
        y=4,
        input_len=3,
        neighborhood_function=member.value,
        random_seed=1,
    )
    np.testing.assert_array_equal(
        from_enum.neighborhood((2, 2), 1.5), from_string.neighborhood((2, 2), 1.5)
    )


def test_sample_mode_accepts_either_spelling() -> None:
    for mode in SampleMode:
        first = make_som(x=4, y=4)
        first.weight_initialization(mode=WeightInit.RANDOM, sample_mode=mode)
        second = make_som(x=4, y=4)
        second.weight_initialization(mode=WeightInit.RANDOM, sample_mode=mode.value)
        np.testing.assert_array_equal(first.get_weights(), second.get_weights())


# ---------------------------------------------------------------------------------------------
# The strategy protocols are satisfied by what already exists
# ---------------------------------------------------------------------------------------------


def test_the_shipped_functions_satisfy_their_protocols() -> None:
    """Structural, so this is a statement about shape rather than inheritance.

    ``runtime_checkable`` only verifies that ``__call__`` exists, not its signature -- that is
    mypy's job, and the suite type-checks under ``--strict``. This catches the coarser mistake of a
    protocol that nothing at all satisfies.
    """
    assert isinstance(gaussian, NeighborhoodFunction)
    assert isinstance(bubble, NeighborhoodFunction)
    assert isinstance(python_som.asymptotic_decay, DecayFunction)
    assert isinstance(python_som.linear_decay, DecayFunction)
    assert isinstance(python_som.euclidean_distance, DistanceFunction)


def test_a_user_supplied_strategy_still_works() -> None:
    """The protocols add checking, not a requirement to inherit from anything.

    Parameter names deliberately differ from the protocol's, which is why each protocol declares
    its parameters positional-only. Without that, this callable would fail type checking purely for
    naming its arguments differently.
    """

    def my_decay(start: float, at: int, out_of: int) -> float:
        return float(start) * (1.0 - at / out_of)

    som = python_som.SOM(
        x=5,
        y=5,
        input_len=3,
        learning_rate_decay=my_decay,
        neighborhood_radius_decay=my_decay,
        random_seed=3,
    )
    error = som.train(np.random.default_rng(0).normal(size=(40, 3)), n_iteration=10)
    assert np.isfinite(error)


# ---------------------------------------------------------------------------------------------
# learning_rate validation
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.0, -0.5, -1.0, float("nan"), float("-inf"), float("inf")])
def test_a_rate_that_cannot_train_is_rejected(rate: float) -> None:
    """Silently accepted through 0.3.0.

    ``0`` freezes every model; ``-1`` drives them away from the data, taking the quantization error
    from 0.0 to 11.7. Neither can be intended, and neither announced itself.

    ``nan`` needs the explicit ``isfinite`` check: ``nan <= 0`` is ``False``, so a bare comparison
    lets it through and every weight becomes ``nan`` on the first step.
    """
    with pytest.raises(ValueError, match="'learning_rate' must be a finite positive number"):
        python_som.SOM(x=4, y=4, input_len=3, learning_rate=rate)


@pytest.mark.parametrize("rate", [1.0, 0.5, 0.001])
def test_a_plausible_rate_is_silent(rate: float) -> None:
    """Including exactly 1.0, which is the boundary and is not warned about."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        python_som.SOM(x=4, y=4, input_len=3, learning_rate=rate)


@pytest.mark.parametrize("rate", [1.0001, 2.0, 5.0])
def test_a_rate_above_one_warns_but_is_accepted(rate: float) -> None:
    """Warned rather than rejected, because it is unwise rather than impossible.

    Kohonen gives no upper bound, and at ``alpha = 5`` with decay disabled the largest weight stayed
    at 3.61 -- the neighborhood damps the correction away from the winner. Rejecting it would invent
    a limit the sources do not give.
    """
    with pytest.warns(UserWarning, match="above 1"):
        som = python_som.SOM(x=4, y=4, input_len=3, learning_rate=rate, random_seed=2)
    assert som.get_shape() == (4, 4)


def test_the_warning_points_at_the_caller() -> None:
    """``stacklevel`` must name the user's line, not a line inside this package.

    A warning that blames ``_som.py`` tells the reader nothing about which of their own constructor
    calls to fix.
    """
    with pytest.warns(UserWarning, match="above 1") as caught:
        python_som.SOM(x=4, y=4, input_len=3, learning_rate=3.0)
    assert caught[0].filename == __file__, f"warning blamed {caught[0].filename}"
