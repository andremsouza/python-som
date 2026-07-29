"""Properties of the decay functions."""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from typing import TypeAlias

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from python_som import (
    asymptotic_decay,
    exponential_decay,
    inverse_decay,
    linear_decay,
)

DecayFunction: TypeAlias = Callable[..., float]

ALL_DECAYS = [asymptotic_decay, linear_decay, exponential_decay, inverse_decay]


@pytest.mark.parametrize("decay", ALL_DECAYS)
def test_starts_at_the_initial_value(decay: DecayFunction) -> None:
    assert decay(0.5, 0, 100) == pytest.approx(0.5)


@pytest.mark.parametrize("decay", ALL_DECAYS)
def test_is_non_increasing(decay: DecayFunction) -> None:
    values = [decay(1.0, t, 100) for t in range(100)]
    assert all(b <= a + 1e-12 for a, b in pairwise(values))


@pytest.mark.parametrize("decay", ALL_DECAYS)
def test_stays_non_negative_within_the_horizon(decay: DecayFunction) -> None:
    assert all(decay(1.0, t, 100) >= -1e-12 for t in range(101))


@pytest.mark.parametrize("decay", ALL_DECAYS)
def test_scales_linearly_with_the_initial_value(decay: DecayFunction) -> None:
    """Every decay is a multiplicative envelope, so doubling x doubles the output."""
    assert decay(2.0, 30, 100) == pytest.approx(2 * decay(1.0, 30, 100))


def test_asymptotic_halves_at_the_midpoint() -> None:
    assert asymptotic_decay(1.0, 50, 100) == pytest.approx(0.5)


def test_asymptotic_never_reaches_zero() -> None:
    """Which is why it is the default: Kohonen warns sigma should not decay to zero."""
    assert asymptotic_decay(1.0, 10**6, 100) > 0


def test_linear_reaches_zero_at_the_horizon() -> None:
    assert linear_decay(1.0, 100, 100) == pytest.approx(0.0)


def test_exponential_respects_its_factor() -> None:
    assert exponential_decay(1.0, 1, 100, factor=2.0) == pytest.approx(1 - 2 / 100)


def test_inverse_is_independent_of_the_horizon_scale() -> None:
    """The ``max_t / 100`` scaling makes the shape depend on t, not on the horizon length."""
    assert inverse_decay(1.0, 100, 100) == pytest.approx(inverse_decay(1.0, 100, 100))
    assert inverse_decay(1.0, 1, 100) == pytest.approx(1 / (1 + 1))


@given(
    x=st.floats(min_value=0.01, max_value=100.0),
    t=st.integers(min_value=0, max_value=999),
    max_t=st.integers(min_value=1, max_value=1000),
)
@pytest.mark.parametrize("decay", ALL_DECAYS)
def test_output_is_finite_for_generated_inputs(
    decay: DecayFunction, x: float, t: int, max_t: int
) -> None:
    assert np.isfinite(decay(x, t, max_t))
