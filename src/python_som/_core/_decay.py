"""Decay functions for the learning rate and the neighborhood radius.

Each function maps an initial value, the current iteration and the total number of iterations to the
current value. They share one signature so that any of them, or a user-supplied equivalent, can be
passed as ``learning_rate_decay`` or ``neighborhood_radius_decay``.

Kohonen (2013) does not prescribe a particular form: "The true mathematical form of sigma(t) is not
crucial, as long as its value is fairly large in the beginning of the process, say, on the order of
half of the diameter of the grid, whereafter it is gradually reduced to a fraction of it in about
1000 steps" (Section 4.1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover
    from ._protocols import DecayFunction

__all__ = [
    "DECAY_FUNCTIONS",
    "asymptotic_decay",
    "exponential_decay",
    "inverse_decay",
    "linear_decay",
    "resolve_decay",
]


def asymptotic_decay(x: float, t: int, max_t: int) -> float:
    """Decay ``x`` hyperbolically, reaching ``x / 2`` at the halfway point.

    Never reaches zero, which suits Kohonen's warning that the neighborhood radius should not decay
    all the way down (Section 4.2).

    :param x: Initial value.
    :param t: Current iteration.
    :param max_t: Total number of iterations.
    :return: Value of ``x`` after ``t`` iterations.
    """
    return x / (1 + t / (max_t / 2))


def linear_decay(x: float, t: int, max_t: int) -> float:
    """Decay ``x`` linearly to zero at ``t == max_t``.

    :param x: Initial value.
    :param t: Current iteration.
    :param max_t: Total number of iterations.
    :return: Value of ``x`` after ``t`` iterations.
    """
    return x * (1.0 - t / max_t)


def exponential_decay(x: float, t: int, max_t: int, factor: float = 2.0) -> float:
    """Decay ``x`` geometrically by ``factor / max_t`` per iteration.

    :param x: Initial value.
    :param t: Current iteration.
    :param max_t: Total number of iterations.
    :param factor: Decay factor. Defaults to 2.0.
    :return: Value of ``x`` after ``t`` iterations.
    """
    return x * (1 - (factor / max_t)) ** t


def inverse_decay(x: float, t: int, max_t: int) -> float:
    """Decay ``x`` inversely with ``t``, scaled so the shape is independent of ``max_t``.

    :param x: Initial value.
    :param t: Current iteration.
    :param max_t: Total number of iterations.
    :return: Value of ``x`` after ``t`` iterations.
    """
    return (max_t / 100) * x / ((max_t / 100) + t)


DECAY_FUNCTIONS: Final[dict[str, DecayFunction]] = {
    "asymptotic_decay": asymptotic_decay,
    "linear_decay": linear_decay,
    "exponential_decay": exponential_decay,
    "inverse_decay": inverse_decay,
}
"""Decay functions by name, so a saved map can name the one it used.

Each key is the function's own name, which is the least surprising mapping and the one a reader can
check against the source without a lookup table. These names are written into artifacts, so they are
public API from 0.4.0 and fixed at 1.0.0.

A decay function is not required to be in here: a caller may pass any callable. What a name buys is
the ability to restore it from a file, and the loader says so explicitly when it cannot.
"""


def resolve_decay(name: str) -> DecayFunction:
    """Look up a decay function by name.

    :param name: Name of the decay function.
    :return: The corresponding function.
    :raises ValueError: If the name is not recognised.
    """
    try:
        return DECAY_FUNCTIONS[name]
    except KeyError as exc:
        valid = sorted(DECAY_FUNCTIONS)
        msg = f"Unknown decay function {name!r}. Value should be one of {valid}"
        raise ValueError(msg) from exc
