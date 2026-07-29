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

__all__ = [
    "asymptotic_decay",
    "exponential_decay",
    "inverse_decay",
    "linear_decay",
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
