"""The two update rules, as pure functions returning new models.

Both return a new array rather than mutating their argument, which is what lets them be tested
without constructing a :class:`~python_som.SOM`, and what 1.0.0's ``fit`` will assign to
``weights_``.

That choice costs a little speed rather than gaining it, which is worth stating plainly because an
earlier draft of this module claimed the opposite. Measured against the in-place
``weights += alpha * h[..., None] * (sample - weights)`` that 0.3.0 shipped, with the two arms
interleaved and compared on medians: the pure form is **roughly 10% slower on small maps** (20x20,
50x50) and **indistinguishable on large ones** (100x100 and up, where the interquartile ranges
overlap). It is never faster. On a 20x20 map the penalty is single-digit milliseconds across a
10,000-iteration run, which is not a reason to give up a function that can be tested without
constructing a network.

The claim it replaces was that the pure form ran up to 2.9x *faster*. That came from a benchmark
whose two arms did not compute quite the same thing and whose repeats were not interleaved, so
thermal drift was read as a speedup; it does not replicate. ``benchmarks/bench_update.py`` is the
corrected version, and it asserts the two forms agree at exactly ``0.0`` before it will report a
timing. Run it rather than trusting the summary above, since the ratios depend on the machine. The
equality itself is a test, in ``tests/test_core_boundary.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

__all__ = ["batch_update", "stepwise_update"]


def stepwise_update(
    weights: npt.NDArray[Any],
    sample: npt.NDArray[Any],
    neighborhood: npt.NDArray[np.floating],
    alpha: float,
) -> npt.NDArray[np.floating]:
    """Move every model toward ``sample`` in proportion to its neighborhood weight.

    This is Eq. (3) of Kohonen (2013),
    ``m_i(t+1) = m_i(t) + h_ci(t) [x(t) - m_i(t)]``, with the learning rate carried in ``alpha``
    rather than folded into ``h``.

    A signed neighborhood is fine here: a negative weight moves the model away from the sample,
    which is the lateral inhibition the mexican hat exists to provide.

    :param weights: Current models, of shape ``(x, y, n_features)``.
    :param sample: The input vector for this step.
    :param neighborhood: Neighborhood weights over the grid, of shape ``(x, y)``.
    :param alpha: Learning rate for this step.
    :return: The updated models, as a new array.
    """
    updated: npt.NDArray[np.floating] = weights + alpha * neighborhood[..., None] * (
        sample - weights
    )
    return updated


def batch_update(
    weights: npt.NDArray[Any],
    sums: npt.NDArray[np.floating],
    counts: npt.NDArray[np.floating],
    hx: npt.NDArray[np.floating],
    hy: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Recompute every model as the neighborhood-weighted mean of the data around it.

    This is Eq. (8) of Kohonen (2013),
    ``m_i = sum_j n_j h_ji xbar_j / sum_j n_j h_ji``, where ``sums[j]`` is ``n_j * xbar_j``.

    That sum runs over every pair of nodes, and because ``h`` depends only on the offset between
    two nodes it is a convolution. Given the neighborhood as a product of per-axis factors,
    ``h_ji == hx[j_x, i_x] * hy[j_y, i_y]``, it contracts to two matrix products and every node is
    computed at once. Kohonen derives Eq. (8) from Eq. (7) on the same grounds, that "the same
    addends occur a great number of times" (Section 4.4); this is the same observation applied once
    more.

    Three properties, each easy to lose:

    **Every model is computed from the models as they stood at the start of the iteration.** Kohonen
    Section 4.4: the old values "are replaced by the respective means, in one concurrent computing
    operation over all nodes of the grid". Nothing here reads a partially updated array.

    **A model with no data in its neighborhood keeps its previous value.** Building the result from
    a zeroed array instead destroys it; on a 30x30 map with 20 samples and a small radius that wiped
    282 of 900 models in a single step. ``out=updated`` with ``where=`` is what preserves it.

    **The denominator needs no tolerance, only ``> 0``.** Every term of ``sum_j n_j h_ji`` is
    non-negative, because a signed neighborhood cannot reach this function: batch training rejects
    the mexican hat, and a caller cannot supply an arbitrary neighborhood since only registered
    names resolve. A sum of non-negative floats admits no cancellation, so it is zero exactly when
    every term is zero, which is exactly the "no data in reach" case.

    :param weights: Current models, of shape ``(x, y, n_features)``.
    :param sums: Per-node sums of the samples mapped to each node.
    :param counts: Per-node counts of the samples mapped to each node.
    :param hx: Per-axis neighborhood factor for the first axis, of shape ``(x, x)``.
    :param hy: Per-axis neighborhood factor for the second axis, of shape ``(y, y)``.
    :return: The updated models, as a new array.
    """
    numerator = np.einsum("ac,bd,cdf->abf", hx, hy, sums, optimize=True)
    denominator = np.einsum("ac,bd,cd->ab", hx, hy, counts, optimize=True)
    updated = weights.copy()
    np.divide(numerator, denominator[..., None], out=updated, where=denominator[..., None] > 0)
    return updated
