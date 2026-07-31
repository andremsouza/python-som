"""The two update rules, as pure functions returning new models.

Both return a new array rather than mutating their argument, so they can be tested without
constructing a :class:`~python_som.SOM`. That costs roughly 10% on small maps and nothing on large
ones; ``benchmarks/bench_update.py`` measures it.
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

    The sum over node pairs is a convolution, and a separable ``h`` contracts it to two matrix
    products with no loop over nodes. See :doc:`/explanation/how-batch-training-is-computed`.

    Three invariants:

    - Every model is computed from the models as they stood at the start of the iteration, which is
      the concurrent update Kohonen requires in Section 4.4.
    - A model with no data in its neighborhood keeps its previous value. Building the result from a
      zeroed array wiped 282 of 900 models in one step on a 30x30 map; ``out=`` with ``where=`` is
      what preserves it.
    - The denominator needs no tolerance, only ``> 0``. Every term is non-negative, since batch
      training rejects signed neighborhoods, so the sum is zero exactly when every term is.

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
