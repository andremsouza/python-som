"""Measure what evaluating the neighborhood once per iteration is worth in batch training.

Eq. (8) needs ``h_ji`` for every pair of nodes, so a naive loop evaluates the neighborhood once per
node, once per iteration. Because a neighborhood depends only on the offset between two nodes, one
kernel over every offset serves the whole grid and each node's neighborhood is a slice of it.

Run it directly; it is not part of the test suite, because a timing assertion on shared CI hardware
would be flaky::

    uv run python benchmarks/bench_batch.py

Method is the same as ``bench_update.py``, for the same reasons: **interleaved** arms so thermal and
load drift is split evenly rather than attributed to one of them, **medians with an interquartile
range** rather than minima, and **equality asserted first** -- a speed comparison between two
functions that disagree measures nothing.
"""

from __future__ import annotations

import functools
import statistics
import timeit
from typing import TYPE_CHECKING

import numpy as np

from python_som import SOM
from python_som._core._match import accumulate
from python_som._core._neighborhood import kernel_view, resolve, resolve_kernel
from python_som._core._update import batch_update

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

#: Batch iterations per timed run.
ITERATIONS = 12

#: Repeats per arm. Odd, so the median is an observation rather than an average of two.
REPEATS = 9

#: Grid, sample count and feature count per case. The feature count is varied as well as the grid,
#: because it shifts how much of the work is the contraction rather than the neighborhood, and so
#: changes what there is to win.
CASES = [((20, 20), 200, 4), ((40, 40), 300, 6), ((40, 40), 300, 12), ((60, 60), 400, 8)]

#: Batch training rejects signed neighborhoods, so only these two can reach this path.
NEIGHBORHOODS = ["gaussian", "bubble"]

#: Fixed so the reported numbers can be reproduced.
SEED = 20260731


def train(
    som: SOM,
    data: npt.NDArray[np.floating],
    name: str,
    *,
    use_kernel: bool,
) -> npt.NDArray[np.floating]:
    """Run batch training either through the kernel or by evaluating per node.

    Reproduces ``SOM._train_batch`` closely enough to time the difference, rather than calling it,
    because the per-node arm no longer exists in the package.

    :param som: A constructed map, used for its shape, radius decay and distance function.
    :param data: Training dataset.
    :param name: Neighborhood function name.
    :param use_kernel: Whether to slice one kernel per iteration or evaluate once per node.
    :return: The trained models.
    """
    shape = som.get_shape()
    weights = som.get_weights().copy()
    per_node = resolve(name)
    build = resolve_kernel(name)

    for step in range(ITERATIONS):
        sigma = som._sigma(step, ITERATIONS)  # noqa: SLF001  the decayed radius for this step
        sums, counts = accumulate(data, weights, shape, som._distance_function)  # noqa: SLF001
        if use_kernel:
            kernel = build(shape, sigma, som._cyclic)  # noqa: SLF001

            def neighborhood_of(
                node: tuple[int, int], evaluated: npt.NDArray[np.floating] = kernel
            ) -> npt.NDArray[np.floating]:
                """Slice the kernel for one node."""
                return kernel_view(evaluated, shape, node)

        else:

            def neighborhood_of(
                node: tuple[int, int],
                radius: float = sigma,
            ) -> npt.NDArray[np.floating]:
                """Evaluate the neighborhood for one node."""
                return per_node(shape, node, radius, som._cyclic)  # noqa: SLF001

        weights = batch_update(weights, sums, counts, neighborhood_of, shape)
    return weights


def main() -> None:
    """Measure both paths on every case and print the comparison."""
    header = (
        f"{'map':>9} {'samples':>8} {'features':>9} {'h':>12} {'per-node':>11} {'kernel':>11} "
        f"{'speedup':>8} {'kernel KB':>10}"
    )
    lines = [header, "-" * len(header)]

    for shape, n_samples, n_features in CASES:
        rng = np.random.default_rng(SEED)
        data = rng.normal(size=(n_samples, n_features))

        for name in NEIGHBORHOODS:
            som = SOM(
                x=shape[0],
                y=shape[1],
                input_len=n_features,
                neighborhood_function=name,
                neighborhood_radius=3.0,
                random_seed=SEED,
            )
            som.weight_initialization(mode="random")

            difference = float(
                np.abs(
                    train(som, data, name, use_kernel=True)
                    - train(som, data, name, use_kernel=False)
                ).max()
            )
            if difference != 0.0:
                message = f"{shape} {name}: the two paths disagree by {difference}"
                raise AssertionError(message)

            # functools.partial rather than a lambda: a lambda here would close over the loop
            # variables and time whatever they held when it ran, not when it was written.
            run_slow = functools.partial(train, som, data, name, use_kernel=False)
            run_fast = functools.partial(train, som, data, name, use_kernel=True)
            slow: list[float] = []
            fast: list[float] = []
            for _ in range(REPEATS):
                slow.append(timeit.timeit(run_slow, number=1))
                fast.append(timeit.timeit(run_fast, number=1))

            median_slow, median_fast = statistics.median(slow), statistics.median(fast)
            kernel_kb = (2 * shape[0] - 1) * (2 * shape[1] - 1) * 8 / 1024
            lines.append(
                f"{shape[0]:>4}x{shape[1]:<4} {n_samples:>8} {n_features:>9} {name:>12} "
                f"{median_slow * 1e3:>9.1f}ms {median_fast * 1e3:>9.1f}ms "
                f"{median_slow / median_fast:>7.2f}x {kernel_kb:>10.0f}"
            )

    lines.append(
        f"\nmedian of {REPEATS} interleaved repeats of {ITERATIONS} batch iterations; "
        f"both paths verified equal at exactly 0.0 before timing."
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
