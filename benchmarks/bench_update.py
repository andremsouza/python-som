"""Compare the pure update rules against the in-place form that 0.3.0 shipped.

The functional core returns new arrays instead of mutating in place. That is an architectural
choice, and this script keeps it an *informed* one: it measures the cost rather than assuming it,
and it refuses to report a timing until it has proved the two forms compute the same thing.

Run it directly; it is not part of the test suite, because a timing assertion on shared CI hardware
would be flaky::

    uv run python benchmarks/bench_update.py

Method, and why each part is there:

- **The arms are interleaved** rather than timed in sequence. A first attempt ran all repeats of one
  arm and then all of the other, and reported the pure form as 2.56x faster at 100x100; interleaved,
  the same comparison came out at 1.01x. The difference was thermal and load drift over the run,
  which interleaving splits evenly between the arms instead of attributing to one of them.
- **Median and interquartile range**, not the minimum. The minimum picks the single luckiest run,
  which is what produced that 2.56x. Overlapping IQRs are reported as indistinguishable rather than
  dressed up as a small win.
- **Equality is asserted first**, at exactly ``0.0``. Comparing the speed of two functions that
  return different answers measures nothing.
"""

from __future__ import annotations

import statistics
import timeit
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from python_som._core._neighborhood import gaussian
from python_som._core._update import stepwise_update

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import numpy.typing as npt

#: Update steps per timed run.
STEPS = 200

#: Repeats per arm. Odd, so the median is an observation rather than an average of two.
REPEATS = 31

#: Map shapes and feature counts to cover, from "smaller than anyone uses" up.
CASES = [((20, 20), 4), ((50, 50), 10), ((100, 100), 20), ((150, 150), 25)]

#: Learning rate. Constant here: decay is the same in both arms, so it would only add noise.
ALPHA = 0.3

#: Fixed so the reported numbers can be reproduced.
SEED = 20260729


class Case(NamedTuple):
    """Everything one timed comparison needs, built once and shared by both arms."""

    #: Initial models, of shape ``(x, y, n_features)``.
    weights: npt.NDArray[np.floating]
    #: One input vector per step.
    samples: npt.NDArray[np.floating]
    #: One neighborhood per step, precomputed so the timing isolates the update itself.
    neighborhoods: list[npt.NDArray[np.floating]]


def build(shape: tuple[int, int], features: int) -> Case:
    """Build the models, samples and neighborhoods for one case.

    The neighborhoods are precomputed because both arms would call ``gaussian`` identically, so
    including it would add a constant to both sides and shrink the ratio toward 1.

    :param shape: Grid shape.
    :param features: Number of input features.
    :return: The case fixture.
    """
    rng = np.random.default_rng(SEED)
    return Case(
        weights=rng.normal(size=(*shape, features)),
        samples=rng.normal(size=(STEPS, features)),
        neighborhoods=[
            gaussian(
                shape,
                (int(rng.integers(shape[0])), int(rng.integers(shape[1]))),
                3.0,
                (False, False),
            )
            for _ in range(STEPS)
        ],
    )


def run_in_place(case: Case) -> npt.NDArray[np.floating]:
    """Run the loop as 0.3.0 did, mutating one array throughout.

    The expression is verbatim from ``_train_stepwise`` at tag ``v0.3.0``.

    :param case: The case fixture.
    :return: The final models.
    """
    weights = case.weights.copy()
    for step in range(STEPS):
        weights += ALPHA * case.neighborhoods[step][..., None] * (case.samples[step] - weights)
    return weights


def run_pure(case: Case) -> npt.NDArray[np.floating]:
    """Run the loop as this package does, rebinding to the core's return value.

    :param case: The case fixture.
    :return: The final models.
    """
    weights = case.weights.copy()
    for step in range(STEPS):
        weights = stepwise_update(weights, case.samples[step], case.neighborhoods[step], ALPHA)
    return weights


def compare(
    first: Callable[[], object], second: Callable[[], object], repeats: int = REPEATS
) -> tuple[float, float, bool]:
    """Time two callables with their repeats interleaved.

    Also used by ``bench_vs_minisom.py``, which is why ``repeats`` is a parameter: a cross-library
    case trains a whole map per repeat, so it cannot afford the 31 this script uses for a single
    update step. Everything else about the method is deliberately shared rather than reimplemented.

    :param first: One arm.
    :param second: The other arm.
    :param repeats: Number of interleaved repeats. Keep it odd, so the median is an observation.
    :return: Median of each, and whether their interquartile ranges overlap.
    """
    first_times: list[float] = []
    second_times: list[float] = []
    for _ in range(repeats):
        first_times.append(timeit.timeit(first, number=1))
        second_times.append(timeit.timeit(second, number=1))

    low_first, high_first = np.percentile(first_times, [25, 75])
    low_second, high_second = np.percentile(second_times, [25, 75])
    overlap = not (high_first < low_second or high_second < low_first)
    return statistics.median(first_times), statistics.median(second_times), overlap


def main() -> None:
    """Measure both forms on every case and print the comparison."""
    header = f"{'map':>14} {'MB':>7} {'in-place':>12} {'pure':>12} {'ratio':>8}  verdict"
    lines = [header, "-" * len(header)]

    for shape, features in CASES:
        case = build(shape, features)

        # Exact equality is the intended comparison, not a float-tolerance mistake: the two arms
        # evaluate the same expression and differ only in where the result is written, so any
        # difference at all would mean one of them is not the update it claims to be.
        difference = float(np.abs(run_in_place(case) - run_pure(case)).max())
        if difference != 0.0:
            message = f"{shape}: the forms disagree by {difference}, so timing them proves nothing"
            raise AssertionError(message)

        median_in, median_pure, overlap = compare(
            lambda c=case: run_in_place(c), lambda c=case: run_pure(c)
        )
        ratio = median_in / median_pure

        if overlap:
            verdict = "indistinguishable (IQRs overlap)"
        elif ratio > 1:
            verdict = f"pure {ratio:.2f}x faster"
        else:
            verdict = f"pure {1 / ratio:.2f}x slower"

        label = f"{shape[0]}x{shape[1]}x{features}"
        lines.append(
            f"{label:>14} {case.weights.nbytes / 1e6:>7.2f} "
            f"{median_in * 1e3:>10.2f}ms {median_pure * 1e3:>10.2f}ms {ratio:>7.2f}x  {verdict}"
        )

    lines.append(
        f"\nmedian of {REPEATS} interleaved repeats of {STEPS} steps; "
        f"both arms verified equal at exactly 0.0 before timing."
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
