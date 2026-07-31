"""Compare this package against MiniSom, the closest comparable implementation.

Run it directly; it is not part of the test suite, because a timing assertion on shared CI hardware
would be flaky::

    uv run python benchmarks/bench_vs_minisom.py

Method is the same as ``bench_update.py`` and ``bench_batch.py``, and the interleaving helper is
imported from the first rather than copied: **interleaved** arms so thermal and load drift is split
evenly rather than attributed to one of them, **medians with an interquartile range** rather than
minima, and **equality asserted first**.

That last rule is doing much more work here than in the other two scripts. Within one package it
guards against a refactor that changed an answer; across two packages it is the entire basis for the
comparison being meaningful at all. MiniSom and this package look far more interchangeable than they
are, so the numbers are only worth printing once the two have been driven into computing the same
thing. Six controls do that, and every one of them is also asserted in
``tests/test_minisom_agreement.py`` so the protocol cannot quietly stop holding:

1. The same initial models are injected into both. Seeding cannot do it: the two use different
   generators and their PCA initializations are different functions.
2. The gaussian only. It is the one neighborhood the two define identically; their mexican hats
   differ by a factor of two in the linear term and their bubbles select different node sets.
3. The same radius schedule, ``asymptotic_decay``, which is the same formula in both packages.
4. Cases chosen so this package's radius floor never engages, since MiniSom has no such floor.
   Asserted, because it is a property of the chosen numbers rather than of the code.
5. For batch, MiniSom's learning rate pinned to a constant 1.0. ``train_batch_offline`` relaxes
   toward the Eq. (8) mean by a decaying ``eta`` where Eq. (8) simply assigns it.
6. Sequential rather than random sample order, because ``arange(T) % n`` and
   ``np.resize(arange(n), T)`` are the same sequence. The random modes are genuinely different
   (i.i.d. draws here, a shuffled fixed multiset there) and no seed reconciles them.

Note which MiniSom methods those map to. ``train_batch`` is *stepwise* training in sequential sample
order, the counterpart of ``mode="sequential"`` here. Kohonen Eq. (8) is ``train_batch_offline``.
Reaching for the name that matches this package's would time a weighted mean against per-sample
gradient steps.

**What is timed.** The training loops, ``_train_batch`` and ``_train_stepwise``, not the public
``train``. This is not a shortcut, it is the difference between a fair comparison and a wrong one.
``train`` also computes the quantization error over the whole dataset to fill in its
``TrainingReport``; MiniSom's entry points do nothing of the kind. Timing ``train`` therefore
charges this package for a full extra pass over the data that the other arm never pays for, and on
the
sequential cases that pass is larger than the training itself: 30 steps touch 30 samples, the report
touches all 400. The first version of this script did exactly that and reported this package as
**7.10x slower** on the largest sequential case. Timing the loop, the same case is **1.08x faster**.
Both numbers were reproducible; only one of them measured training. ``bench_batch.py`` avoids the
same trap by reproducing the loop rather than calling the public method.

Two tables are printed and the difference between them matters:

- **Same mathematics** applies all six controls, so both libraries compute the same result and the
  only thing left to measure is how fast they compute it.
- **Own initializer** keeps the algorithm and every hyperparameter fixed and lets each library seed
  its own models the way its documentation says to. It is not an algorithm comparison; it measures
  one packaging decision, which happens to be the only default the two do not already share.
"""

from __future__ import annotations

import functools
import tracemalloc
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np
from bench_update import compare
from minisom import MiniSom

import python_som
from python_som._core._decay import asymptotic_decay
from python_som._core._distance import euclidean_distance
from python_som._core._match import quantization

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import numpy.typing as npt

#: Repeats per arm. Odd, so the median is an observation rather than an average of two.
REPEATS = 9

#: Grid, sample count, feature count and iteration count per case.
CASES = [((20, 20), 200, 4, 60), ((40, 40), 300, 6, 40), ((60, 60), 400, 8, 30)]

#: Initial radius, shared by both libraries. Large enough that control 4 holds for every case above.
SIGMA_0 = 3.0

#: Initial learning rate, shared by both libraries. Unused by batch training.
LEARNING_RATE_0 = 0.5

#: Largest tolerated disagreement between the trained models, relative to the largest component.
#: Measured at 2.8e-16 for sequential and 1.3e-15 for batch, so this is three orders of headroom.
#: Exact equality is not available across two packages: they accumulate the same sums in different
#: orders. It is the same constant as ``tests/test_minisom_agreement.py``, for the same reason.
TOLERANCE = 1e-12

#: Fixed so the reported numbers can be reproduced.
SEED = 20260730


def build(
    shape: tuple[int, int], n_samples: int, n_features: int
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Build the dataset and the shared initial models for one case.

    :param shape: Grid shape.
    :param n_samples: Number of samples.
    :param n_features: Number of input features.
    :return: The dataset and the initial models.
    """
    rng = np.random.default_rng(SEED)
    return rng.normal(size=(n_samples, n_features)), rng.normal(size=(*shape, n_features))


def our_error(weights: npt.NDArray[np.floating], data: npt.NDArray[np.floating]) -> float:
    """Return the quantization error of any library's models, under this package's definition.

    Applied to every arm rather than calling each library's own method. Here the two definitions do
    agree, which ``tests/test_minisom_agreement.py`` asserts, but recomputing keeps the reported
    column meaning one thing regardless of whose models produced it.

    :param weights: Models, of shape ``(x, y, n_features)``.
    :param data: Dataset of shape ``(n_samples, n_features)``.
    :return: Mean distance from each sample to its best-matching model.
    """
    return float(quantization(data, weights, euclidean_distance).mean())


# ---------------------------------------------------------------------------------------------
# The two arms
# ---------------------------------------------------------------------------------------------


def train_ours(
    shape: tuple[int, int],
    data: npt.NDArray[np.floating],
    initial: npt.NDArray[np.floating],
    n_iteration: int,
    mode: str,
) -> npt.NDArray[np.floating]:
    """Train this package's map from the injected models.

    :param shape: Grid shape.
    :param data: Training dataset.
    :param initial: Models to start from.
    :param n_iteration: Number of iterations.
    :param mode: Either ``'batch'`` or ``'sequential'``.
    :return: The trained models.
    """
    som = python_som.SOM(
        x=shape[0],
        y=shape[1],
        input_len=data.shape[1],
        learning_rate=LEARNING_RATE_0,
        learning_rate_decay=asymptotic_decay,
        neighborhood_radius=SIGMA_0,
        neighborhood_radius_decay=asymptotic_decay,
        neighborhood_function="gaussian",
        random_seed=SEED,
    )
    som._weights = initial.copy()  # noqa: SLF001  control 1: the same models on both sides
    # The private loops rather than `train`. See "What is timed" in the module docstring.
    if mode == "batch":
        som._train_batch(data, n_iteration, verbose=False)  # noqa: SLF001
    else:
        som._train_stepwise(data, n_iteration, mode=mode, verbose=False)  # noqa: SLF001
    return som.get_weights()


def train_theirs(
    shape: tuple[int, int],
    data: npt.NDArray[np.floating],
    initial: npt.NDArray[np.floating],
    n_iteration: int,
    mode: str,
) -> npt.NDArray[np.floating]:
    """Train MiniSom from the injected models, with the schedules matched.

    :param shape: Grid shape.
    :param data: Training dataset.
    :param initial: Models to start from.
    :param n_iteration: Number of iterations.
    :param mode: Either ``'batch'`` or ``'sequential'``.
    :return: The trained models.
    """
    batch = mode == "batch"
    peer = MiniSom(
        shape[0],
        shape[1],
        data.shape[1],
        sigma=SIGMA_0,
        # Control 5: Eq. (8) has no step size, so the blend has to be turned off for batch. For
        # stepwise the rate is part of Eq. (3) and matches this package's default.
        learning_rate=1.0 if batch else LEARNING_RATE_0,
        decay_function=(lambda rate, t, max_t: 1.0) if batch else "asymptotic_decay",  # noqa: ARG005
        sigma_decay_function="asymptotic_decay",
        neighborhood_function="gaussian",
        random_seed=SEED,
    )
    peer._weights = initial.copy()  # noqa: SLF001
    if batch:
        peer.train_batch_offline(data, n_iteration)
    else:
        peer.train_batch(data, n_iteration)  # despite the name, this is stepwise sequential
    weights: npt.NDArray[np.floating] = peer.get_weights()
    return weights


def seed_and_train_ours(
    shape: tuple[int, int],
    data: npt.NDArray[np.floating],
    n_iteration: int,
) -> npt.NDArray[np.floating]:
    """Seed and train with this package's own initializer and defaults.

    :param shape: Grid shape.
    :param data: Training dataset.
    :param n_iteration: Number of iterations.
    :return: The trained models.
    """
    som = python_som.SOM(x=shape[0], y=shape[1], input_len=data.shape[1], random_seed=SEED)
    som.weight_initialization(mode="linear", data=data)
    som._train_batch(data, n_iteration, verbose=False)  # noqa: SLF001  see "What is timed"
    return som.get_weights()


def seed_and_train_theirs(
    shape: tuple[int, int],
    data: npt.NDArray[np.floating],
    n_iteration: int,
) -> npt.NDArray[np.floating]:
    """Seed and train with MiniSom's own initializer and defaults.

    :param shape: Grid shape.
    :param data: Training dataset.
    :param n_iteration: Number of iterations.
    :return: The trained models.
    """
    peer = MiniSom(shape[0], shape[1], data.shape[1], random_seed=SEED)
    peer.pca_weights_init(data)
    peer.train_batch_offline(data, n_iteration)
    weights: npt.NDArray[np.floating] = peer.get_weights()
    return weights


def assert_the_floor_never_engaged(shape: tuple[int, int], n_iteration: int) -> None:
    """Check control 4 for one case.

    This package floors the decayed radius at ``min_neighborhood_radius``; MiniSom does not. If the
    floor engaged, the two would be running different schedules and the agreement check below would
    pass or fail for the wrong reason.

    :param shape: Grid shape.
    :param n_iteration: Number of iterations.
    :raises AssertionError: If the floor changed the radius at any step.
    """
    som = python_som.SOM(
        x=shape[0], y=shape[1], input_len=2, neighborhood_radius=SIGMA_0, random_seed=SEED
    )
    for t in range(n_iteration):
        if som._sigma(t, n_iteration) != asymptotic_decay(SIGMA_0, t, n_iteration):  # noqa: SLF001
            message = f"{shape}: the radius floor engaged at t={t}, so the schedules differ"
            raise AssertionError(message)


def peak_megabytes(run: Callable[[], object]) -> float:
    """Return the peak memory one call allocates, in megabytes.

    Measured in a pass of its own rather than during the timed repeats, because tracing every
    allocation slows the thing being timed. NumPy registers its allocator with ``tracemalloc``, so
    array data is included and not just the Python objects wrapping it.

    :param run: The callable to measure.
    :return: Peak traced allocation in megabytes.
    """
    tracemalloc.start()
    try:
        run()
        return tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()


# ---------------------------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------------------------


def same_mathematics() -> list[str]:
    """Time both libraries on identical inputs, after proving they compute the same thing.

    :return: The rendered table.
    :raises AssertionError: If the two disagree by more than the tolerance.
    """
    header = (
        f"{'map':>9} {'samples':>8} {'features':>9} {'iters':>6} {'mode':>11} "
        f"{'python-som':>11} {'MiniSom':>11} {'ratio':>8} {'peak MB':>16}  verdict"
    )
    lines = [header, "-" * len(header)]

    for shape, n_samples, n_features, n_iteration in CASES:
        data, initial = build(shape, n_samples, n_features)
        assert_the_floor_never_engaged(shape, n_iteration)

        for mode in ("batch", "sequential"):
            ours = functools.partial(train_ours, shape, data, initial, n_iteration, mode)
            theirs = functools.partial(train_theirs, shape, data, initial, n_iteration, mode)

            trained_ours, trained_theirs = ours(), theirs()
            difference = float(np.abs(trained_ours - trained_theirs).max()) / float(
                np.abs(trained_theirs).max()
            )
            if difference > TOLERANCE:
                message = (
                    f"{shape} {mode}: the two libraries disagree by {difference:.2e} relative, "
                    f"above the {TOLERANCE:.0e} tolerance, so timing them proves nothing"
                )
                raise AssertionError(message)

            peak_ours, peak_theirs = peak_megabytes(ours), peak_megabytes(theirs)
            median_ours, median_theirs, overlap = compare(ours, theirs, REPEATS)
            ratio = median_theirs / median_ours
            if overlap:
                verdict = "indistinguishable (IQRs overlap)"
            elif ratio > 1:
                verdict = f"python-som {ratio:.2f}x faster"
            else:
                verdict = f"python-som {1 / ratio:.2f}x slower"

            lines.append(
                f"{shape[0]:>4}x{shape[1]:<4} {n_samples:>8} {n_features:>9} {n_iteration:>6} "
                f"{mode:>11} {median_ours * 1e3:>9.1f}ms {median_theirs * 1e3:>9.1f}ms "
                f"{ratio:>7.2f}x {peak_ours:>7.1f} /{peak_theirs:>7.1f}  {verdict}"
            )

    lines.append(
        f"\nmedian of {REPEATS} interleaved repeats; trained models verified equal to within "
        f"{TOLERANCE:.0e} relative before timing. Peak MB is python-som / MiniSom, measured "
        f"separately from the timings."
    )
    return lines


def own_initializer() -> list[str]:
    """Compare the two initializers, with the algorithm and every hyperparameter held fixed.

    The defaults happen to be identical on both sides (sigma 1.0, learning rate 0.5, asymptotic
    decay on both schedules, gaussian, euclidean), so the initializer is the only default that
    differs and this table isolates it rather than confounding it with anything else.

    The two are genuinely different functions, not two spellings of one. This package places models
    at ``mean + c1*sqrt(lambda1)*v1 + c2*sqrt(lambda2)*v2``, scaling each principal direction by the
    data's actual extent along it, from an SVD of the centred matrix. MiniSom places them at
    ``mean + c1*v1 + c2*v2`` with unit eigenvectors, so the initial sheet is one unit across
    whatever the data's spread, and it eigendecomposes the covariance matrix rather than the data.

    :return: The rendered table.
    """
    header = (
        f"{'map':>9} {'samples':>8} {'features':>9} {'iters':>6} "
        f"{'python-som':>11} {'MiniSom':>11} {'qe ours':>9} {'qe theirs':>10}  better"
    )
    lines = [header, "-" * len(header)]

    for shape, n_samples, n_features, n_iteration in CASES:
        rng = np.random.default_rng(SEED)
        data = rng.normal(size=(n_samples, n_features)) * 10.0 + 100.0

        # functools.partial rather than a closure: a closure over the loop variables would time
        # whatever they held when it ran, not when it was written. Same reason bench_batch.py does.
        ours = functools.partial(seed_and_train_ours, shape, data, n_iteration)
        theirs = functools.partial(seed_and_train_theirs, shape, data, n_iteration)

        error_ours = our_error(ours(), data)
        error_theirs = our_error(theirs(), data)
        median_ours, median_theirs, _ = compare(ours, theirs, REPEATS)

        better = "python-som" if error_ours < error_theirs else "MiniSom"
        lines.append(
            f"{shape[0]:>4}x{shape[1]:<4} {n_samples:>8} {n_features:>9} {n_iteration:>6} "
            f"{median_ours * 1e3:>9.1f}ms {median_theirs * 1e3:>9.1f}ms "
            f"{error_ours:>9.4f} {error_theirs:>10.4f}  {better}"
        )

    lines.append(
        "\nBatch training on both sides; only the initializer differs. Quantization error is "
        "recomputed under this package's definition for both arms. Data is deliberately offset "
        "from the origin and scaled by 10, which is where the two initializers diverge most."
    )
    return lines


def main() -> None:
    """Print both tables, with the versions they were measured against."""
    print(
        f"python-som {python_som.__version__}, MiniSom {version('minisom')}, "
        f"NumPy {np.__version__}\n"
    )
    print("SAME MATHEMATICS: both libraries computing the same result, timed")
    print("\n".join(same_mathematics()))
    print("\n")
    print("OWN INITIALIZER: same algorithm and hyperparameters, each library seeding its own")
    print("\n".join(own_initializer()))


if __name__ == "__main__":
    main()
