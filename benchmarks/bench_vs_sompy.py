r"""Compare this package against SOMPY, across a process boundary because it has to be.

Run it directly, once the environment below exists::

    uv run python benchmarks/bench_vs_sompy.py

**SOMPY needs an environment of its own, and you have to build it by hand.** Nothing here installs
anything; if the environment is missing the script prints this command and stops::

    uv venv .venv-sompy --python 3.12
    uv pip install --python .venv-sompy \\
        "numpy==1.26.4" "scipy==1.13.1" "scikit-learn==1.5.2" "scikit-image==0.24.0" \\
        "numexpr==2.10.1" "joblib==1.4.2" "matplotlib==3.9.2" \\
        "SOMPY @ git+https://github.com/sevamoo/SOMPY@6aca604b06e5eea1391ecf507810c7aabafc3f8b"

Every part of that is load-bearing, and each was found by the install failing:

- **NumPy below 2.** SOMPY uses ``np.Inf``, removed in NumPy 2.0, at class-definition time. Its own
  metadata says ``numpy >= 1.7`` with no upper bound, so a plain install resolves a NumPy it cannot
  import. Everything else is pinned alongside it in one resolve, because installing any of them
  afterwards pulls NumPy 2 straight back in.
- **Python 3.12.** The ceiling for NumPy 1.26 wheels.
- **scikit-image, joblib and matplotlib.** Imported by SOMPY, absent from its ``install_requires``.
  ``sompy/__init__`` reaches all three through ``from .visualization import *``.
- **A commit SHA.** The repository has no tags, and it is not on PyPI under a name that resolves to
  it: ``pip install sompy`` gets an unrelated 2016 package by a different author.

Which is a lot of scaffolding, so it is worth saying what it buys: SOMPY's batch training is
Kohonen Eq. (8) by way of the Helsinki SOM Toolbox, the same algorithm this package implements, and
comparing two implementations of one published equation is the most informative benchmark available.

Method matches the other scripts in this directory: interleaved arms, medians, and **agreement
asserted before any timing is printed**. Interleaving survives the process boundary because a worker
is launched per repeat and times only its own training call, so the roughly half-second of
interpreter start-up, scipy import and matplotlib import falls outside every measurement.

Controls, beyond the ones in ``bench_vs_minisom.py``:

- SOMPY's normalization is turned off. At its ``'var'`` default it would train on z-scored data.
- The radius follows ``linspace(start, end, n_iteration)`` on both sides, because that is the only
  schedule ``_batchtrain`` can run and this package accepts any callable.
- Batch only. SOMPY implements no stepwise training.

**Agreement here is looser than with MiniSom, for a reason in SOMPY's code rather than in ours.** It
rounds the codebook to six decimals after every update, and its best-matching-unit search expands
``||x-w||^2`` as ``-2x.w + ||w||^2``, which is the less numerically stable arrangement. So the
tolerance is set from measurement rather than from round-off, and the measured figure is printed
with the table so a reader can see what was tolerated rather than take the word for it.

Case sizes stop at 60x60 because SOMPY materialises the full ``(nnodes, nnodes)`` neighborhood: 104
MB there, and 800 MB at 100x100. That cap is printed with the results rather than left silent.
"""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import tempfile
import timeit
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import python_som
from python_som._core._distance import euclidean_distance
from python_som._core._match import quantization

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

#: The interpreter that can import SOMPY. Never created by this script.
SOMPY_PYTHON = Path(__file__).resolve().parent.parent / ".venv-sompy" / "bin" / "python"

#: The worker that runs inside it.
WORKER = Path(__file__).resolve().parent / "sompy_worker.py"

#: Printed when the environment is missing, rather than installing anything.
SETUP = """\
SOMPY needs an environment of its own, because it cannot be imported on NumPy 2.
Build it, then run this script again:

  uv venv .venv-sompy --python 3.12
  uv pip install --python .venv-sompy \\
      "numpy==1.26.4" "scipy==1.13.1" "scikit-learn==1.5.2" "scikit-image==0.24.0" \\
      "numexpr==2.10.1" "joblib==1.4.2" "matplotlib==3.9.2" \\
      "SOMPY @ git+https://github.com/sevamoo/SOMPY@6aca604b06e5eea1391ecf507810c7aabafc3f8b"
"""

#: Repeats per arm. Odd, so the median is an observation rather than an average of two.
REPEATS = 9

#: Grid, sample count, feature count and iteration count per case. Capped at 60x60; see the module
#: docstring for why that is SOMPY's limit rather than an arbitrary choice.
CASES = [((20, 20), 200, 4, 40), ((40, 40), 300, 6, 30), ((60, 60), 400, 8, 20)]

#: The radius ramp, shared by both arms. ``_batchtrain`` runs ``linspace(start, end, n_iteration)``
#: and nothing else, so this package is the one that has to match. The end sits above the default
#: 0.5 radius floor, so that floor never engages.
RADIUS_START = 3.0
RADIUS_END = 1.0

#: Largest tolerated disagreement, relative to the largest model component. Measured at 1.4e-07 to
#: 1.9e-07 across the cases above, consistent with SOMPY rounding its codebook to six decimals on
#: every update: models of order 1 lose about 5e-07 that way. Two orders of headroom. This cannot be
#: the 1e-12 that MiniSom is held to, and the reason is in SOMPY's code rather than in this package.
TOLERANCE = 1e-5

#: Fixed so the reported numbers can be reproduced.
SEED = 20260730


def ramp(x: float, t: int, max_t: int) -> float:
    """Return the radius at iteration ``t``, matching ``np.linspace(start, end, max_t)[t]``.

    :param x: Ignored. The ramp is defined by its endpoints, not by a starting value.
    :param t: Current iteration.
    :param max_t: Total number of iterations.
    :return: The radius for this iteration.
    """
    del x
    if max_t == 1:
        return RADIUS_START
    return RADIUS_START + (RADIUS_END - RADIUS_START) * t / (max_t - 1)


def our_error(weights: npt.NDArray[np.floating], data: npt.NDArray[np.floating]) -> float:
    """Return the quantization error of any library's models, under this package's definition.

    Recomputed rather than read from SOMPY, which is not optional here as it was with MiniSom:
    ``calculate_quantization_error`` returns ``mean(|x - m|)`` over every feature, an elementwise
    mean absolute error, where the standard definition is the mean Euclidean distance per sample.
    SOMPY is inconsistent with itself about this, since the value it logs per epoch during training
    *is* an L2 distance.

    :param weights: Models, of shape ``(x, y, n_features)``.
    :param data: Dataset of shape ``(n_samples, n_features)``.
    :return: Mean distance from each sample to its best-matching model.
    """
    return float(quantization(data, weights, euclidean_distance).mean())


def train_ours(
    shape: tuple[int, int],
    data: npt.NDArray[np.floating],
    initial: npt.NDArray[np.floating],
    n_iteration: int,
) -> npt.NDArray[np.floating]:
    """Train this package's map from the injected models, on the matched radius ramp.

    ``_train_batch`` rather than ``train``, for the reason set out in ``bench_vs_minisom.py``:
    ``train`` also scores the whole dataset to fill in its report, and the other arm does not.

    :param shape: Grid shape.
    :param data: Training dataset.
    :param initial: Models to start from.
    :param n_iteration: Number of iterations.
    :return: The trained models.
    """
    som = python_som.SOM(
        x=shape[0],
        y=shape[1],
        input_len=data.shape[1],
        neighborhood_radius=RADIUS_START,
        neighborhood_radius_decay=ramp,
        neighborhood_function="gaussian",
        random_seed=SEED,
    )
    som._weights = initial.copy()  # noqa: SLF001
    som._train_batch(data, n_iteration, verbose=False)  # noqa: SLF001
    return som.get_weights()


def train_theirs(
    shape: tuple[int, int],
    data: npt.NDArray[np.floating],
    initial: npt.NDArray[np.floating],
    n_iteration: int,
) -> tuple[npt.NDArray[np.floating], float, str]:
    """Run one SOMPY training call in its own interpreter and bring the result back.

    The worker times its own training call, so what this function costs in interpreter start-up and
    imports, roughly half a second, is not part of the number it returns.

    :param shape: Grid shape.
    :param data: Training dataset.
    :param initial: Models to start from.
    :param n_iteration: Number of iterations.
    :return: The trained models, the seconds the worker spent training, and its NumPy version.
    :raises RuntimeError: If the worker fails.
    """
    workdir = Path(tempfile.mkdtemp(prefix="sompy-bench-"))
    try:
        (workdir / "spec.json").write_text(
            json.dumps(
                {
                    "shape": list(shape),
                    "n_iteration": n_iteration,
                    "radius_start": RADIUS_START,
                    "radius_end": RADIUS_END,
                }
            ),
            encoding="utf-8",
        )
        np.savez(workdir / "input.npz", data=data, initial=initial)

        finished = subprocess.run(  # noqa: S603
            [str(SOMPY_PYTHON), str(WORKER), str(workdir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if finished.returncode != 0:
            message = f"the SOMPY worker failed:\n{finished.stdout}\n{finished.stderr}"
            raise RuntimeError(message)

        with np.load(workdir / "output.npz") as payload:
            return payload["weights"], float(payload["seconds"]), str(payload["numpy_version"])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    """Measure every case and print the comparison, or explain how to enable it."""
    if not SOMPY_PYTHON.exists():
        print(SETUP)
        return

    header = (
        f"{'map':>9} {'samples':>8} {'features':>9} {'iters':>6} "
        f"{'python-som':>11} {'SOMPY':>11} {'ratio':>8} {'max diff':>10} "
        f"{'qe ours':>9} {'qe theirs':>10}  verdict"
    )
    lines = [header, "-" * len(header)]
    # Filled in by the first worker run. Printed at the end rather than the start because this
    # process cannot see the other environment's NumPy: `importlib.metadata` would report the one
    # imported here, which is exactly the version SOMPY cannot run on.
    their_numpy = "unknown"

    for shape, n_samples, n_features, n_iteration in CASES:
        rng = np.random.default_rng(SEED)
        data = rng.normal(size=(n_samples, n_features))
        initial = rng.normal(size=(*shape, n_features))

        ours = train_ours(shape, data, initial, n_iteration)
        theirs, _, their_numpy = train_theirs(shape, data, initial, n_iteration)
        difference = float(np.abs(ours - theirs).max()) / float(np.abs(theirs).max())
        if difference > TOLERANCE:
            message = (
                f"{shape}: the two libraries disagree by {difference:.2e} relative, above the "
                f"{TOLERANCE:.0e} tolerance, so timing them proves nothing"
            )
            raise AssertionError(message)

        # Interleaved across the process boundary: one worker launch per repeat, alternating with
        # this package's arm, so drift is split evenly rather than attributed to one of them.
        our_times: list[float] = []
        their_times: list[float] = []
        for _ in range(REPEATS):
            our_times.append(
                timeit.timeit(lambda: train_ours(shape, data, initial, n_iteration), number=1)  # noqa: B023
            )
            their_times.append(train_theirs(shape, data, initial, n_iteration)[1])

        median_ours = statistics.median(our_times)
        median_theirs = statistics.median(their_times)
        low_ours, high_ours = np.percentile(our_times, [25, 75])
        low_theirs, high_theirs = np.percentile(their_times, [25, 75])
        overlap = not (high_ours < low_theirs or high_theirs < low_ours)

        ratio = median_theirs / median_ours
        if overlap:
            verdict = "indistinguishable (IQRs overlap)"
        elif ratio > 1:
            verdict = f"python-som {ratio:.2f}x faster"
        else:
            verdict = f"python-som {1 / ratio:.2f}x slower"

        lines.append(
            f"{shape[0]:>4}x{shape[1]:<4} {n_samples:>8} {n_features:>9} {n_iteration:>6} "
            f"{median_ours * 1e3:>9.1f}ms {median_theirs * 1e3:>9.1f}ms {ratio:>7.2f}x "
            f"{difference:>10.1e} {our_error(ours, data):>9.4f} "
            f"{our_error(theirs, data):>10.4f}  {verdict}"
        )

    nodes = CASES[-1][0][0] * CASES[-1][0][1]
    lines.insert(
        0,
        f"python-som {python_som.__version__} on NumPy {np.__version__}, against "
        f"SOMPY @6aca604 on NumPy {their_numpy}\n",
    )
    lines.append(
        f"\nmedian of {REPEATS} interleaved repeats, batch training only, gaussian only. "
        f"'max diff' is the agreement actually measured, against a {TOLERANCE:.0e} tolerance; "
        f"SOMPY rounds its codebook to 6 decimals every iteration, so exact agreement is not "
        f"available. Quantization error is recomputed under this package's definition for both "
        f"arms, because SOMPY's own method returns an elementwise MAE instead.\n"
        f"Capped at {CASES[-1][0][0]}x{CASES[-1][0][1]}: SOMPY builds the full "
        f"({nodes}, {nodes}) neighborhood matrix, {nodes * nodes * 8 / 1e6:.0f} MB here and "
        f"800 MB at 100x100."
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
