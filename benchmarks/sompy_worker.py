"""Train one SOMPY map and report how long it took. Runs inside ``.venv-sompy``, not this project.

SOMPY cannot be imported alongside python-som. It uses ``np.Inf`` at four sites, three of them as
default arguments evaluated when the class body executes, and ``np.Inf`` was removed in NumPy 2.0,
so ``import sompy`` raises ``AttributeError`` on any NumPy this project supports. Two further
reasons to keep it at arm's length even if that were fixed: importing it reconfigures the **root
logger** for the whole process through ``logging.config.dictConfig``, and it pulls in matplotlib and
scikit-image by way of ``sompy.visualization``.

So the comparison runs across a process boundary. ``bench_vs_sompy.py`` writes a case into a
directory, launches this file with the interpreter from ``.venv-sompy``, and reads the result back.
This half imports **nothing** from python_som, and must not: it would fail on this environment's
NumPy 1.26.

Not meant to be run by hand. See ``bench_vs_sompy.py`` for the setup command and the protocol.

Protocol, all inside the directory given as the only argument:

==============  ==========================================================================
``spec.json``   in: ``shape``, ``n_iteration``, ``radius_start``, ``radius_end``
``input.npz``   in: ``data`` of shape ``(n_samples, n_features)``, ``initial`` of ``(x, y, f)``
``output.npz``  out: ``weights`` of ``(x, y, f)``, ``seconds`` for the training call alone,
                and ``numpy_version``, which the caller cannot read for itself
==============  ==========================================================================
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sompy import SOMFactory


def train(
    data: np.ndarray,
    initial: np.ndarray,
    shape: tuple[int, int],
    n_iteration: int,
    radius_start: float,
    radius_end: float,
) -> tuple[np.ndarray, float]:
    """Train one SOMPY map from injected models and time the training call alone.

    Three things here are deliberate and each is a control the comparison depends on.

    ``normalization="None"`` selects SOMPY's ``NoNormalizer``, a real pass-through class. At its
    ``'var'`` default SOMPY would z-score the data and its models would end up in a different space
    from the ones it is being compared against.

    ``_batchtrain`` rather than ``train``, because ``train`` re-runs initialization unconditionally
    and would throw the injected codebook away. It is also the only training SOMPY implements:
    ``SOMFactory.build`` accepts ``training='seq'`` and then ignores it.

    The codebook is ``(nnodes, n_features)`` with node ``i`` at row ``i // cols``, column
    ``i % cols``, so a C-order reshape converts between that and the ``(x, y, f)`` both other
    libraries use.

    :param data: Training dataset.
    :param initial: Models to start from, of shape ``(x, y, n_features)``.
    :param shape: Grid shape.
    :param n_iteration: Number of iterations.
    :param radius_start: Neighborhood radius at the first iteration.
    :param radius_end: Neighborhood radius at the last iteration.
    :return: The trained models as ``(x, y, f)``, and the elapsed seconds.
    """
    som = SOMFactory.build(
        data,
        mapsize=list(shape),
        normalization="None",
        initialization="random",
        neighborhood="gaussian",
        lattice="rect",
    )
    n_features = data.shape[1]
    som.codebook.matrix = initial.reshape(som.codebook.nnodes, n_features).copy()
    som.codebook.initialized = True

    started = time.perf_counter()
    som._batchtrain(trainlen=n_iteration, radiusin=radius_start, radiusfin=radius_end, njob=1)  # noqa: SLF001
    elapsed = time.perf_counter() - started

    trained = np.asarray(som.codebook.matrix).reshape(shape[0], shape[1], n_features)
    return trained, elapsed


def main() -> None:
    """Read the case from the directory named on the command line and write the result back."""
    # Before anything else. SOMPY's package __init__ installs a root handler at DEBUG, and
    # `_batchtrain` logs an epoch line plus two `timeit` lines per iteration. Left alone that writes
    # thousands of lines to a pipe from inside the timed region, which would be measured as SOMPY
    # being slow at I/O. `train()` would set the level itself; `_batchtrain` does not.
    logging.getLogger().setLevel(logging.ERROR)

    workdir = Path(sys.argv[1])
    spec = json.loads((workdir / "spec.json").read_text(encoding="utf-8"))
    with np.load(workdir / "input.npz") as payload:
        data, initial = payload["data"], payload["initial"]

    weights, seconds = train(
        data,
        initial,
        (int(spec["shape"][0]), int(spec["shape"][1])),
        int(spec["n_iteration"]),
        float(spec["radius_start"]),
        float(spec["radius_end"]),
    )
    # The NumPy version travels with the result. The caller runs a different one by construction,
    # so it cannot report this environment's from its own imports, and a comparison that names the
    # wrong NumPy beside its numbers is worse than one that names none.
    np.savez(
        workdir / "output.npz",
        weights=weights,
        seconds=np.array(seconds),
        numpy_version=np.array(np.__version__),
    )


if __name__ == "__main__":
    main()
