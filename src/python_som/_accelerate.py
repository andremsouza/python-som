"""Optional numba kernel for the best-matching-unit search. Import is always safe.

Installed with ``pip install "python-som[fast]"``. Without it :func:`bmu_kernel` returns None and
everything runs on the NumPy path, which stays the reference implementation and the default.

**It is an extra rather than a dependency because of one constraint.** numba requires ``numpy<2.5``,
and this package develops and tests against 2.5. A hard dependency would cap every user's NumPy
below the version we test on and grow the install from one package to three, 93 MB of which 57 MB is
llvmlite. Behind an extra, that reaches only someone who asked for it.

**numba is imported on first use, not on import of this module.** Importing it costs 104 ms, and
``import python_som`` paying that whether or not a map is ever trained would undo a good part of
what the extra buys. The first training call absorbs it alongside the JIT compile.

The kernel fuses the matrix product and the ``argmin`` of
:func:`~python_som._core._match.bmu_indices`, keeping the running minimum in a register so the score
matrix is never written at all. That is the whole of the win: the same arithmetic as the BLAS path
with a fraction of the memory traffic. It does not try to beat BLAS at the product itself.

This module is shell, not core. ``python_som._core`` stays numpy-only, and the kernel reaches it as
an argument rather than an import.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from ._core._protocols import BmuKernel

__all__ = ["bmu_kernel"]


@functools.cache
def bmu_kernel() -> BmuKernel | None:
    """Return the compiled best-matching-unit kernel, or None without the ``fast`` extra.

    Cached, so numba is imported and the kernel compiled at most once per process.

    :return: The kernel, or None.
    """
    try:
        from numba import njit, prange  # noqa: PLC0415  deliberately deferred; see the module docs
    except ImportError:
        return None

    @njit(parallel=True, cache=True)
    def fused_bmu(
        centred_data: npt.NDArray[np.floating],
        centred_models: npt.NDArray[np.floating],
        squared: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.intp]:
        """Return the index of the nearest model for each sample, without a score matrix.

        Computes ``||w||^2 - 2 x.w`` and keeps the smallest, which orders the models exactly as
        ``||x - w||`` does: the dropped ``||x||^2`` is constant per sample. Both arrays arrive
        already centred, so the caller owns the cancellation fix rather than this kernel.

        ``<`` rather than ``<=``, so ties resolve to the lowest index and match ``argmin``.

        :param centred_data: Samples, shifted, of shape ``(n_samples, n_features)``.
        :param centred_models: Models, shifted, of shape ``(n_nodes, n_features)``.
        :param squared: Squared norm of each centred model.
        :return: One flat node index per sample.
        """
        n_samples, n_features = centred_data.shape
        n_nodes = centred_models.shape[0]
        out = np.empty(n_samples, dtype=np.intp)
        for s in prange(n_samples):
            best = np.inf
            best_node = 0
            for node in range(n_nodes):
                score = squared[node]
                for f in range(n_features):
                    score -= 2.0 * centred_data[s, f] * centred_models[node, f]
                if score < best:
                    best = score
                    best_node = node
            out[s] = best_node
        return out

    kernel: BmuKernel = fused_bmu
    return kernel
