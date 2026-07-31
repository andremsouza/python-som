"""Optional numba kernel for the best-matching-unit search. Import is always safe.

Enabled by ``pip install numba``, which this package detects and uses automatically. Without it
:func:`bmu_kernel` returns None and everything runs on the NumPy path, which stays the reference
implementation and the default.

**Deliberately not a dependency and not an extra.** numba 0.66 requires ``numpy<2.5`` while this
package releases against 2.5, so requiring it would cap every user's NumPy; and declaring it as an
extra caps the *lockfile*, because uv resolves every extra together, which would leave development
and CI testing against an older NumPy than the release. Installing it separately keeps that
constraint in the environment that opted into it.

**numba is imported on first use**, not when this module is imported: it costs 104 ms, and the
first training call absorbs that alongside the JIT compile.

The kernel fuses the matrix product and the ``argmin``, keeping the running minimum in a register so
the score matrix is never written. Worth 1.0x to 2.4x, measured. Shell, not core: the kernel reaches
``_core`` as an argument rather than an import.
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

    # No cover: numba compiles this, so the interpreter never executes the body and coverage cannot
    # instrument it. What it does is checked by tests/test_numba_kernel.py, which asserts it returns
    # the same nodes as the NumPy path.
    @njit(parallel=True, cache=True)
    def fused_bmu(  # pragma: no cover
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

    kernel: BmuKernel = fused_bmu  # pragma: no cover  only when numba is installed
    return kernel  # pragma: no cover
