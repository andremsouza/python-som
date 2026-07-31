"""The axis-matrix contraction, which is how batch training evaluates Eq. (8).

Eq. (8) needs ``h_ji`` for every pair of nodes. Because a neighborhood depends only on the offset
between two nodes the sum is a convolution, and both neighborhoods batch training admits are
separable, so it contracts to two matrix products against ``(X, X)`` and ``(Y, Y)`` matrices with no
loop over nodes.

The per-node path is benchmarked alongside it. That is what the contraction replaced, and keeping it
measured is what makes the claim checkable rather than historical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from python_som._core._neighborhood import axis_matrix, resolve, resolve_axis_profile
from python_som._core._update import batch_update

from .common import FEATURES, RADIUS, SEED, SHAPES

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import numpy as np
    import numpy.typing as npt

#: Neighborhoods batch training admits, and so the ones with an axis profile.
SEPARABLE = ["gaussian", "bubble"]

#: No wrapping, wrapping on one axis, wrapping on both. The cyclic fold is the part of the offset
#: machinery most likely to be got wrong, and the part whose cost is least obvious.
CYCLIC = [(False, False), (True, False), (True, True)]


class AxisMatrix:
    """Building the two per-axis matrices, once per iteration."""

    params = (SHAPES, SEPARABLE, CYCLIC)
    param_names = ("shape", "neighborhood", "cyclic")

    profile: Callable[..., npt.NDArray[np.floating]]

    def setup(self, shape: tuple[int, int], neighborhood: str, cyclic: tuple[bool, bool]) -> None:
        """Resolve the axis profile outside the timed region.

        :param shape: Grid shape.
        :param neighborhood: Neighborhood function name.
        :param cyclic: Whether each axis wraps.
        """
        del shape, cyclic
        self.profile = resolve_axis_profile(neighborhood)

    def time_build(
        self, shape: tuple[int, int], neighborhood: str, cyclic: tuple[bool, bool]
    ) -> None:
        """Time building both matrices.

        :param shape: Grid shape.
        :param neighborhood: Unused.
        :param cyclic: Whether each axis wraps.
        """
        del neighborhood
        axis_matrix(shape[0], RADIUS, cyclic=cyclic[0], profile=self.profile)
        axis_matrix(shape[1], RADIUS, cyclic=cyclic[1], profile=self.profile)

    def peakmem_build(
        self, shape: tuple[int, int], neighborhood: str, cyclic: tuple[bool, bool]
    ) -> None:
        """Track their size, which is the justification for the approach.

        ``X^2 + Y^2`` floats, against the ``(x, y, x, y)`` tensor the naive form would need, which
        reaches 800 MB on a 100x100 map.

        :param shape: Grid shape.
        :param neighborhood: Unused.
        :param cyclic: Whether each axis wraps.
        """
        del neighborhood
        axis_matrix(shape[0], RADIUS, cyclic=cyclic[0], profile=self.profile)
        axis_matrix(shape[1], RADIUS, cyclic=cyclic[1], profile=self.profile)


class Contraction:
    """One whole Eq. (8) update: both contractions and the guarded divide."""

    params = (SHAPES, FEATURES)
    param_names = ("shape", "n_features")

    weights: npt.NDArray[np.floating]
    sums: npt.NDArray[np.floating]
    counts: npt.NDArray[np.floating]
    hx: npt.NDArray[np.floating]
    hy: npt.NDArray[np.floating]

    def setup(self, shape: tuple[int, int], n_features: int) -> None:
        """Build the models, accumulators and axis matrices outside the timed region.

        :param shape: Grid shape.
        :param n_features: Number of features.
        """
        import numpy as np  # noqa: PLC0415  asv collects this module without running setup

        rng = np.random.default_rng(SEED)
        self.weights = rng.normal(size=(*shape, n_features))
        self.sums = rng.normal(size=(*shape, n_features))
        self.counts = rng.integers(0, 3, size=shape).astype(float)
        profile = resolve_axis_profile("gaussian")
        self.hx = axis_matrix(shape[0], RADIUS, cyclic=False, profile=profile)
        self.hy = axis_matrix(shape[1], RADIUS, cyclic=False, profile=profile)

    def time_update(self, shape: tuple[int, int], n_features: int) -> None:
        """Time the contraction.

        :param shape: Unused.
        :param n_features: Unused.
        """
        del shape, n_features
        batch_update(self.weights, self.sums, self.counts, self.hx, self.hy)

    def peakmem_update(self, shape: tuple[int, int], n_features: int) -> None:
        """Track what one update holds at once.

        :param shape: Unused.
        :param n_features: Unused.
        """
        del shape, n_features
        batch_update(self.weights, self.sums, self.counts, self.hx, self.hy)


class PerNode:
    """Evaluating the neighborhood once per node, which the contraction replaced."""

    params = (SHAPES,)
    param_names = ("shape",)

    evaluate: Callable[..., npt.NDArray[np.floating]]
    nodes: list[tuple[int, int]]

    def setup(self, shape: tuple[int, int]) -> None:
        """Resolve the per-node function and build the node list outside the timed region.

        :param shape: Grid shape.
        """
        self.evaluate = resolve("gaussian")
        self.nodes = [(x, y) for x in range(shape[0]) for y in range(shape[1])]

    def time_evaluate_every_node(self, shape: tuple[int, int]) -> None:
        """Time the path the contraction replaced, on the same work.

        :param shape: Grid shape.
        """
        for node in self.nodes:
            self.evaluate(shape, node, RADIUS, (False, False))
