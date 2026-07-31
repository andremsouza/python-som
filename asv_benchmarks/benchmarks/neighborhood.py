"""The neighborhood kernel, which is the optimization 0.4.0 rests on.

Batch training needs ``h_ji`` for every pair of nodes. Because a neighborhood depends only on the
offset between two nodes, one kernel over every offset serves the whole grid and each node's
neighborhood is a slice of it. Evaluating per node instead was 42% of batch training on a 40x40 map.

Both paths are still benchmarked, and the per-node one is the point: it is what the kernel
replaced, so keeping it measured is what makes the claim checkable rather than historical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from python_som._core._neighborhood import kernel_view, resolve, resolve_kernel

from .common import RADIUS, SHAPES

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import numpy as np
    import numpy.typing as npt

#: Both neighborhoods batch training accepts, plus the signed one only stepwise can use.
NEIGHBORHOODS = ["gaussian", "bubble", "mexican_hat"]

#: No wrapping, wrapping on one axis, wrapping on both. The cyclic fold is the part of the offset
#: machinery most likely to be got wrong, and the part whose cost is least obvious.
CYCLIC = [(False, False), (True, False), (True, True)]


class Kernel:
    """Building one kernel per iteration."""

    params = (SHAPES, NEIGHBORHOODS, CYCLIC)
    param_names = ("shape", "neighborhood", "cyclic")

    build: Callable[..., npt.NDArray[np.floating]]

    def setup(self, shape: tuple[int, int], neighborhood: str, cyclic: tuple[bool, bool]) -> None:
        """Resolve the kernel builder outside the timed region.

        :param shape: Grid shape.
        :param neighborhood: Neighborhood function name.
        :param cyclic: Whether each axis wraps.
        """
        del shape, cyclic
        self.build = resolve_kernel(neighborhood)

    def time_build(
        self, shape: tuple[int, int], neighborhood: str, cyclic: tuple[bool, bool]
    ) -> None:
        """Time building the kernel once.

        :param shape: Grid shape.
        :param neighborhood: Unused.
        :param cyclic: Whether each axis wraps.
        """
        del neighborhood
        self.build(shape, RADIUS, cyclic)

    def peakmem_build(
        self, shape: tuple[int, int], neighborhood: str, cyclic: tuple[bool, bool]
    ) -> None:
        """Track the kernel's size, which is the justification for the whole approach.

        A ``(2X-1, 2Y-1)`` kernel is 198 KB at 80x80 against the 800 MB a full ``(x, y, x, y)``
        tensor would need. A regression here would otherwise be silent.

        :param shape: Grid shape.
        :param neighborhood: Unused.
        :param cyclic: Whether each axis wraps.
        """
        del neighborhood
        self.build(shape, RADIUS, cyclic)


class Slice:
    """Taking one node's neighborhood out of a built kernel.

    Must stay a view rather than a copy. Copying ``(X, Y)`` floats per node would give back most of
    what the kernel wins, and this is where that would show up as a trend.
    """

    params = (SHAPES,)
    param_names = ("shape",)

    kernel: npt.NDArray[np.floating]
    nodes: list[tuple[int, int]]

    def setup(self, shape: tuple[int, int]) -> None:
        """Build the kernel and the node list outside the timed region.

        :param shape: Grid shape.
        """
        self.kernel = resolve_kernel("gaussian")(shape, RADIUS, (False, False))
        self.nodes = [(x, y) for x in range(shape[0]) for y in range(shape[1])]

    def time_slice_every_node(self, shape: tuple[int, int]) -> None:
        """Time slicing the kernel once per node, which is one batch iteration's worth.

        :param shape: Grid shape.
        """
        for node in self.nodes:
            kernel_view(self.kernel, shape, node)


class PerNode:
    """Evaluating the neighborhood once per node, which the kernel replaced."""

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
        """Time the path the kernel replaced, on the same work.

        :param shape: Grid shape.
        """
        for node in self.nodes:
            self.evaluate(shape, node, RADIUS, (False, False))
