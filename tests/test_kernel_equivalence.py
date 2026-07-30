"""The kernel form of each neighborhood must equal the per-node form exactly, not approximately.

Batch training evaluates the neighborhood once per iteration and slices it per node, rather than
evaluating it once per node. That is only sound because a neighborhood depends on the offset between
two nodes and never on where the winner sits, so these tests exist to hold that property down.

The bar is **exactly 0.0**, not a tolerance. A speedup that moves trained weights is a bug, and a
tolerance would hide precisely the class of error this replaces: two implementations of one formula
that agree on the cases someone thought to check.

The design makes the equality structural rather than hoped for -- both forms call the same private
profile, differing only in which offsets they pass -- so these tests guard the *premise*
(offset-only dependence, and the right slice) rather than a typo in a second copy of a formula.
"""

from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

import python_som
from python_som import Neighborhood
from python_som._core._match import accumulate
from python_som._core._neighborhood import (
    NEIGHBORHOOD_FUNCTIONS,
    NEIGHBORHOOD_KERNELS,
    axis_offsets,
    bubble,
    gaussian,
    kernel_view,
    mexican_hat,
    offset_span,
    resolve_kernel,
)
from python_som._core._update import batch_update

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from python_som._core._neighborhood import NeighborhoodFunction

#: Grid shapes to sweep, including degenerate single-row and single-column maps, where the offset
#: span collapses and an off-by-one in the slice would be invisible on a square grid.
SHAPES = [(10, 10), (7, 13), (20, 20), (9, 4), (1, 5), (6, 1)]

#: Radii to sweep. ``0.0`` is admissible for the bubble alone, and is included because it is the one
#: value where the neighborhood is a single node and the slice has to be exactly right.
RADII = [0.0, 0.5, 1.0, 2.5, 4.0, 7.0]

#: All four combinations, so a mixed toroidal map (one axis wrapping, one not) is covered. Each axis
#: folds independently, which is why one slice serves every combination.
CYCLIC = list(itertools.product([False, True], repeat=2))

#: The three distinct functions, ignoring the ``mexican_hat``/``mexicanhat`` alias.
FUNCTIONS = {"gaussian": gaussian, "bubble": bubble, "mexican_hat": mexican_hat}


def _evaluate_per_node(
    function: NeighborhoodFunction,
    shape: tuple[int, int],
    sigma: float,
    cyclic: tuple[bool, bool],
    node: tuple[int, int],
) -> np.ndarray:
    """Evaluate one node's neighborhood directly, as batch training did before the kernel.

    Takes everything explicitly at module level rather than closing over the loop variables, so a
    late-binding mistake cannot quietly make both arms of the comparison the same.

    :param function: The per-node neighborhood function.
    :param shape: Shape of the grid.
    :param sigma: This iteration's radius.
    :param cyclic: Whether each axis wraps.
    :param node: Node whose neighborhood is wanted.
    :return: Neighborhood weights over the grid.
    """
    return function(shape, node, sigma, cyclic)


def _admissible(name: str, sigma: float) -> bool:
    """Whether this function accepts this radius.

    :param name: Neighborhood function name.
    :param sigma: Radius.
    :return: True if the call would not raise.
    """
    return sigma > 0 or name == "bubble"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("cyclic", CYCLIC)
def test_kernel_equals_per_node_evaluation_for_every_node(
    shape: tuple[int, int], cyclic: tuple[bool, bool]
) -> None:
    """Sweep every function, radius and **node** of the grid, asserting exact equality.

    This is the load-bearing test of the optimization. Across all parameters it covers 40,832
    (function, shape, cyclic, radius, node) combinations, every one of which must agree at 0.0.
    """
    for name, function in FUNCTIONS.items():
        build = resolve_kernel(name)
        for sigma in RADII:
            if not _admissible(name, sigma):
                continue
            kernel = build(shape, sigma, cyclic)
            assert kernel.shape == (2 * shape[0] - 1, 2 * shape[1] - 1)
            for node in itertools.product(range(shape[0]), range(shape[1])):
                expected = function(shape, node, sigma, cyclic)
                actual = kernel_view(kernel, shape, node)
                difference = np.abs(expected - actual).max()
                assert difference == 0.0, (
                    f"{name} on {shape}, cyclic={cyclic}, sigma={sigma}, node={node}: "
                    f"kernel and per-node evaluation differ by {difference}"
                )


def test_the_sweep_really_covers_every_node() -> None:
    """Guard the test above against silently shrinking.

    A parametrised sweep that stops covering what its docstring claims is worse than no sweep, so
    the count is asserted rather than described.
    """
    total = sum(
        shape[0] * shape[1]
        for shape in SHAPES
        for cyclic in CYCLIC
        for name in FUNCTIONS
        for sigma in RADII
        if _admissible(name, sigma)
    )
    assert total == 40832, total


@pytest.mark.parametrize("name", sorted(NEIGHBORHOOD_KERNELS))
def test_every_registered_function_has_a_kernel(name: str) -> None:
    """Batch training takes the kernel path unconditionally, with no fallback branch.

    That is only safe if the two registries agree, so it is asserted rather than assumed. A name in
    ``NEIGHBORHOOD_FUNCTIONS`` without a kernel would be an ``AttributeError`` deep in training.
    """
    assert name in NEIGHBORHOOD_FUNCTIONS
    assert callable(NEIGHBORHOOD_KERNELS[name])


def test_the_two_registries_have_the_same_keys() -> None:
    """Including the ``mexican_hat``/``mexicanhat`` alias, which is easy to add to only one."""
    assert set(NEIGHBORHOOD_KERNELS) == set(NEIGHBORHOOD_FUNCTIONS)


def test_kernel_view_is_a_view_and_not_a_copy() -> None:
    """Copying the slice would give back much of what evaluating once saved.

    ``batch_update`` calls this for every node, so an accidental copy would allocate ``x * y``
    floats per node per iteration -- exactly the cost the kernel exists to avoid.
    """
    kernel = NEIGHBORHOOD_KERNELS["gaussian"]((12, 9), 2.0, (False, False))
    view = kernel_view(kernel, (12, 9), (5, 4))
    assert np.shares_memory(view, kernel), "kernel_view must not copy"
    assert view.base is not None


@pytest.mark.parametrize("cyclic", [False, True])
def test_offset_span_covers_exactly_the_reachable_offsets(cyclic: bool) -> None:
    """The span must contain every offset ``i - c`` that any pair of nodes can produce.

    One element short and the slice for a corner node would read out of bounds or silently wrap.
    """
    length = 9
    span = offset_span(length, cyclic=cyclic)
    assert span.shape == (2 * length - 1,)

    reachable = {
        float(offset)
        for centre in range(length)
        for offset in axis_offsets(length, centre, cyclic=cyclic)
    }
    assert reachable <= set(span.tolist())


@pytest.mark.parametrize("cyclic", [False, True])
def test_offset_span_agrees_with_axis_offsets_elementwise(cyclic: bool) -> None:
    """The span is ``axis_offsets`` read at a shifted origin, which is what makes the slice valid.

    Asserted per element so a fold applied with the wrong period would fail here rather than as a
    puzzling difference in trained weights. This is the specific trap: the span is ``2L-1`` wide but
    must fold with period ``L``.
    """
    length = 11
    span = offset_span(length, cyclic=cyclic)
    for centre in range(length):
        expected = axis_offsets(length, centre, cyclic=cyclic)
        actual = span[length - 1 - centre : 2 * length - 1 - centre]
        np.testing.assert_array_equal(actual, expected)


def test_a_cyclic_span_cannot_be_faked_with_a_wider_axis() -> None:
    """Pin the reason ``offset_span`` exists instead of reusing ``axis_offsets`` on a ``2L-1`` axis.

    On a flat grid the two coincide, which is what makes this an easy and wrong simplification: the
    fold has to use the real period ``L``, and an axis of width ``2L-1`` folds with the wrong one.
    """
    length = 10
    correct = offset_span(length, cyclic=True)
    naive = axis_offsets(2 * length - 1, length - 1, cyclic=True)
    assert not np.array_equal(correct, naive), (
        "if these ever agree, the simplification is safe and this test should be revisited"
    )


@pytest.mark.parametrize("sigma", [0.0, -1.0, -0.5, float("nan"), float("inf")])
def test_kernels_validate_the_radius_exactly_as_the_per_node_form_does(sigma: float) -> None:
    """Validation lives in the shared profile, so neither form can accept what the other rejects."""
    shape = (6, 6)
    for name, function in FUNCTIONS.items():
        build = resolve_kernel(name)
        per_node_raised = kernel_raised = False
        try:
            function(shape, (3, 3), sigma, (False, False))
        except ValueError:
            per_node_raised = True
        try:
            build(shape, sigma, (False, False))
        except ValueError:
            kernel_raised = True
        assert per_node_raised == kernel_raised, (
            f"{name} at sigma={sigma}: per-node raised={per_node_raised}, kernel={kernel_raised}"
        )


def test_resolve_kernel_rejects_an_unknown_name() -> None:
    """The same error shape as ``resolve``, naming the valid options."""
    with pytest.raises(ValueError, match="Invalid value for 'neighborhood_function' parameter"):
        resolve_kernel("spectral")


# ---------------------------------------------------------------------------------------------
# End to end: training through the kernel equals training through per-node evaluation
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("neighborhood", [Neighborhood.GAUSSIAN, Neighborhood.BUBBLE])
@pytest.mark.parametrize("cyclic", [(False, False), (True, True), (True, False)])
def test_batch_training_is_unchanged_by_the_kernel(
    neighborhood: Neighborhood, cyclic: tuple[bool, bool]
) -> None:
    """Many iterations, with a decaying radius, against the per-node path it replaced.

    The single-iteration case is already covered by
    ``test_batch_matches_a_reference_implementation`` in ``tests/test_training.py``, which builds
    Eq. (8) as a literal double loop. This one runs the
    full loop for long enough that the radius decays through several values, because the kernel is
    rebuilt on each iteration and a stale-kernel bug would only show up after the first.

    Only gaussian and bubble appear: batch training rejects signed neighborhoods, so the mexican hat
    cannot reach this path at all.
    """
    rng = np.random.default_rng(20260731)
    data = rng.normal(size=(80, 3))

    weights = np.asarray(
        python_som.SOM(
            x=7,
            y=5,
            input_len=3,
            neighborhood_function=neighborhood,
            neighborhood_radius=3.0,
            cyclic_x=cyclic[0],
            cyclic_y=cyclic[1],
            random_seed=11,
        ).get_weights()
    )

    def train(*, use_kernel: bool) -> np.ndarray:
        """Run batch training with either the kernel path or a per-node one.

        :param use_kernel: Whether to slice a kernel or evaluate the neighborhood per node.
        :return: The trained models.
        """
        som = python_som.SOM(
            x=7,
            y=5,
            input_len=3,
            neighborhood_function=neighborhood,
            neighborhood_radius=3.0,
            cyclic_x=cyclic[0],
            cyclic_y=cyclic[1],
            random_seed=11,
        )
        som._weights = weights.copy()
        shape = som.get_shape()
        current = weights.copy()
        for step in range(12):
            sigma = som._sigma(step, 12)
            sums, counts = accumulate(data, current, shape, som._distance_function)
            neighborhood_of: Callable[[tuple[int, int]], np.ndarray]
            if use_kernel:
                kernel = resolve_kernel(neighborhood.value)(shape, sigma, cyclic)
                neighborhood_of = functools.partial(kernel_view, kernel, shape)
            else:
                neighborhood_of = functools.partial(
                    _evaluate_per_node, FUNCTIONS[neighborhood.value], shape, sigma, cyclic
                )
            current = batch_update(current, sums, counts, neighborhood_of, shape)
        return current

    difference = np.abs(train(use_kernel=True) - train(use_kernel=False)).max()
    assert difference == 0.0, f"kernel path drifted from per-node path by {difference}"
