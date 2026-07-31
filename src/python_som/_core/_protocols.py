"""Contracts for the three strategies a caller can replace.

A neighborhood, a decay and a distance are all things a user may supply their own version of. Typed
as bare ``Callable[...]`` aliases, mypy checks little more than the argument count; as Protocols it
checks the shape of the call against a named contract, and the error names the protocol rather than
printing two structural types side by side.

**Every parameter is positional-only** (the ``/`` in each ``__call__``). Without it, a Protocol
requires the *names* to match as well as the types, so a user's ``def my_decay(rate, step, total)``
would fail against a protocol that named them differently. Positional-only says what is actually
true: these are called positionally, and only the order and the types matter.

These are structural, so nothing needs to inherit from them. Every function already in the package
satisfies its protocol, and so does any existing user-supplied callable with the right signature --
this adds checking, not a requirement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import numpy.typing as npt

__all__ = [
    "AxisProfile",
    "DecayFunction",
    "DistanceFunction",
    "KernelFunction",
    "NeighborhoodFunction",
]


@runtime_checkable
class NeighborhoodFunction(Protocol):
    """Weights the winner's correction across the grid, as a function of grid distance.

    Kohonen (2013) Eq. (5) requires this to depend on ``sqdist(c, i)`` alone -- the distance between
    two nodes -- not on the two axis offsets separately. A separable product of per-axis profiles
    satisfies the signature but is only correct for the gaussian.
    """

    def __call__(
        self,
        shape: tuple[int, int],
        c: tuple[int, int],
        sigma: float,
        cyclic: tuple[bool, bool],
        /,
    ) -> npt.NDArray[np.floating]:
        """Evaluate the neighborhood centred on ``c``.

        :param shape: Shape of the network.
        :param c: Coordinates of the winner.
        :param sigma: Neighborhood radius.
        :param cyclic: Whether each axis wraps around.
        :return: Weights with the shape of the network.
        """
        ...


@runtime_checkable
class DecayFunction(Protocol):
    """Reduces a learning rate or a neighborhood radius as training proceeds."""

    def __call__(self, value: float, step: int, total: int, /) -> float:
        """Return the decayed value for this step.

        :param value: The initial value being decayed.
        :param step: Current iteration, counted from zero.
        :param total: Total number of iterations.
        :return: The value to use at this step.
        """
        ...


@runtime_checkable
class DistanceFunction(Protocol):
    """Dissimilarity between an input vector and one or many models.

    Called both with a single model and with the whole ``(x, y, n_features)`` array, so an
    implementation must broadcast over leading axes rather than assume one vector.
    """

    def __call__(self, x: Any, weights: Any, /) -> npt.NDArray[np.floating]:  # noqa: ANN401
        """Return the distance from ``x`` to each of ``weights``.

        :param x: Input vector.
        :param weights: One model, or an array of them.
        :return: Distances, with the leading shape of ``weights``.
        """
        ...


@runtime_checkable
class AxisProfile(Protocol):
    """The per-axis factor of a separable neighborhood, as a function of offsets along one axis.

    Batch training contracts one of these per axis instead of evaluating the neighborhood per node.
    Only defined where the factorisation is an identity: the gaussian, because the exponential
    factors, and the bubble, because its metric is Chebyshev. It is not a general way to build a
    neighborhood, and :class:`NeighborhoodFunction` remains the definition.
    """

    def __call__(self, d: npt.NDArray[np.floating], sigma: float, /) -> npt.NDArray[np.floating]:
        """Evaluate the factor over offsets along one axis.

        :param d: Offsets along the axis.
        :param sigma: Neighborhood radius.
        :return: Weights for those offsets.
        """
        ...


@runtime_checkable
class KernelFunction(Protocol):
    """A neighborhood evaluated over every offset at once, independent of any particular winner.

    .. deprecated:: 0.7.0
        Batch training now contracts an :class:`AxisProfile` per axis, and nothing in the package
        produces a kernel. Retained because it is part of the public surface; it will be removed at
        1.0.0.
    """

    def __call__(
        self, shape: tuple[int, int], sigma: float, cyclic: tuple[bool, bool], /
    ) -> npt.NDArray[np.floating]:
        """Evaluate the neighborhood over every reachable offset.

        :param shape: Shape of the network.
        :param sigma: Neighborhood radius.
        :param cyclic: Whether each axis wraps around.
        :return: Weights of shape ``(2 * shape[0] - 1, 2 * shape[1] - 1)``.
        """
        ...
