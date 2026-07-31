"""Contracts for the strategies a caller can replace.

Protocols rather than bare ``Callable`` aliases, so mypy checks the shape of the call against a
named contract instead of only the argument count.

**Every parameter is positional-only.** Without the ``/``, a Protocol also requires the parameter
*names* to match, so a user's ``def my_decay(rate, step, total)`` would fail against a protocol that
named them differently.

Structural, so nothing inherits from them: this adds checking, not a requirement. See
:doc:`/how-to/use-a-custom-strategy`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import numpy.typing as npt

__all__ = [
    "AxisProfile",
    "BmuKernel",
    "DecayFunction",
    "DistanceFunction",
    "KernelFunction",
    "NeighborhoodFunction",
]


@runtime_checkable
class NeighborhoodFunction(Protocol):
    """Weights the winner's correction across the grid, as a function of grid distance.

    Kohonen (2013) Eq. (5) requires this to depend on ``sqdist(c, i)`` alone, not on the two axis
    offsets separately. A separable product satisfies the signature and is only correct for the
    gaussian.
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
    implementation must broadcast over leading axes.
    """

    def __call__(self, x: Any, weights: Any, /) -> npt.NDArray[np.floating]:  # noqa: ANN401
        """Return the distance from ``x`` to each of ``weights``.

        :param x: Input vector.
        :param weights: One model, or an array of them.
        :return: Distances, with the leading shape of ``weights``.
        """
        ...


@runtime_checkable
class BmuKernel(Protocol):
    """An accelerated best-matching-unit search, supplied from outside the core.

    Optional. ``python_som._accelerate`` provides one with the ``fast`` extra installed; otherwise
    the NumPy path in :func:`~python_som._core._match.bmu_indices` runs. Passed as an argument
    rather than imported, so the core stays numpy-only.

    Both arrays arrive already shifted by a common vector, which is what stops the expanded norm
    cancelling far from the origin. A kernel must not shift them again.
    """

    def __call__(
        self,
        centred_data: npt.NDArray[np.floating],
        centred_models: npt.NDArray[np.floating],
        squared: npt.NDArray[np.floating],
        /,
    ) -> npt.NDArray[np.intp]:
        """Return the index of the nearest model for each sample.

        :param centred_data: Samples, shifted, of shape ``(n_samples, n_features)``.
        :param centred_models: Models, shifted, of shape ``(n_nodes, n_features)``.
        :param squared: Squared norm of each centred model.
        :return: One flat node index per sample, ties going to the lowest index.
        """
        ...


@runtime_checkable
class AxisProfile(Protocol):
    """The per-axis factor of a separable neighborhood, over offsets along one axis.

    Defined only where the factorisation is an identity, which is the gaussian and the bubble. Not a
    general way to build a neighborhood: :class:`NeighborhoodFunction` remains the definition.
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
