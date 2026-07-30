"""The SOM class: validation, state, the training loops, and delegation to the core.

This is the shell. Every numeric decision lives in :mod:`python_som._core`, whose functions take
their inputs explicitly and return values; this module holds the state those functions are given and
the loops that thread it.

Reference:
Teuvo Kohonen, Essentials of the self-organizing map, Neural Networks 37 (2013) 52-65,
ISSN 0893-6080, https://doi.org/10.1016/j.neunet.2012.09.018
"""

from __future__ import annotations

import logging
import secrets
import warnings
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt

from ._convert import to_numpy
from ._core._decay import asymptotic_decay
from ._core._distance import euclidean_distance
from ._core._initialize import linear_models, random_models, sample_models
from ._core._linalg import auto_dimensions
from ._core._maps import activation_matrix, label_map, u_matrix, winner_map
from ._core._match import accumulate, activate, quantization, winner
from ._core._neighborhood import (
    SIGNED_NEIGHBORHOODS,
    kernel_view,
    resolve,
    resolve_kernel,
)
from ._core._update import batch_update, stepwise_update
from ._enums import (
    Neighborhood,
    NeighborhoodStr,
    SampleMode,
    TrainingMode,
    TrainingModeStr,
    WeightInit,
    WeightInitStr,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections import Counter
    from collections.abc import Iterable

    from ._convert import DataLike
    from ._core._neighborhood import NeighborhoodFunction
    from ._core._protocols import DecayFunction, DistanceFunction

try:
    import tqdm

    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover
    TQDM_AVAILABLE = False

__all__ = ["SOM"]

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: Accepted values, derived from the enums so the two cannot drift. Kept as plain strings because
#: they appear verbatim in error messages.
TRAINING_MODES = tuple(m.value for m in TrainingMode)
INITIALIZATION_MODES = tuple(m.value for m in WeightInit)

#: Iterations per sample when ``n_iteration`` is not given, by training mode. Kohonen (2013)
#: Section 3.1: the batch process "usually needs to be reiterated a few to a few dozen times",
#: against "an order of magnitude" more steps for the stepwise one.
DEFAULT_ITERATIONS_PER_SAMPLE = {"random": 1000, "sequential": 1000, "batch": 10}

#: Width of an automatically drawn seed. 128 bits matches what NumPy's own SeedSequence uses when it
#: seeds itself from the OS entropy pool.
_SEED_BITS = 128


#: Learning rates above this are accepted but warned about. Kohonen gives no hard upper bound, so
#: this is a plausibility threshold rather than a limit: Eq. (3) moves a model a fraction
#: ``alpha * h`` of the way to the sample, and a fraction above 1 overshoots it.
_IMPLAUSIBLE_LEARNING_RATE = 1.0


def _validate_learning_rate(learning_rate: float) -> None:
    """Reject a learning rate that cannot train, and warn about one that is merely unwise.

    Unchecked through 0.3.0, and the two failure modes are different in kind:

    A **non-positive** rate is rejected. ``alpha = 0`` freezes every model, so training runs to
    completion and changes nothing. ``alpha = -1`` is worse: it moves models *away* from the samples
    they match, taking the quantization error from 0.0 to 11.7 and the largest weight to 30 on a map
    that started inside the unit cube. Neither can be what a caller meant, and both are silent.

    A rate **above 1** is warned about, not rejected. Eq. (3) moves a model a fraction ``alpha * h``
    of the way to the sample, so above 1 it overshoots and oscillates around the target rather than
    settling on it. It does not necessarily diverge: measured at ``alpha = 5`` with decay disabled,
    the largest weight stayed at 3.61, because the neighborhood damps the correction away from the
    winner. Kohonen sets no upper bound, so rejecting it would invent a limit the sources do not
    give.

    :param learning_rate: The rate to check.
    :raises ValueError: If the rate is not a finite positive number.
    """
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        msg = f"'learning_rate' must be a finite positive number, got {learning_rate!r}"
        raise ValueError(msg)
    if learning_rate > _IMPLAUSIBLE_LEARNING_RATE:
        warnings.warn(
            f"'learning_rate' is {learning_rate!r}, above 1. Each step moves a model more than the "
            "whole distance to the sample, so training will overshoot and oscillate rather than "
            "converge. Kohonen (2013) Section 4.1 uses rates below 1.",
            UserWarning,
            stacklevel=3,
        )


class SOM:
    """A 2-D self-organizing map over NumPy arrays, pandas DataFrames or plain lists.

    Features:
        - Stepwise and batch training
        - Random, random-sampling and linear (PCA) weight initialization
        - Automatic selection of the map size ratio (with PCA)
        - Support for cyclic arrays, for toroidal maps
        - Gaussian, bubble and mexican hat neighborhood functions
        - Support for custom decay functions
        - Support for visualization (U-matrix, activation matrix)
        - Support for supervised learning (label map)

    Reference:
    Teuvo Kohonen, Essentials of the self-organizing map, Neural Networks 37 (2013) 52-65,
    https://doi.org/10.1016/j.neunet.2012.09.018
    """

    def __init__(
        self,
        x: int | None,
        y: int | None,
        input_len: int,
        learning_rate: float = 0.5,
        learning_rate_decay: DecayFunction = asymptotic_decay,
        neighborhood_radius: float = 1.0,
        neighborhood_radius_decay: DecayFunction = asymptotic_decay,
        neighborhood_function: Neighborhood | NeighborhoodStr = Neighborhood.GAUSSIAN,
        distance_function: DistanceFunction = euclidean_distance,
        cyclic_x: bool = False,
        cyclic_y: bool = False,
        random_seed: int | None = None,
        data: DataLike | None = None,
        *,
        min_neighborhood_radius: float = 0.5,
    ) -> None:
        """Construct a self-organizing map.

        :param x: Number of rows. If None, chosen from ``data`` by PCA. At least one of ``x`` and
            ``y`` must be given.
        :param y: Number of columns. If None, chosen from ``data`` by PCA.
        :param input_len: Number of features per input vector.
        :param learning_rate: Initial learning rate. Irrelevant for batch training.
        :param learning_rate_decay: Decay function for the learning rate.
        :param neighborhood_radius: Initial neighborhood radius.
        :param neighborhood_radius_decay: Decay function for the neighborhood radius.
        :param neighborhood_function: One of ``'gaussian'``, ``'bubble'``, ``'mexicanhat'``
            (or its alias ``'mexican_hat'``).
        :param distance_function: Dissimilarity between an input vector and the models.
        :param cyclic_x: Whether the map wraps around vertically (toroidal).
        :param cyclic_y: Whether the map wraps around horizontally (toroidal).
        :param random_seed: Seed for this instance's random generator.
        :param data: Dataset used for automatic sizing. Required when ``x`` or ``y`` is None.
        :param min_neighborhood_radius: Floor applied to the decayed radius during training.
            Kohonen (2013) Section 4.2 warns that "the final value of sigma shall not go to zero,
            because otherwise the process loses its ordering power. It should always remain, say,
            above half of the grid spacing." Defaults to 0.5, that half-spacing.
        :raises ValueError: If the dimensions, the neighborhood function name, or
            ``min_neighborhood_radius`` are invalid.
        """
        if x is None and y is None:
            msg = "At least one of the dimensions (x, y) must be specified"
            raise ValueError(msg)
        if x is None or y is None:
            if data is None:
                msg = (
                    "If one of the dimensions is not specified, a dataset must be provided "
                    "for automatic size initialization."
                )
                raise ValueError(msg)
            x, y = auto_dimensions(x, y, to_numpy(data))

        if x <= 0 or y <= 0:
            msg = f"Map dimensions must be positive, got ({x}, {y})"
            raise ValueError(msg)
        if input_len <= 0:
            msg = f"'input_len' must be positive, got {input_len}"
            raise ValueError(msg)
        if not np.isfinite(min_neighborhood_radius) or min_neighborhood_radius <= 0:
            msg = (
                "'min_neighborhood_radius' must be a finite positive number, "
                f"got {min_neighborhood_radius!r}"
            )
            raise ValueError(msg)
        _validate_learning_rate(learning_rate)

        self._shape: tuple[int, int] = (int(x), int(y))
        self._input_len = int(input_len)
        self._learning_rate = float(learning_rate)
        self._learning_rate_decay = learning_rate_decay
        self._neighborhood_radius = float(neighborhood_radius)
        self._neighborhood_radius_decay = neighborhood_radius_decay
        self._min_neighborhood_radius = float(min_neighborhood_radius)
        self._neighborhood_function_name = neighborhood_function
        self._neighborhood_function: NeighborhoodFunction = resolve(neighborhood_function)
        self._distance_function = distance_function
        self._cyclic = (bool(cyclic_x), bool(cyclic_y))

        # secrets, rather than an unseeded Generator or SeedSequence, so that the intent reads as
        # "draw unpredictable bits for a seed" and the seed we store is always a plain int.
        self._random_seed = (
            int(random_seed) if random_seed is not None else secrets.randbits(_SEED_BITS)
        )
        self._rng = np.random.default_rng(self._random_seed)
        self._weights = random_models(self._shape, self._input_len, self._rng)

    # ------------------------------------------------------------------ accessors

    def get_shape(self) -> tuple[int, int]:
        """Return the shape of the network."""
        return self._shape

    def get_weights(self) -> npt.NDArray[np.floating]:
        """Return the weight matrix of the network."""
        return self._weights

    def get_random_seed(self) -> int:
        """Return the seed of this instance's random generator."""
        return self._random_seed

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the learning rate.

        :param learning_rate: New learning rate.
        """
        self._learning_rate = float(learning_rate)

    def set_neighborhood_radius(self, neighborhood_radius: float) -> None:
        """Set the neighborhood radius.

        :param neighborhood_radius: New neighborhood radius.
        """
        self._neighborhood_radius = float(neighborhood_radius)

    def neighborhood(self, c: tuple[int, int], sigma: float) -> npt.NDArray[np.floating]:
        """Evaluate the neighborhood function centred on ``c``.

        :param c: Coordinates of the winner.
        :param sigma: Neighborhood radius.
        :return: Neighborhood weights, with the shape of the network.
        """
        return self._neighborhood_function(self._shape, c, sigma, self._cyclic)

    # ------------------------------------------------------------------ inference

    def activate(self, x: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Return the distance from ``x`` to every model of the network.

        :param x: Input vector.
        :return: Distances, with the shape of the network.
        """
        return activate(x, self._weights, self._distance_function)

    def winner(self, x: npt.ArrayLike) -> tuple[int, int]:
        """Return the coordinates of the best-matching unit for ``x``.

        :param x: Input vector.
        :return: Coordinates of the winner.
        """
        return winner(x, self._weights, self._distance_function)

    def quantization(self, data: DataLike) -> npt.NDArray[np.floating]:
        """Return the distance from each sample to its best-matching model.

        :param data: Dataset of shape ``(n_samples, n_features)``.
        :return: One distance per sample.
        """
        return quantization(to_numpy(data), self._weights, self._distance_function)

    def quantization_error(self, data: DataLike) -> float:
        """Return the mean distance from each sample to its best-matching model.

        :param data: Dataset of shape ``(n_samples, n_features)``.
        :return: Quantization error.
        """
        return float(self.quantization(data).mean())

    # ------------------------------------------------------------------ summaries

    def distance_matrix(self, normalize: bool = False) -> npt.NDArray[np.floating]:
        """Return the U-matrix: the summed distance from each model to its immediate neighbours.

        :param normalize: Whether to rescale the result to ``[0, 1]``.
        :return: U-matrix, with the shape of the network.
        """
        return u_matrix(
            self._weights,
            self._shape,
            self._cyclic,
            self._distance_function,
            normalize=normalize,
        )

    def activation_matrix(self, data: DataLike) -> npt.NDArray[np.floating]:
        """Return how many samples map to each node.

        :param data: Dataset of shape ``(n_samples, n_features)``.
        :return: Counts, with the shape of the network.
        """
        return activation_matrix(
            to_numpy(data), self._weights, self._shape, self._distance_function
        )

    def winner_map(self, data: DataLike) -> dict[tuple[int, int], list[npt.NDArray[Any]]]:
        """Return, for each node, the samples that map to it.

        :param data: Dataset of shape ``(n_samples, n_features)``.
        :return: Mapping from node coordinates to the samples assigned to that node.
        """
        return winner_map(to_numpy(data), self._weights, self._shape, self._distance_function)

    def label_map(self, data: DataLike, labels: DataLike) -> dict[tuple[int, int], Counter[Any]]:
        """Return, for each node, the frequency of each label mapped to it.

        :param data: Dataset of shape ``(n_samples, n_features)``.
        :param labels: One label per sample, in the same order as ``data``.
        :return: Mapping from node coordinates to a label counter.
        :raises ValueError: If ``data`` and ``labels`` have different lengths.
        """
        return label_map(
            to_numpy(data),
            to_numpy(labels),
            self._weights,
            self._shape,
            self._distance_function,
        )

    # ------------------------------------------------------------------ training

    def train(
        self,
        data: DataLike,
        n_iteration: int | None = None,
        mode: TrainingMode | TrainingModeStr = TrainingMode.RANDOM,
        verbose: bool = False,
    ) -> float:
        """Train the map and return the resulting quantization error.

        :param data: Training dataset of shape ``(n_samples, n_features)``.
        :param n_iteration: Number of iterations. Defaults to 1000 per sample for the stepwise
            modes and 10 per sample for batch.
        :param mode: One of ``'random'``, ``'sequential'`` or ``'batch'``. Kohonen (2013)
            Section 3.1 recommends batch: "its convergence is an order of magnitude faster and
            safer", and it has no learning-rate parameter.
        :param verbose: Whether to report progress. Emits a tqdm progress bar when tqdm is
            installed, and logs at INFO level on the ``python_som`` logger either way.
        :return: Quantization error after training.
        :raises ValueError: If ``mode`` is unknown, if the dataset is empty, or if batch training
            is combined with a signed neighborhood function.
        """
        array = to_numpy(data)
        if len(array) == 0:
            msg = "Cannot train on an empty dataset"
            raise ValueError(msg)
        if mode not in TRAINING_MODES:
            msg = (
                f"Invalid value for 'mode' parameter: {mode!r}. "
                f"Value should be one of {list(TRAINING_MODES)}"
            )
            raise ValueError(msg)
        if mode == "batch" and self._neighborhood_function_name in SIGNED_NEIGHBORHOODS:
            msg = (
                f"The {self._neighborhood_function_name!r} neighborhood function cannot be used "
                "with the 'batch' training mode: the weighted mean of Kohonen (2013), Eq. (8), is "
                "undefined for a neighborhood function that takes negative values, as its "
                "denominator is not sign-definite. Use mode='random' or mode='sequential'."
            )
            raise ValueError(msg)

        if n_iteration is None:
            n_iteration = DEFAULT_ITERATIONS_PER_SAMPLE[mode] * len(array)
        if n_iteration <= 0:
            msg = f"'n_iteration' must be positive, got {n_iteration}"
            raise ValueError(msg)

        logger.info("Training with %d iterations in %r mode", n_iteration, mode)

        if mode == "batch":
            self._train_batch(array, n_iteration, verbose=verbose)
        else:
            self._train_stepwise(array, n_iteration, mode=mode, verbose=verbose)

        error = self.quantization_error(array)
        logger.info("Quantization error: %g", error)
        return error

    def _sigma(self, t: int, n_iteration: int) -> float:
        """Return the neighborhood radius at iteration ``t``, floored.

        :param t: Current iteration.
        :param n_iteration: Total number of iterations.
        :return: Neighborhood radius, never below ``min_neighborhood_radius``.
        """
        sigma = self._neighborhood_radius_decay(self._neighborhood_radius, t, n_iteration)
        return max(float(sigma), self._min_neighborhood_radius)

    @staticmethod
    def _progress(iterable: Iterable[_T], total: int, *, verbose: bool) -> Iterable[_T]:
        """Wrap ``iterable`` in a tqdm bar when progress reporting is wanted and possible.

        :param iterable: Iterable to wrap.
        :param total: Expected number of items.
        :param verbose: Whether progress was requested.
        :return: The iterable, wrapped or not.
        """
        if verbose and TQDM_AVAILABLE:
            return tqdm.tqdm(iterable, total=total, desc="Training")
        return iterable

    def _train_stepwise(
        self, array: npt.NDArray[Any], n_iteration: int, *, mode: str, verbose: bool
    ) -> None:
        """Train one sample at a time, updating the winner and its neighbourhood.

        Implements Eq. (3) of Kohonen (2013). ``'sequential'`` cycles through the dataset in order,
        wrapping around until ``n_iteration`` steps have run.

        ``'random'`` draws samples **with replacement**, i.i.d., which is the stochastic
        approximation of Robbins and Monro (1951) that Kohonen cites in Section 4.1. Before 0.3.0
        the draw used ``replace=(n_iteration > len(data))``, so it was a random permutation when the
        iteration count did not exceed the sample count and i.i.d. only beyond it. That made the
        character of the sampling depend on the iteration count, which is why it is now uniform.

        :param array: Training dataset.
        :param n_iteration: Number of iterations.
        :param mode: Either ``'random'`` or ``'sequential'``.
        :param verbose: Whether to show a progress bar.
        """
        if mode == "random":
            indices = self._rng.integers(len(array), size=n_iteration)
        else:
            indices = np.resize(np.arange(len(array)), n_iteration)

        for t, index in enumerate(self._progress(indices, n_iteration, verbose=verbose)):
            alpha = self._learning_rate_decay(self._learning_rate, t, n_iteration)
            sigma = self._sigma(t, n_iteration)
            sample = array[index]
            winning = self.winner(sample)
            self._weights = stepwise_update(
                self._weights, sample, self.neighborhood(winning, sigma), alpha
            )

    def _train_batch(self, array: npt.NDArray[Any], n_iteration: int, *, verbose: bool) -> None:
        """Train with the batch algorithm, updating every model concurrently.

        Implements Eq. (8) of Kohonen (2013). The winner map is recomputed from the models as they
        stood at the start of each iteration, which is what makes the update concurrent.

        The neighborhood is evaluated **once per iteration**, not once per node. Eq. (8) needs
        ``h_ji`` for every pair of nodes, and a neighborhood depends only on the offset between the
        two -- so a single kernel over every offset serves the whole grid, and each node's
        neighborhood is a slice of it. Evaluating per node instead made the neighborhood 42% of
        batch training on a 40x40 map, more than the contraction it feeds; measured end to end, the
        kernel is worth **1.2x to 1.5x**, more with the gaussian than the cheaper bubble. See
        :func:`~python_som._core._neighborhood.offset_span` for why the offset-only dependence holds
        on a torus as well as a flat grid, and ``benchmarks/bench_batch.py`` for the measurement.

        :param array: Training dataset.
        :param n_iteration: Number of iterations.
        :param verbose: Whether to show a progress bar.
        """
        build_kernel = resolve_kernel(self._neighborhood_function_name)
        for t in self._progress(range(n_iteration), n_iteration, verbose=verbose):
            sigma = self._sigma(t, n_iteration)
            sums, counts = accumulate(array, self._weights, self._shape, self._distance_function)
            kernel = build_kernel(self._shape, sigma, self._cyclic)

            def neighborhood_of(
                node: tuple[int, int], evaluated: npt.NDArray[np.floating] = kernel
            ) -> npt.NDArray[np.floating]:
                """Take this iteration's neighborhood for ``node`` out of the kernel.

                ``evaluated`` is a default argument rather than a closure over ``kernel`` so that
                the value is bound at definition time, once per iteration.

                :param node: Coordinates of the node whose neighborhood is wanted.
                :param evaluated: This iteration's kernel.
                :return: Neighborhood weights over the grid, as a view into the kernel.
                """
                return kernel_view(evaluated, self._shape, node)

            self._weights = batch_update(self._weights, sums, counts, neighborhood_of, self._shape)

    # ------------------------------------------------------------------ initialization

    def weight_initialization(
        self,
        mode: WeightInit | WeightInitStr = WeightInit.RANDOM,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the models of the network.

        :param mode: One of ``'random'``, ``'linear'`` or ``'sample'``.
        :param kwargs: Passed through to the chosen initializer. ``'random'`` accepts
            ``sample_mode`` (``'standard_normal'`` or ``'uniform'``); ``'linear'`` and ``'sample'``
            require ``data``.
        :raises ValueError: If ``mode`` is unknown, or if the arguments do not suit it.
        """
        if mode not in INITIALIZATION_MODES:
            msg = (
                f"Invalid value for 'mode' parameter: {mode!r}. "
                f"Value should be one of {list(INITIALIZATION_MODES)}"
            )
            raise ValueError(msg)
        try:
            if mode == "random":
                self._weights = random_models(
                    self._shape,
                    self._input_len,
                    self._rng,
                    sample_mode=kwargs.pop("sample_mode", SampleMode.STANDARD_NORMAL),
                )
            elif mode == "linear":
                self._weights = linear_models(to_numpy(kwargs.pop("data")), self._shape)
            else:
                self._weights = sample_models(to_numpy(kwargs.pop("data")), self._shape, self._rng)
        except KeyError as exc:
            # Without this the caller sees a bare KeyError naming the missing key and nothing else.
            msg = f"{mode!r} initialization requires a {exc} argument"
            raise ValueError(msg) from exc
        if kwargs:
            msg = (
                f"Unexpected argument(s) for {mode!r} initialization: {sorted(kwargs)}. "
                f"Valid modes are {list(INITIALIZATION_MODES)}."
            )
            raise ValueError(msg)
