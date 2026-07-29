"""Implementation of the 2-D self-organizing map.

Reference:
Teuvo Kohonen, Essentials of the self-organizing map, Neural Networks 37 (2013) 52-65,
ISSN 0893-6080, https://doi.org/10.1016/j.neunet.2012.09.018
"""

from __future__ import annotations

import logging
import secrets
from collections import Counter
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.decomposition
import sklearn.preprocessing

from ._decay import asymptotic_decay
from ._distance import euclidean_distance
from ._neighborhood import (
    NEIGHBORHOOD_FUNCTIONS,
    SIGNED_NEIGHBORHOODS,
    bubble,
    resolve,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable
    from typing import TypeAlias

    from ._neighborhood import NeighborhoodFunction

    #: Anything the public methods accept as a dataset.
    DataLike: TypeAlias = npt.NDArray[Any] | pd.DataFrame | pd.Series[Any] | list[Any]

try:
    import tqdm

    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover
    TQDM_AVAILABLE = False

__all__ = ["SOM"]

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

TRAINING_MODES = ("random", "sequential", "batch")
INITIALIZATION_MODES = ("random", "linear", "sample")

#: Iterations per sample when ``n_iteration`` is not given, by training mode.
DEFAULT_ITERATIONS_PER_SAMPLE = {"random": 1000, "sequential": 1000, "batch": 10}

#: Denominators below this are treated as empty in the batch update, guarding against a
#: near-zero divisor in Kohonen Eq. (8).
_BATCH_DENOMINATOR_TOLERANCE = 1e-12

#: Width of an automatically drawn seed. 128 bits matches what NumPy's own
#: SeedSequence uses when it seeds itself from the OS entropy pool.
_SEED_BITS = 128

#: Linear initialization projects onto two principal components, so it needs at least this
#: many samples and features.
_MIN_PCA_DIMENSIONS = 2


class SOM:
    """A 2-D self-organizing map over NumPy arrays, pandas DataFrames or plain lists.

    Features:
        - Stepwise and batch training
        - Random, random-sampling and linear (PCA) weight initialization
        - Automatic selection of the map size ratio (with PCA)
        - Support for cyclic arrays, for toroidal or spherical maps
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
        learning_rate_decay: Callable[[float, int, int], float] = asymptotic_decay,
        neighborhood_radius: float = 1.0,
        neighborhood_radius_decay: Callable[[float, int, int], float] = asymptotic_decay,
        neighborhood_function: str = "gaussian",
        distance_function: Callable[[Any, Any], npt.NDArray[np.floating]] = euclidean_distance,
        cyclic_x: bool = False,
        cyclic_y: bool = False,
        random_seed: int | None = None,
        data: DataLike | None = None,
        *,
        min_neighborhood_radius: float = 0.5,
    ) -> None:
        """Construct a self-organizing map.

        :param x: Number of rows. If None, chosen from ``data`` by PCA; see
            :meth:`_auto_dimension`. At least one of ``x`` and ``y`` must be given.
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
            x, y = self._auto_dimension(x, y, data)

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

        self._weights = self._rng.standard_normal(
            size=(self._shape[0], self._shape[1], self._input_len)
        )

    @staticmethod
    def _auto_dimension(
        x: int | None,
        y: int | None,
        data: DataLike | None,
    ) -> tuple[int, int]:
        """Choose the missing dimension from the two largest principal components.

        Kohonen (2013) Section 3.5: "it is advisable to select the lengths of the horizontal and
        vertical dimensions of the array to correspond to the lengths of the two largest principal
        components (i.e., those with the highest eigenvalues of the input correlation matrix),
        because then the SOM complies better with the low-order signal statistics."

        A principal component is a unit direction; its *length* in the data is the extent along it,
        which is the standard deviation ``sqrt(lambda)``. The parenthetical identifies which
        components to use, not what quantity to measure. The side-length ratio is therefore
        ``sqrt(lambda_1 / lambda_2)``, which is also what Kohonen's SOM Toolbox implements.

        The x axis is taken to align with the first principal component, so it is the longer side.

        :param x: Number of rows, or None to derive it.
        :param y: Number of columns, or None to derive it.
        :param data: Dataset to run PCA on.
        :return: Both dimensions, as positive integers.
        :raises ValueError: If no dataset is available for sizing.
        """
        if data is None:
            msg = (
                "If one of the dimensions is not specified, a dataset must be provided "
                "for automatic size initialization."
            )
            raise ValueError(msg)
        array = data.to_numpy() if isinstance(data, pd.DataFrame) else np.asarray(data)
        scaled = sklearn.preprocessing.StandardScaler().fit_transform(array)
        pca = sklearn.decomposition.PCA(n_components=2, random_state=0)
        pca.fit(scaled)
        ratio = float(np.sqrt(pca.explained_variance_[0] / pca.explained_variance_[1]))
        if x is None:
            x = max(1, round(y / ratio))  # type: ignore[operator]
        if y is None:
            y = max(1, round(x / ratio))
        return int(x), int(y)

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

    def activate(self, x: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Return the distance from ``x`` to every model of the network.

        :param x: Input vector.
        :return: Distances, with the shape of the network.
        """
        return self._distance_function(x, self._weights)

    def winner(self, x: npt.ArrayLike) -> tuple[int, int]:
        """Return the coordinates of the best-matching unit for ``x``.

        :param x: Input vector.
        :return: Coordinates of the winner.
        """
        activation_map = self.activate(x)
        index = np.unravel_index(activation_map.argmin(), activation_map.shape)
        return int(index[0]), int(index[1])

    def quantization(self, data: DataLike) -> npt.NDArray[np.floating]:
        """Return the distance from each sample to its best-matching model.

        :param data: Dataset of shape ``(n_samples, n_features)``.
        :return: One distance per sample.
        """
        array = self._to_numpy(data)
        return np.array([self._distance_function(i, self._weights[self.winner(i)]) for i in array])

    def quantization_error(self, data: DataLike) -> float:
        """Return the mean distance from each sample to its best-matching model.

        :param data: Dataset of shape ``(n_samples, n_features)``.
        :return: Quantization error.
        """
        return float(self.quantization(data).mean())

    def distance_matrix(self, normalize: bool = False) -> npt.NDArray[np.floating]:
        """Return the U-matrix: the summed distance from each model to its immediate neighbours.

        The distances to one node's neighbourhood are computed and consumed one node at a time
        rather than being accumulated into a full ``(x, y, x, y)`` tensor, which would cost
        ``(x*y)**2`` floats: about 800 MB for a 100x100 map, for a result of ``x*y`` numbers.

        :param normalize: Whether to rescale the result to ``[0, 1]``.
        :return: U-matrix, with the shape of the network.
        """
        um = np.zeros(self._shape)
        for index in np.ndindex(self._shape):
            adjacency = self._adjacency(index)
            distances = self._distance_function(self._weights[index], self._weights)
            um[index] = np.sum(adjacency * distances)
        if normalize:
            spread = np.max(um) - np.min(um)
            if spread > 0:
                um = (um - np.min(um)) / spread
        return um

    def _adjacency(self, c: tuple[int, ...]) -> npt.NDArray[np.floating]:
        """Return an indicator over ``c`` and the nodes immediately adjacent to it.

        Used by :meth:`distance_matrix`, deliberately independent of the configured neighborhood
        function: the U-matrix describes the grid, not the training schedule. The centre is included
        and contributes a distance of zero, so it does not affect the sum.

        :param c: Coordinates of the centre.
        :return: Indicator array, with the shape of the network.
        """
        return bubble(self._shape, (int(c[0]), int(c[1])), 1.0, self._cyclic)

    def activation_matrix(self, data: DataLike) -> npt.NDArray[np.floating]:
        """Return how many samples map to each node.

        :param data: Dataset of shape ``(n_samples, n_features)``.
        :return: Counts, with the shape of the network.
        """
        array = self._to_numpy(data)
        counts = np.zeros(self._shape)
        for i in array:
            counts[self.winner(i)] += 1
        return counts

    def winner_map(self, data: DataLike) -> dict[tuple[int, int], list[npt.NDArray[Any]]]:
        """Return, for each node, the samples that map to it.

        :param data: Dataset of shape ``(n_samples, n_features)``.
        :return: Mapping from node coordinates to the samples assigned to that node.
        """
        array = self._to_numpy(data)
        result: dict[tuple[int, int], list[npt.NDArray[Any]]] = {
            (int(i), int(j)): [] for i, j in np.ndindex(self._shape)
        }
        for i in array:
            result[self.winner(i)].append(i)
        return result

    def label_map(
        self,
        data: DataLike,
        labels: DataLike,
    ) -> dict[tuple[int, int], Counter[Any]]:
        """Return, for each node, the frequency of each label mapped to it.

        :param data: Dataset of shape ``(n_samples, n_features)``.
        :param labels: One label per sample, in the same order as ``data``.
        :return: Mapping from node coordinates to a label counter.
        :raises ValueError: If ``data`` and ``labels`` have different lengths.
        """
        array = self._to_numpy(data)
        label_array = self._to_numpy(labels)
        if len(array) != len(label_array):
            msg = (
                f"'data' and 'labels' must have the same length, got "
                f"{len(array)} and {len(label_array)}"
            )
            raise ValueError(msg)
        counts: dict[tuple[int, int], Counter[Any]] = {
            (int(i), int(j)): Counter() for i, j in np.ndindex(self._shape)
        }
        for instance, label in zip(array, label_array, strict=True):
            counts[self.winner(instance)].update([label])
        return counts

    def train(
        self,
        data: DataLike,
        n_iteration: int | None = None,
        mode: str = "random",
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
        array = self._to_numpy(data)
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
        self,
        array: npt.NDArray[Any],
        n_iteration: int,
        *,
        mode: str,
        verbose: bool,
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
            winner = self.winner(sample)
            self._weights += (
                alpha * self.neighborhood(winner, sigma)[..., None] * (sample - self._weights)
            )

    def _train_batch(self, array: npt.NDArray[Any], n_iteration: int, *, verbose: bool) -> None:
        """Train with the batch algorithm, updating every model concurrently.

        Implements Eq. (8) of Kohonen (2013),
        ``m_i = sum_j n_j h_ji xbar_j / sum_j n_j h_ji``, where ``n_j`` is the number of samples
        mapped to node ``j`` and ``xbar_j`` their mean.

        Two departures from a naive transcription:

        - Models whose neighbourhood contains no data keep their previous value. Starting from a
          zeroed array instead would destroy them; on a 30x30 map with 20 samples and a small
          radius, that wiped 282 of 900 models in a single step.
        - The per-node sums are contracted with NumPy rather than looped over in Python. The
          neighbourhood is evaluated once per node and contracted against the per-node sums and
          counts, which avoids materialising the full ``(x, y, x, y)`` tensor. That tensor would be
          faster still but costs ``(x*y)**2`` floats, about 800 MB for a 100x100 map.

        :param array: Training dataset.
        :param n_iteration: Number of iterations.
        :param verbose: Whether to show a progress bar.
        """
        for t in self._progress(range(n_iteration), n_iteration, verbose=verbose):
            sigma = self._sigma(t, n_iteration)
            sums, counts = self._accumulate(array)

            new_weights = self._weights.copy()
            for node in np.ndindex(self._shape):
                h = self.neighborhood((int(node[0]), int(node[1])), sigma)
                denominator = float(np.sum(h * counts))
                if abs(denominator) > _BATCH_DENOMINATOR_TOLERANCE:
                    new_weights[node] = np.einsum("xy,xyf->f", h, sums) / denominator
            self._weights = new_weights

    def _accumulate(
        self, array: npt.NDArray[Any]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Sum the samples mapped to each node, and count them.

        :param array: Dataset of shape ``(n_samples, n_features)``.
        :return: The per-node sums of shape ``(x, y, n_features)`` and counts of shape ``(x, y)``.
        """
        sums = np.zeros((*self._shape, self._input_len))
        counts = np.zeros(self._shape)
        for sample in array:
            node = self.winner(sample)
            sums[node] += sample
            counts[node] += 1
        return sums, counts

    def weight_initialization(
        self,
        mode: str = "random",
        **kwargs: Any,  # noqa: ANN401  # each initializer takes a different set
    ) -> None:
        """Initialize the models of the network.

        :param mode: One of ``'random'``, ``'linear'`` or ``'sample'``.
        :param kwargs: Passed through to the chosen initializer. ``'random'`` accepts
            ``sample_mode`` (``'standard_normal'`` or ``'uniform'``); ``'linear'`` and ``'sample'``
            require ``data``.
        :raises ValueError: If ``mode`` is unknown.
        """
        modes: dict[str, Callable[..., None]] = {
            "random": self._init_random,
            "linear": self._init_linear,
            "sample": self._init_sample,
        }
        if mode not in modes:
            msg = (
                f"Invalid value for 'mode' parameter: {mode!r}. "
                f"Value should be one of {list(INITIALIZATION_MODES)}"
            )
            raise ValueError(msg)
        modes[mode](**kwargs)

    def _init_random(self, sample_mode: str = "standard_normal") -> None:
        """Draw every model from a random distribution.

        :param sample_mode: Either ``'standard_normal'`` or ``'uniform'``.
        :raises ValueError: If ``sample_mode`` is unknown.
        """
        if sample_mode == "standard_normal":
            self._weights = self._rng.standard_normal(size=self._weights.shape)
        elif sample_mode == "uniform":
            self._weights = self._rng.random(size=self._weights.shape)
        else:
            msg = (
                f"Invalid value for 'sample_mode' parameter: {sample_mode!r}. "
                "Value should be one of ['standard_normal', 'uniform']"
            )
            raise ValueError(msg)

    def _init_linear(self, data: DataLike) -> None:
        """Spread the models over the plane of the two largest principal components.

        Kohonen (2013) Section 4.3: "much faster and convergence follow if the initial values are
        selected as a regular, two-dimensional sequence of vectors taken along a hyperplane spanned
        by the two largest principal components of x (i.e., principal components associated with
        the two highest eigenvalues)". This is the recommended initializer, and unlike the others
        it is deterministic.

        Each model is ``mean + c1 * sqrt(lambda_1) * v1 + c2 * sqrt(lambda_2) * v2`` with ``c1``,
        ``c2`` evenly spaced over ``[-1, 1]``, so the models span the principal plane and are
        centred on the data. PCA is fitted on the raw data, not on standardized data, so the models
        live in the same space as the inputs they are compared against during training.

        :param data: Dataset to run PCA on.
        :raises ValueError: If the dataset has fewer than two samples or features.
        """
        array = self._to_numpy(data).astype(float)
        if min(array.shape) < _MIN_PCA_DIMENSIONS:
            msg = (
                "Linear initialization needs at least 2 samples and 2 features, got shape "
                f"{array.shape}"
            )
            raise ValueError(msg)
        pca = sklearn.decomposition.PCA(n_components=2, random_state=0)
        pca.fit(array)
        components = pca.components_
        scale = np.sqrt(pca.explained_variance_)
        for i, c1 in enumerate(np.linspace(-1, 1, num=self._shape[0])):
            for j, c2 in enumerate(np.linspace(-1, 1, num=self._shape[1])):
                self._weights[i, j] = (
                    pca.mean_ + c1 * scale[0] * components[0] + c2 * scale[1] * components[1]
                )

    def _init_sample(self, data: DataLike) -> None:
        """Set every model to a randomly chosen sample from ``data``.

        :param data: Dataset to sample from.
        """
        array = self._to_numpy(data)
        size = self._shape[0] * self._shape[1]
        chosen = self._rng.choice(len(array), size=size, replace=size > len(array))
        self._weights = array[chosen].reshape(self._weights.shape).astype(float)

    @staticmethod
    def _to_numpy(
        data: DataLike,
    ) -> npt.NDArray[Any]:
        """Convert a DataFrame, Series, list or array to a NumPy array.

        This is the library's data-input boundary: everything downstream of it works on
        ``np.ndarray`` only.

        The pandas branch is strictly redundant. ``np.asarray`` already converts a DataFrame or a
        Series through the ``__array__`` protocol, with identical results including for nullable
        extension dtypes. It is kept here only so that removing the pandas dependency is its own
        reviewable change rather than a side effect of this one.

        :param data: Input data.
        :return: The data as a NumPy array.
        """
        if isinstance(data, pd.DataFrame | pd.Series):
            return data.to_numpy()
        return np.asarray(data)


#: Names of the available neighborhood functions.
AVAILABLE_NEIGHBORHOOD_FUNCTIONS = tuple(sorted(NEIGHBORHOOD_FUNCTIONS))
