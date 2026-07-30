"""scikit-learn adapter. Import this only if you want a map to work inside scikit-learn.

:class:`~python_som.SOM` already provides ``fit``, ``transform``, ``predict`` and ``score``, which
is enough when *you* are the one calling them. It is not enough when scikit-learn does the calling:
since 1.7, ``Pipeline.predict``, ``GridSearchCV`` and ``cross_val_score`` all reach for
``__sklearn_tags__``, and the recommended way to have it is to inherit ``BaseEstimator``.

Measured against scikit-learn 1.9, the methods on :class:`~python_som.SOM` alone give ``clone`` and
``Pipeline.fit`` and then fail: ``Pipeline.predict``, ``GridSearchCV`` and ``cross_val_score`` all
raise ``AttributeError``. :class:`SOMEstimator` passes all five.

So the integration lives here rather than in the core, and scikit-learn stays optional::

    pip install "python-som[sklearn]"

This is the ports-and-adapters shape the package already uses. An adapter may depend on the thing it
adapts; the core stays numpy-only, and importing :mod:`python_som` pulls none of this in.

Defining ``__sklearn_tags__`` by hand was the alternative and was rejected: it couples to an
internal that already changed shape once between 1.6 and 1.7, and scikit-learn's own error message
says it does not recommend the approach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
except ImportError as exc:  # pragma: no cover  exercised in a subprocess, not in this one
    msg = (
        "python_som.sklearn needs scikit-learn, which is not installed. Install it with "
        '`pip install "python-som[sklearn]"`. The rest of python_som does not need it: '
        "python_som.SOM already has fit, transform, predict and score for direct use."
    )
    raise ImportError(msg) from exc

from ._core._decay import asymptotic_decay
from ._core._distance import euclidean_distance
from ._enums import Neighborhood, NeighborhoodStr, TrainingMode, TrainingModeStr
from ._som import SOM

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from ._artifact import TrainingReport
    from ._core._protocols import DecayFunction, DistanceFunction

__all__ = ["SOMEstimator"]


class SOMEstimator(ClusterMixin, TransformerMixin, BaseEstimator):
    """A self-organizing map as a scikit-learn estimator.

    A SOM is a topologically-constrained k-means, so this follows ``KMeans``: ``transform`` gives a
    cluster-distance space, ``predict`` gives one label per sample, ``score`` is negated so that
    larger is better, and fitted attributes carry a trailing underscore.

    >>> from python_som.sklearn import SOMEstimator
    >>> from sklearn.model_selection import GridSearchCV
    >>> search = GridSearchCV(SOMEstimator(), {"x": [4, 6]}, cv=3)   # doctest: +SKIP

    **Every argument is stored unmodified.** scikit-learn's ``clone`` rebuilds an estimator by
    passing ``get_params()`` back to ``__init__`` and then checks the result is identical, so an
    ``__init__`` that validates, coerces or derives anything breaks cloning. All of that is deferred
    to :meth:`fit`, which is why this class holds settings rather than a
    :class:`~python_som.SOM`.

    ``input_len`` is deliberately absent: scikit-learn infers the feature count from ``X``, and
    :attr:`n_features_in_` reports it after fitting.
    """

    def __init__(
        self,
        x: int = 10,
        y: int = 10,
        *,
        n_iteration: int | None = None,
        mode: TrainingMode | TrainingModeStr = TrainingMode.BATCH,
        learning_rate: float = 0.5,
        learning_rate_decay: DecayFunction | None = None,
        neighborhood_radius: float = 1.0,
        neighborhood_radius_decay: DecayFunction | None = None,
        neighborhood_function: Neighborhood | NeighborhoodStr = Neighborhood.GAUSSIAN,
        distance_function: DistanceFunction | None = None,
        cyclic_x: bool = False,
        cyclic_y: bool = False,
        random_seed: int | None = None,
        min_neighborhood_radius: float = 0.5,
        initialization: str = "linear",
    ) -> None:
        """Record the settings a map will be built from.

        Keyword-only after the grid dimensions, as ``KMeans`` is. The decays and the distance
        default to None rather than to the functions themselves, so that the recorded parameters
        stay exactly what the caller passed; :meth:`fit` substitutes the real defaults.

        :param x: Number of rows.
        :param y: Number of columns.
        :param n_iteration: Iterations to train for. Defaults as for :meth:`~python_som.SOM.train`.
        :param mode: Training mode. Batch by default, which Kohonen Section 3.1 recommends.
        :param learning_rate: Initial learning rate. Unused by batch training.
        :param learning_rate_decay: Decay for the learning rate.
        :param neighborhood_radius: Initial neighborhood radius.
        :param neighborhood_radius_decay: Decay for the radius.
        :param neighborhood_function: Which neighborhood to use.
        :param distance_function: Dissimilarity between an input and the models.
        :param cyclic_x: Whether the map wraps vertically.
        :param cyclic_y: Whether the map wraps horizontally.
        :param random_seed: Seed for this estimator's generator.
        :param min_neighborhood_radius: Floor applied to the decayed radius.
        :param initialization: How to seed the models before training, as for
            :meth:`~python_som.SOM.weight_initialization`.
        """
        self.x = x
        self.y = y
        self.n_iteration = n_iteration
        self.mode = mode
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.neighborhood_radius = neighborhood_radius
        self.neighborhood_radius_decay = neighborhood_radius_decay
        self.neighborhood_function = neighborhood_function
        self.distance_function = distance_function
        self.cyclic_x = cyclic_x
        self.cyclic_y = cyclic_y
        self.random_seed = random_seed
        self.min_neighborhood_radius = min_neighborhood_radius
        self.initialization = initialization

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> SOMEstimator:  # noqa: ANN401, ARG002, N803
        """Build a map from the recorded settings and train it on ``X``.

        A fresh map every call, unlike :meth:`python_som.SOM.fit`, which continues from wherever
        its models were. Refitting an estimator is expected to start over: ``GridSearchCV`` fits the
        same cloned estimator on fold after fold, and carrying weights between folds would leak one
        fold into the next.

        :param X: Training dataset of shape ``(n_samples, n_features)``.
        :param y: Ignored.
        :param kwargs: Ignored; accepted because ``ClusterMixin.fit_predict`` forwards them.
        :return: This estimator.
        """
        data = np.asarray(X, dtype=float)
        # Substituted here rather than defaulted in the signature: a callable default that `clone`
        # round-trips would compare unequal to itself under some wrappers, and None keeps the
        # recorded parameters exactly what the caller passed.
        som = SOM(
            x=self.x,
            y=self.y,
            input_len=data.shape[1],
            learning_rate=self.learning_rate,
            learning_rate_decay=(
                asymptotic_decay if self.learning_rate_decay is None else self.learning_rate_decay
            ),
            neighborhood_radius=self.neighborhood_radius,
            neighborhood_radius_decay=(
                asymptotic_decay
                if self.neighborhood_radius_decay is None
                else self.neighborhood_radius_decay
            ),
            neighborhood_function=self.neighborhood_function,
            distance_function=(
                euclidean_distance if self.distance_function is None else self.distance_function
            ),
            cyclic_x=self.cyclic_x,
            cyclic_y=self.cyclic_y,
            random_seed=self.random_seed,
            min_neighborhood_radius=self.min_neighborhood_radius,
        )
        needs_data = self.initialization in {"linear", "sample"}
        som.weight_initialization(
            mode=self.initialization,  # type: ignore[arg-type]
            **({"data": data} if needs_data else {}),
        )
        som.train(data, n_iteration=self.n_iteration, mode=self.mode)

        self.som_ = som
        self.n_features_in_ = data.shape[1]
        self.weights_ = som.get_weights()
        self.quantization_error_ = som.quantization_error(data)
        self.labels_ = som.predict(data)
        return self

    def transform(self, X: Any) -> npt.NDArray[np.floating]:  # noqa: ANN401, N803
        """Return the distance from each sample to every node.

        :param X: Dataset of shape ``(n_samples, n_features)``.
        :return: Distances of shape ``(n_samples, x * y)``.
        """
        return self.som_.transform(X)

    def predict(self, X: Any) -> npt.NDArray[np.integer]:  # noqa: ANN401, N803
        """Return the flat index of the best-matching node for each sample.

        :param X: Dataset of shape ``(n_samples, n_features)``.
        :return: One flat node index per sample.
        """
        return self.som_.predict(X)

    def score(self, X: Any, y: Any = None) -> float:  # noqa: ANN401, ARG002, N803
        """Return the negated quantization error, so that larger is better.

        :param X: Dataset to score.
        :param y: Ignored.
        :return: Negated mean quantization error.
        """
        return self.som_.score(X)

    @property
    def report_(self) -> TrainingReport | None:
        """The training report of the fitted map, or None before fitting.

        :return: The report.
        """
        if not hasattr(self, "som_"):
            return None
        return self.som_.last_report
