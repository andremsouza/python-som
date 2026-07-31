"""The training loops, which are what a performance change in this package almost always touches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .common import FEATURES, SHAPES, data, som

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    import python_som

#: Iterations per timed run. Batch is the more expensive mode per iteration, so it gets fewer.
BATCH_ITERATIONS = 10
STEPWISE_ITERATIONS = 200


class Batch:
    """Kohonen Eq. (8), the mode this package treats as primary."""

    params = (SHAPES, FEATURES)
    param_names = ("shape", "n_features")

    som: python_som.SOM
    data: np.ndarray

    def setup(self, shape: tuple[int, int], n_features: int) -> None:
        """Build the map and dataset outside the timed region.

        :param shape: Grid shape.
        :param n_features: Number of features.
        """
        self.som = som(shape, n_features)
        self.data = data(n_features)

    def time_train(self, shape: tuple[int, int], n_features: int) -> None:
        """Time batch training.

        The private loop rather than ``train``, which would also score the whole dataset to fill in
        its ``TrainingReport``. That pass is real work a user pays for, but it is not training, and
        folding it in here would blur a change in the loop with a change in the metric.

        :param shape: Unused; asv passes the parameters back.
        :param n_features: Unused.
        """
        del shape, n_features
        self.som._train_batch(self.data, BATCH_ITERATIONS, verbose=False)  # noqa: SLF001

    def peakmem_train(self, shape: tuple[int, int], n_features: int) -> None:
        """Track the memory batch training holds at once.

        Worth tracking rather than assuming: the neighborhood kernel this package introduced in
        0.4.0 is a deliberate memory decision, and the alternative it replaced was a tensor that
        reached 800 MB on a 100x100 map.

        :param shape: Unused.
        :param n_features: Unused.
        """
        del shape, n_features
        self.som._train_batch(self.data, BATCH_ITERATIONS, verbose=False)  # noqa: SLF001


class Stepwise:
    """Kohonen Eq. (3), one sample at a time."""

    params = (SHAPES, FEATURES, ["random", "sequential"])
    param_names = ("shape", "n_features", "mode")

    som: python_som.SOM
    data: np.ndarray

    def setup(self, shape: tuple[int, int], n_features: int, mode: str) -> None:
        """Build the map and dataset outside the timed region.

        :param shape: Grid shape.
        :param n_features: Number of features.
        :param mode: Training mode.
        """
        del mode
        self.som = som(shape, n_features)
        self.data = data(n_features)

    def time_train(self, shape: tuple[int, int], n_features: int, mode: str) -> None:
        """Time stepwise training in either sample order.

        :param shape: Unused.
        :param n_features: Unused.
        :param mode: Either ``'random'`` or ``'sequential'``.
        """
        del shape, n_features
        self.som._train_stepwise(  # noqa: SLF001
            self.data, STEPWISE_ITERATIONS, mode=mode, verbose=False
        )


class MexicanHat:
    """The signed neighborhood, which batch rejects and so only stepwise can exercise."""

    params = (SHAPES,)
    param_names = ("shape",)

    som: python_som.SOM
    data: Any

    def setup(self, shape: tuple[int, int]) -> None:
        """Build the map and dataset outside the timed region.

        :param shape: Grid shape.
        """
        self.som = som(shape, FEATURES[0], neighborhood="mexican_hat")
        self.data = data(FEATURES[0])

    def time_train(self, shape: tuple[int, int]) -> None:
        """Time stepwise training with the mexican hat.

        :param shape: Unused.
        """
        del shape
        self.som._train_stepwise(  # noqa: SLF001
            self.data, STEPWISE_ITERATIONS, mode="random", verbose=False
        )
