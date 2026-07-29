"""The three ways of choosing initial models, each returning a new array.

Kohonen (2013) Section 4.3 recommends the linear one: "much faster and convergence follow if the
initial values are selected as a regular, two-dimensional sequence of vectors taken along a
hyperplane spanned by the two largest principal components". It is also the only deterministic one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._linalg import pca

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

__all__ = ["linear_models", "random_models", "sample_models"]

#: Linear initialization projects onto two principal components, so it needs at least this many
#: samples and features.
_MIN_PCA_DIMENSIONS = 2


def linear_models(data: npt.NDArray[Any], shape: tuple[int, int]) -> npt.NDArray[np.floating]:
    """Spread the models over the plane of the two largest principal components.

    Each model is ``mean + c1 sqrt(lambda_1) v1 + c2 sqrt(lambda_2) v2`` with ``c1``, ``c2`` evenly
    spaced over ``[-1, 1]``, so the models span the principal plane and are centred on the data.

    PCA is fitted on the raw data rather than on standardized data, so the models live in the same
    space as the inputs they are compared against during training. Fitting on standardized data
    while comparing against raw data put them in a different space entirely.

    :param data: Dataset of shape ``(n_samples, n_features)``.
    :param shape: Shape of the grid.
    :return: Models of shape ``(x, y, n_features)``.
    :raises ValueError: If the dataset has fewer than two samples or two features.
    """
    array = data.astype(float)
    if min(array.shape) < _MIN_PCA_DIMENSIONS:
        msg = (
            "Linear initialization needs at least 2 samples and 2 features, got shape "
            f"{array.shape}"
        )
        raise ValueError(msg)
    fit = pca(array)
    scale = np.sqrt(fit.explained_variance)
    models = np.empty((*shape, array.shape[1]))
    for i, c1 in enumerate(np.linspace(-1, 1, num=shape[0])):
        for j, c2 in enumerate(np.linspace(-1, 1, num=shape[1])):
            models[i, j] = (
                fit.mean + c1 * scale[0] * fit.components[0] + c2 * scale[1] * fit.components[1]
            )
    return models


def sample_models(
    data: npt.NDArray[Any],
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> npt.NDArray[np.floating]:
    """Set every model to a randomly chosen sample from ``data``.

    Draws without replacement when there are at least as many samples as nodes, so that a large
    enough dataset gives every node a distinct starting point.

    :param data: Dataset to sample from.
    :param shape: Shape of the grid.
    :param rng: Generator to draw with.
    :return: Models of shape ``(x, y, n_features)``.
    """
    size = shape[0] * shape[1]
    chosen = rng.choice(len(data), size=size, replace=size > len(data))
    return data[chosen].reshape((*shape, data.shape[1])).astype(float)


def random_models(
    shape: tuple[int, int],
    n_features: int,
    rng: np.random.Generator,
    sample_mode: str = "standard_normal",
) -> npt.NDArray[np.floating]:
    """Draw every model from a random distribution.

    Kohonen (2013) Section 4.3 notes this "was originally used only to demonstrate the capability of
    the SOM to become ordered, starting from an arbitrary initial state".

    :param shape: Shape of the grid.
    :param n_features: Number of features per model.
    :param rng: Generator to draw with.
    :param sample_mode: Either ``'standard_normal'`` or ``'uniform'``.
    :return: Models of shape ``(x, y, n_features)``.
    :raises ValueError: If ``sample_mode`` is not recognised.
    """
    size = (*shape, n_features)
    if sample_mode == "standard_normal":
        return rng.standard_normal(size=size)
    if sample_mode == "uniform":
        return rng.random(size=size)
    msg = (
        f"Invalid value for 'sample_mode' parameter: {sample_mode!r}. "
        "Value should be one of ['standard_normal', 'uniform']"
    )
    raise ValueError(msg)
