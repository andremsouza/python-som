"""Principal component analysis, and the map sizing that depends on it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import sklearn.decomposition
import sklearn.preprocessing

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

__all__ = ["PrincipalComponents", "auto_dimensions", "pca", "standardize"]

#: Sizing and linear initialization both project onto two components.
_N_COMPONENTS = 2


class PrincipalComponents(NamedTuple):
    """The parts of a PCA fit that this library uses.

    :param mean: Column means of the input, the origin of the component space.
    :param components: The unit direction of each component, one per row.
    :param explained_variance: The eigenvalue associated with each component.
    """

    mean: npt.NDArray[np.floating]
    components: npt.NDArray[np.floating]
    explained_variance: npt.NDArray[np.floating]


def pca(data: npt.NDArray[Any], n_components: int = _N_COMPONENTS) -> PrincipalComponents:
    """Fit a PCA and return its mean, components and explained variance.

    :param data: Array of shape ``(n_samples, n_features)``.
    :param n_components: Number of components to keep.
    :return: The fitted mean, components and explained variance.
    """
    fitted = sklearn.decomposition.PCA(n_components=n_components, random_state=0)
    fitted.fit(data)
    return PrincipalComponents(
        mean=fitted.mean_,
        components=fitted.components_,
        explained_variance=fitted.explained_variance_,
    )


def standardize(data: npt.NDArray[Any]) -> npt.NDArray[np.floating]:
    """Centre each column on zero and scale it to unit variance.

    :param data: Array of shape ``(n_samples, n_features)``.
    :return: The standardized array.
    """
    scaled: npt.NDArray[np.floating] = sklearn.preprocessing.StandardScaler().fit_transform(data)
    return scaled


def auto_dimensions(x: int | None, y: int | None, data: npt.NDArray[Any]) -> tuple[int, int]:
    """Choose the missing grid dimension from the two largest principal components.

    Kohonen (2013) Section 3.5: "it is advisable to select the lengths of the horizontal and
    vertical dimensions of the array to correspond to the lengths of the two largest principal
    components
    (i.e., those with the highest eigenvalues of the input correlation matrix), because then the SOM
    complies better with the low-order signal statistics."

    A principal component is a unit direction; its *length* in the data is the extent along it,
    which is the standard deviation ``sqrt(lambda)``. The parenthetical identifies which
    components to use, not what quantity to measure. The side-length ratio is therefore
    ``sqrt(lambda_1 / lambda_2)``, which is also what Kohonen's SOM Toolbox implements.

    The x axis is taken to align with the first principal component, so it is the longer side.

    :param x: Number of rows, or None to derive it.
    :param y: Number of columns, or None to derive it.
    :param data: Dataset to run PCA on, already converted to an array.
    :return: Both dimensions, as positive integers.
    """
    fit = pca(standardize(data))
    ratio = float(np.sqrt(fit.explained_variance[0] / fit.explained_variance[1]))
    if x is None:
        x = max(1, round(y / ratio))  # type: ignore[operator]
    if y is None:
        y = max(1, round(x / ratio))
    return int(x), int(y)
