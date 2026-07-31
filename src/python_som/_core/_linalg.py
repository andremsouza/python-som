"""Principal component analysis, and the map sizing that depends on it.

Built on ``np.linalg.svd`` rather than scikit-learn, which it reproduces exactly, sign convention
and degenerate columns included. ``tests/test_linalg_matches_sklearn.py`` re-checks that against the
real scikit-learn on every CI run, which is why scikit-learn is still a test dependency.

See :doc:`/explanation/why-linear-initialization-is-an-svd` for why the SVD is also the more
accurate of the two routes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

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

    For ``X - mean = U S V^T``, the rows of ``V^T`` are the component directions and ``S^2 / (n-1)``
    the variance along each.

    The sign convention is **v-based**: each component is oriented so its largest-magnitude loading
    is positive, matching scikit-learn's ``svd_flip(..., u_based_decision=False)``. The other
    convention would give a valid but different PCA, and lay the initial models out reversed.

    :param data: Array of shape ``(n_samples, n_features)``.
    :param n_components: Number of components to keep.
    :return: The fitted mean, components and explained variance.
    """
    array = np.asarray(data, dtype=float)
    mean = array.mean(axis=0)
    _, singular_values, right_vectors = np.linalg.svd(array - mean, full_matrices=False)

    # Orient each component on its largest-magnitude loading. The flip is applied to all of them
    # before truncation, which is the order scikit-learn does it in.
    dominant = np.argmax(np.abs(right_vectors), axis=1)
    signs = np.sign(right_vectors[np.arange(right_vectors.shape[0]), dominant])
    right_vectors = right_vectors * signs[:, None]

    explained_variance = singular_values**2 / (array.shape[0] - 1)
    return PrincipalComponents(
        mean=mean,
        components=right_vectors[:n_components],
        explained_variance=explained_variance[:n_components],
    )


def standardize(data: npt.NDArray[Any]) -> npt.NDArray[np.floating]:
    """Centre each column on zero and scale it to unit variance.

    The variance is the population variance (``ddof=0``), which is what ``StandardScaler`` uses.

    **On constant columns.** A column with no variance would be divided by zero, giving ``inf`` or
    ``nan`` and poisoning the PCA downstream. Such a column is scaled by 1 instead, leaving it at
    its centred value of zero. The test for it is not ``variance == 0`` but scikit-learn's bound
    from Chan, Golub and LeVeque on the error of the two-pass variance algorithm, so that a column
    which is constant in exact arithmetic is still caught when floating-point noise leaves it merely
    very small.

    :param data: Array of shape ``(n_samples, n_features)``.
    :return: The standardized array.
    """
    array = np.asarray(data, dtype=float)
    mean = array.mean(axis=0)
    variance = array.var(axis=0)

    n_samples = array.shape[0]
    eps = np.finfo(np.float64).eps
    indistinguishable_from_constant = variance <= (
        n_samples * eps * variance + (n_samples * mean * eps) ** 2
    )

    scale = np.sqrt(variance)
    scale[indistinguishable_from_constant] = 1.0

    standardized: npt.NDArray[np.floating] = (array - mean) / scale
    return standardized


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
