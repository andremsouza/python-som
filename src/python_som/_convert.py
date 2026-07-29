"""The data-input port: whatever the caller passed, in, an ``ndarray`` out.

This is the only module in the package that knows pandas exists, which is what lets ruff's
``TID251`` ban pandas everywhere else. Everything downstream of here works on ``np.ndarray`` only.

The pandas branch is strictly redundant. ``np.asarray`` already converts a DataFrame or a Series
through the ``__array__`` protocol, with results identical to ``.to_numpy()`` including for nullable
extension dtypes. It is kept so that removing the pandas dependency is its own reviewable change
rather than a side effect of the module split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from typing import TypeAlias

    #: Anything the public methods accept as a dataset. Anything implementing ``__array__`` works in
    #: practice, including polars, pyarrow and xarray objects; these are the types we promise.
    DataLike: TypeAlias = npt.NDArray[Any] | pd.DataFrame | pd.Series[Any] | list[Any]

__all__ = ["to_numpy"]


def to_numpy(data: DataLike) -> npt.NDArray[Any]:
    """Convert a DataFrame, Series, list or array to a NumPy array.

    :param data: Input data.
    :return: The data as a NumPy array.
    """
    if isinstance(data, pd.DataFrame | pd.Series):
        return data.to_numpy()
    return np.asarray(data)
