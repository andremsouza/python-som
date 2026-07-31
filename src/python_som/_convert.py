"""The data-input port: whatever the caller passed, in, an ``ndarray`` out.

``np.asarray`` handles every input through the ``__array__`` protocol, so pandas, polars, pyarrow,
xarray and CuPy all work without this package importing any of them.

Small, and it stays a module because it is the one place that decides what "a dataset" means.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from typing import TypeAlias

    import numpy.typing as npt

    #: Anything the public methods accept as a dataset. Named rather than inlined so the promise has
    #: somewhere to live: anything implementing ``__array__`` works, which covers pandas, polars,
    #: pyarrow and xarray without importing any of them.
    DataLike: TypeAlias = npt.ArrayLike

__all__ = ["to_numpy"]


def to_numpy(data: DataLike) -> npt.NDArray[Any]:
    """Convert anything array-like to a NumPy array.

    Handles DataFrames, Series, lists, and any object implementing ``__array__``. Existing arrays
    pass through without a copy.

    :param data: Input data.
    :return: The data as a NumPy array.
    """
    return np.asarray(data)
