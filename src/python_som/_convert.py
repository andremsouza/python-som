"""The data-input port: whatever the caller passed, in, an ``ndarray`` out.

Through 0.3.0 this module special-cased pandas, testing ``isinstance(data, pd.DataFrame |
pd.Series)`` before calling ``.to_numpy()``. That was the only use of pandas, and it was redundant:
``np.asarray`` already converts both through the ``__array__`` protocol, with identical results
including for nullable extension dtypes, which convert to ``float64`` with ``nan`` either way.

Dropping the special case removes a required dependency and **widens** what the package
accepts, because ``__array__`` is a protocol rather than a library. polars, pyarrow, xarray and CuPy
objects all implement it and now work without python-som knowing any of them exist. Fewer
dependencies and more capability at once, which is the argument for a port rather than an adapter
per library.

The module stays, small as it is, because it is the one place that decides what "a dataset" means.
When that decision needs to change -- a dtype policy, a shape check, an explicit error for ragged
input -- there is one place to change it, and the core keeps receiving ``ndarray`` and nothing else.
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
