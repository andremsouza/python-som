"""The functional core: pure functions over NumPy arrays.

Every function takes its inputs explicitly and returns a value. Nothing here reads instance state,
performs I/O, or imports anything but NumPy, which ruff's ``TID251`` enforces.

The shell is ``python_som._convert``, which turns whatever the caller passed into an ``ndarray``,
``python_som._som``, which holds the state and the training loops, and ``python_som._accelerate``,
which supplies the optional kernel.

References:
Teuvo Kohonen, Essentials of the self-organizing map, Neural Networks 37 (2013) 52-65,
https://doi.org/10.1016/j.neunet.2012.09.018

O. J. Vrieze, Kohonen network, in: Artificial Neural Networks, Lecture Notes in Computer Science,
vol. 931, Springer, 1995, pp. 83-100, https://doi.org/10.1007/BFb0027024
"""

from __future__ import annotations

from ._decay import (
    asymptotic_decay,
    exponential_decay,
    inverse_decay,
    linear_decay,
)
from ._distance import euclidean_distance
from ._neighborhood import (
    AXIS_PROFILES,
    NEIGHBORHOOD_FUNCTIONS,
    SIGNED_NEIGHBORHOODS,
    axis_matrix,
    axis_offsets,
    bubble,
    bubble_axis_profile,
    gaussian,
    gaussian_axis_profile,
    mexican_hat,
    resolve,
    resolve_axis_profile,
    squared_grid_distance,
)

__all__ = [
    "AXIS_PROFILES",
    "NEIGHBORHOOD_FUNCTIONS",
    "SIGNED_NEIGHBORHOODS",
    "asymptotic_decay",
    "axis_matrix",
    "axis_offsets",
    "bubble",
    "bubble_axis_profile",
    "euclidean_distance",
    "exponential_decay",
    "gaussian",
    "gaussian_axis_profile",
    "inverse_decay",
    "linear_decay",
    "mexican_hat",
    "resolve",
    "resolve_axis_profile",
    "squared_grid_distance",
]
