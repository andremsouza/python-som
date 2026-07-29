"""The functional core: pure functions over NumPy arrays.

Every function here takes all of its inputs explicitly and returns a value. Nothing in this package
reads instance state, performs I/O, or knows about pandas, tqdm, or any other library beyond NumPy.
That is not a stylistic preference: it is enforced, because ruff's ``TID251`` bans those imports
everywhere except the shell modules that exist to adapt them.

The shell around it is small by design:

- ``python_som._convert`` converts whatever the caller passed into an ``ndarray``. The only module
  that knows pandas exists.
- ``python_som._som`` holds the :class:`~python_som.SOM` class: validation, state, the training
  loops, and delegation to the functions here.

References:
Teuvo Kohonen, Essentials of the self-organizing map, Neural Networks 37 (2013) 52-65,
https://doi.org/10.1016/j.neunet.2012.09.018

O. J. Vrieze, Kohonen network, in: Artificial Neural Networks: An Introduction to ANN Theory and
Practice, Lecture Notes in Computer Science, vol. 931, Springer, 1995, pp. 83-100,
https://doi.org/10.1007/BFb0027024
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
    NEIGHBORHOOD_FUNCTIONS,
    SIGNED_NEIGHBORHOODS,
    axis_offsets,
    bubble,
    gaussian,
    mexican_hat,
    resolve,
    squared_grid_distance,
)

__all__ = [
    "NEIGHBORHOOD_FUNCTIONS",
    "SIGNED_NEIGHBORHOODS",
    "asymptotic_decay",
    "axis_offsets",
    "bubble",
    "euclidean_distance",
    "exponential_decay",
    "gaussian",
    "inverse_decay",
    "linear_decay",
    "mexican_hat",
    "resolve",
    "squared_grid_distance",
]
