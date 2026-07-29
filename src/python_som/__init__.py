"""Python implementation of Kohonen's 2-D self-organizing map.

The public surface is :class:`SOM`, plus the decay, distance and neighborhood functions that can be
passed to it or used directly.

>>> import numpy as np, python_som
>>> data = np.random.default_rng(0).normal(size=(150, 4))
>>> som = python_som.SOM(x=10, y=10, input_len=4, random_seed=0)
>>> som.weight_initialization(mode="linear", data=data)
>>> error = som.train(data, n_iteration=100, mode="batch")

Reference:
Teuvo Kohonen, Essentials of the self-organizing map, Neural Networks 37 (2013) 52-65,
ISSN 0893-6080, https://doi.org/10.1016/j.neunet.2012.09.018
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
    bubble,
    gaussian,
    mexican_hat,
)
from ._som import SOM

__all__ = [
    "NEIGHBORHOOD_FUNCTIONS",
    "SIGNED_NEIGHBORHOODS",
    "SOM",
    "asymptotic_decay",
    "bubble",
    "euclidean_distance",
    "exponential_decay",
    "gaussian",
    "inverse_decay",
    "linear_decay",
    "mexican_hat",
]

__version__ = "0.3.0"

# Backwards-compatible aliases. Before 0.3.0 these functions lived in this module under
# underscore-prefixed names; they were private by convention but reachable, and the README listed
# them. Keep them working rather than breaking imports silently.
_asymptotic_decay = asymptotic_decay
_linear_decay = linear_decay
_exponential_decay = exponential_decay
_inverse_decay = inverse_decay
_euclidean_distance = euclidean_distance
